import logging
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import (CollectionInfo, Distance,
                                       FieldCondition, Filter, MatchValue,
                                       PointStruct, VectorParams, ScoredPoint)
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error


class Embedder:

    def __init__(self, model_name: str, cache_dir: str):
        self.embedding_model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
        self.embedding_dim: int = list(self.embedding_model.embed("Test for dims"))[0].shape[0]

    def embed(self, text: str) -> np.ndarray:
        emb_generatpor = self.embedding_model.embed(text)
        return list(emb_generatpor)[0]


class VecDB:
    _ALLOWED_DISTANCES = ("cosine", "euclidean")

    def __init__(self, host: str, port: int, collection: str, vector_size: int, distance: str = "cosine"):
        if distance not in self._ALLOWED_DISTANCES:
            raise ValueError(f"Distance {distance} not allowed. Allowed distances are {self._ALLOWED_DISTANCES}")

        self.distance = distance
        self.client = QdrantClient(host, port=int(port))
        self.collection = collection
        self.manual_collection = "manual_" + self.collection
        self.reduced_collection = "reduced_" + self.collection
        self.num_components = None  # Number of components for PCA
        self.pca = None
        self.vector_size = vector_size

        if not self.collection_exists():
            self._create_collection()

        if self.collection_exists("reduced_"):
            self.num_components = self.client.get_collection(self.reduced_collection).model_dump()["config"]["params"]["vectors"]["size"]
            self.populate_pca(self.num_components)

    def collection_exists(self, collection_name: str = None) -> bool:
        if collection_name == "manual_":
            name = self.manual_collection
        elif collection_name == "reduced_":
            name = self.reduced_collection
        else:
            name = self.collection
        try:

            _ = self.client.get_collection(name)
            return True
        except UnexpectedResponse:
            return False

    def _create_collection(self, num_components: int = None):
        '''If number of components is provided it will create a 'reduced' collection for PCA'''

        dist = Distance.COSINE if self.distance == "cosine" else Distance.EUCLID

        if num_components is None:
            self.client.create_collection(self.collection,
                                          vectors_config=VectorParams(size=self.vector_size, distance=dist))
            self.client.create_collection(self.manual_collection, vectors_config=VectorParams(size=self.vector_size, distance=dist))
        else:
            self.client.create_collection(self.reduced_collection,
                                          vectors_config=VectorParams(size=num_components, distance=dist))

    def remove_collection(self, collection_name: str = None):
        if collection_name == "reduced_":
            self.client.delete_collection(self.reduced_collection)
        else:
            self.client.delete_collection(self.collection)
            self.client.delete_collection(self.manual_collection)

    def find_closest(self, query: str, vector: np.ndarray, k: int, reduced: bool = False) -> List[ScoredPoint]:
        vec_list: List[float] = vector.tolist()
        # query_filter = Filter(must=[FieldCondition(key="is_accepted_tag", match=MatchValue(value=True))])
        query_filter = None
        # search of manually defined tag matches and return them if found
        manual_res = self.find_manual(vector, query)
        if manual_res:
            return [ScoredPoint(id=name, version=0, payload={"name": name}, score=1) for name in manual_res.payload['manual']]
        elif not reduced:
            res = self.client.search(self.collection, query_vector=vec_list, limit=k, query_filter=query_filter)
        else:
            reduced_vector = self.pca.transform(vector.reshape(1, -1)).reshape((-1))
            res = self.client.search(self.reduced_collection, query_vector=reduced_vector.tolist(), limit=k, query_filter=query_filter)
        return res

    def find_manual(self, vector: np.ndarray, name: str) -> Optional[ScoredPoint]:
        vec_list: List[float] = vector.tolist()
        query_filter = Filter(must=FieldCondition(key='name', match=MatchValue(value=name)))
        res = self.client.search(self.manual_collection, query_vector=vec_list, query_filter=query_filter)
        if res:
            return res[0]
        else:
            return None

    def update_manual_by_name(self, name: str, vector: np.ndarray, manual_tags: Optional[List[str]]):
        res = self.find_manual(vector, name)
        if res and manual_tags:
            print(res.id)
            self.store(vector, {"name": res.payload["name"], "manual": manual_tags}, res.id, True)
        elif res:
            self.client.delete(
                collection_name=self.manual_collection,
                points_selector=[res.id]
            )
        elif manual_tags:
            self.store(vector=vector, payload={"name": name, "manual": manual_tags}, manual=True)
        return manual_tags

    def store(self, vector: np.ndarray, payload: Dict[str, Any], idx: Optional[int] = None, manual: bool = False) -> bool:
        vec_list: List[float] = vector.tolist()
        rnd_id = idx if idx is not None else uuid.uuid4().int & (1 << 64) - 1
        try:
            collection = self.manual_collection if manual else self.collection
            self.client.upsert(collection, points=[PointStruct(id=rnd_id, vector=vec_list, payload=payload)])
            return True
        except Exception:
            return False

    def get_item_count(self) -> int:
        collection_info: CollectionInfo = self.client.get_collection(self.collection)
        nb_points = collection_info.points_count

        if nb_points is None:
            return -1

        return nb_points

    def populate_pca(self, num_components: int):
        '''Populates 'reduced_' collection by reduced vectors using PCA technique'''

        self.num_components = num_components
        response = self.client.scroll(collection_name=self.collection, with_vectors=True, limit=self.get_item_count())
        vectors = [point.vector for point in response[0]]
        vector_ids = [point.id for point in response[0]]
        vector_names = [point.payload["name"] for point in response[0]]
        vectors = np.array(vectors)

        # Apply PCA to reduce dimensionality
        self.pca = PCA(n_components=self.num_components)  # Adjust n_components as needed
        reduced_vectors = self.pca.fit_transform(vectors)
        logging.info(f"reduced_vectors.shape: {reduced_vectors.shape}")
        explained_variance = sum(self.pca.explained_variance_ratio_)

        reconstructed_vectors = self.pca.inverse_transform(reduced_vectors)
        reconstruction_error = mean_squared_error(vectors, reconstructed_vectors)

        # Delete existing 'reduced_' collection and create a new collection for reduced vectors
        if self.collection_exists("reduced_"):
            self.remove_collection("reduced_")
        self._create_collection(self.num_components)

        # Upsert reduced vectors to the new collection
        self.client.upsert(
            collection_name=self.reduced_collection,
            points=[
                PointStruct(id=vector_id, payload={"name": vector_name}, vector=reduced_vector.tolist())
                for vector_id, vector_name, reduced_vector in zip(vector_ids, vector_names, reduced_vectors)
            ]
        )

        return f'Reduced vectors for PCA with {self.num_components} components stored in collection: "reduced_"', explained_variance, reconstruction_error

