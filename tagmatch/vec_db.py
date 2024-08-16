import uuid
from typing import Any, Dict, List, Optional

import numpy as np
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import (CollectionInfo, Distance,
                                       FieldCondition, Filter, MatchValue,
                                       PointStruct, VectorParams, ScoredPoint)


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
        self.vector_size = vector_size

        if not self.collection_exists():
            self._create_collection()

    def collection_exists(self) -> bool:
        try:
            _ = self.client.get_collection(self.collection)
            return True
        except UnexpectedResponse:
            return False

    def _create_collection(self):
        dist = Distance.COSINE if self.distance == "cosine" else Distance.EUCLID
        self.client.create_collection(self.collection,
                                      vectors_config=VectorParams(size=self.vector_size, distance=dist))
        self.client.create_collection(self.manual_collection, vectors_config=VectorParams(size=self.vector_size, distance=dist))

    def remove_collection(self):
        self.client.delete_collection(self.collection)

    def find_closest(self, vector: np.ndarray, k: int) -> List[str]:
        vec_list: List[float] = vector.tolist()
        # query_filter = Filter(must=[FieldCondition(key="is_accepted_tag", match=MatchValue(value=True))])
        query_filter = None
        res = self.client.search(self.collection, query_vector=vec_list, limit=k, query_filter=query_filter)
        return res

    def find_manual(self, vector: np.ndarray, name: str) -> Optional[ScoredPoint]:
        vec_list: List[float] = vector.tolist()
        query_filter = Filter(must=FieldCondition(key='name', match=MatchValue(value=name)))
        res = self.client.search(self.manual_collection, query_vector=vec_list, query_filter=query_filter)
        if res:
            return res[0]
        else:
            return None

    def update_manual_by_name(self, name: str, vector: np.ndarray, manual_tags: list):
        res = self.find_manual(vector, name)
        if res:
            print(res.id)
            self.store(vector, {"name": res.payload["name"], "manual": manual_tags}, res.id, True)
        else:
            self.store(vector=vector, payload={"name": name, "manual": manual_tags}, manual=True)

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
