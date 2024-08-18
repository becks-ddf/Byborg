import io
import logging
import os
import threading
from typing import List, Tuple, Optional
from query import Query, BatchQuery

import pandas as pd
from fastapi import (BackgroundTasks, FastAPI, File, HTTPException, Request,
                     UploadFile)
from fastapi.responses import JSONResponse
from pydantic_settings import BaseSettings

from tagmatch.fuzzysearcher import FuzzyMatcher
from tagmatch.logging_config import setup_logging
from tagmatch.vec_db import Embedder, VecDB

if not os.path.exists("./data"):
    os.makedirs("./data")

setup_logging(file_path="./data/service.log")


class Settings(BaseSettings):
    model_name: str
    cache_dir: str
    qdrant_host: str
    qdrant_port: int
    qdrant_collection: str

    class Config:
        env_file = ".env"


settings = Settings()

app = FastAPI()
logger = logging.getLogger("fastapi")
# In-memory storage for tags
app.names_storage: List[str] = []

# Placeholder for the semantic search components
embedder = Embedder(model_name=settings.model_name, cache_dir=settings.cache_dir)
vec_db = VecDB(host=settings.qdrant_host,
               port=settings.qdrant_port,
               collection=settings.qdrant_collection,
               vector_size=embedder.embedding_dim)
app.fuzzy_matcher = FuzzyMatcher([])

# Flag to track background task status
task_running = threading.Event()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request: {request.method} {request.url}")

    if request.method in ["POST", "PUT", "PATCH"]:
        body = await request.body()
        logger.info(f"Request Body: {body.decode('utf-8')}")

    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response


@app.post("/create_reduced_collection/")
async def create_reduced_collection(num_components: int):
    return vec_db.populate_pca(num_components)


@app.post("/upload-csv/")
async def upload_csv(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if task_running.is_set():
        raise HTTPException(status_code=400, detail="A task is already running. Please try again later.")

    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400,
                            detail="Invalid file format. Please upload a CSV file (needs to env with '.csv'.")

    content = await file.read()
    df = pd.read_csv(io.BytesIO(content), sep=None, header=0)

    if 'name' not in df.columns:
        raise HTTPException(status_code=400, detail="CSV file must contain a 'name' column.")

    names_storage = df['name'].dropna().unique().tolist()

    if len(names_storage) == 0:
        raise HTTPException(status_code=400, detail="No names found in the CSV file.")

    # Return error if the collection is already existing (and it's populated)
    if vec_db.collection_exists():
        raise HTTPException(status_code=400, detail="Collection already exists. Please delete the collection first.")
    else:
        vec_db._create_collection()

    # Set the flag to indicate that a task is running
    task_running.set()

    # Add background task to process the CSV file
    background_tasks.add_task(process_csv, names_storage)

    return {"message": "File accepted for processing. Names will be extracted and stored in the background."}


def process_csv(names_storage: List[str]):
    try:
        # Store embedded vectors for semantic search
        for name in names_storage:
            vector = embedder.embed(name)
            vec_db.store(vector, {"name": name})

        app.names_storage = names_storage
        app.fuzzy_matcher = FuzzyMatcher(app.names_storage)
    finally:
        # Clear the flag to indicate that the task has completed
        task_running.clear()


@app.post("/update_manual/")
async def update_manual(query: str, manual_tags: Optional[List[str]]):
    res = vec_db.update_manual_by_name(query, embedder.embed(query), manual_tags)
    return {"message": f"Manual tags {res} for query {query} has been updated "}


@app.delete("/purge/")
async def delete_collection():
    app.names_storage = []
    vec_db.remove_collection("reduced_")
    vec_db.remove_collection()
    return {"message": f"DB deleted successfully."}


@app.get("/search/")
async def search(query: str, k: int = 5, reduced: bool = False):
    if not query:
        raise HTTPException(status_code=400, detail="Query parameter is required.")

    # Fuzzy search
    fuzzy_matches = app.fuzzy_matcher.get_top_k_matches(query, k)
    logging.info(f"fuzzy_matches: {fuzzy_matches}")
    # Semantic search
    query_vector = embedder.embed(query)
    try:
        semantic_matches = vec_db.find_closest(query, query_vector, k, reduced)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    # Formatting the response
    semantic_results = [{"name": match.payload["name"], "score": match.score} for match in semantic_matches]
    typo_results = [{"name": match["matched"], "score": match["score"]} for match in fuzzy_matches]

    response = {"match": {"semantic": semantic_results, "typo": typo_results}}
    logging.info(f"response: {response}")
    return JSONResponse(content=response)



@app.post("/search/")
async def search2(query: Query):
    response = await search(query.query, query.k)
    return response


@app.post("/batch/")
async def batch(queries: List[Query]):
    responses = [await search2(query) for query in queries]
    return responses


@app.get("/task-status/")
async def task_status():
    if task_running.is_set():
        return {"status": "running", "processed_items": vec_db.get_item_count()}
    else:
        return {"status": "finished", "nb_items_stored": vec_db.get_item_count()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
