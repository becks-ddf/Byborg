from pydantic import BaseModel
from typing import List


class Query(BaseModel):
    query: str
    k: int = 5

class BatchQuery(BaseModel):
    queries: List[Query]
