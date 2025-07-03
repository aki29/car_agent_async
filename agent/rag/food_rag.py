from pathlib import Path
import json
from typing import List
from pydantic import BaseModel, Field
from langchain.schema import Document
from .base import BaseRAG
import time

class Eatery(BaseModel):
    type: str  # 餐廳
    name: str
    address: str
    description: str
    rating: float | None = Field(default=None)
    review_count: int | None = Field(default=None)


class FoodRAG(BaseRAG):
    domain = "food"

    def _load_docs(self) -> List[Document]:
        start = time.time()
        fp = self.data_dir / "food.jsonl"
        docs: List[Document] = []
        with fp.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                eatery = Eatery(**json.loads(line))
                docs.append(
                    Document(
                        page_content=f"{eatery.name}\n{eatery.description}\n{eatery.address}",
                        metadata=eatery.dict(),
                    )
                )
        print(f"[{self.domain}] loaded {len(docs)} docs in {time.time() - start:.2f}s")
        return docs

    async def aretrieve(self, query: str, k: int = 6, user_loc=None):
        return await super().aretrieve(query, k)
