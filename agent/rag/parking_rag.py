from pathlib import Path
import json
from typing import List
from pydantic import BaseModel, Field
from langchain.schema import Document
from .base import BaseRAG
import time

class ParkingLot(BaseModel):
    type: str  # 停車場
    name: str
    address: str
    description: str
    lat: float | None = Field(default=None)
    long: float | None = Field(default=None)


class ParkingRAG(BaseRAG):
    domain = "parking"

    def _load_docs(self) -> List[Document]:
        start = time.time()
        fp = self.data_dir / "parking.jsonl"
        docs: List[Document] = []
        with fp.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                lot = ParkingLot(**json.loads(line))
                docs.append(
                    Document(
                        page_content=f"{lot.name}\n{lot.description}\n{lot.address}",
                        metadata=lot.dict(),
                    )
                )
        print(f"[{self.domain}] loaded {len(docs)} docs in {time.time() - start:.2f}s")
        return docs

    async def aretrieve(self, query: str, k: int = 6, user_loc=None):
        return await super().aretrieve(query, k)
