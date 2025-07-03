from pathlib import Path
import json
from typing import List
from pydantic import BaseModel, Field
from langchain.schema import Document
from .base import BaseRAG
import time

class POI(BaseModel):
    type: str
    name: str
    address: str
    description: str
    lat: float | None = Field(default=None)
    long: float | None = Field(default=None)


class POIRAG(BaseRAG):
    domain = "poi"

    def _load_docs(self) -> List[Document]:
        start = time.time()
        fp = self.data_dir / "poi.jsonl"
        docs: List[Document] = []
        with fp.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                poi = POI(**json.loads(line))
                docs.append(
                    Document(
                        page_content=f"{poi.name}\n{poi.description}\n{poi.address}",
                        metadata=poi.dict(),
                    )
                )
        print(f"[{self.domain}] loaded {len(docs)} docs in {time.time() - start:.2f}s")
        return docs

    async def aretrieve(self, query: str, k: int = 6, user_loc=None):
        return await super().aretrieve(query, k)
