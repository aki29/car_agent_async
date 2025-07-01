from pathlib import Path
import json
from typing import List
from pydantic import BaseModel, Field
from langchain.schema import Document
from .base import BaseRAG


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
        return docs

    async def aretrieve(self, query: str, k: int = 6, user_loc=None):
        return await super().aretrieve(query, k)
