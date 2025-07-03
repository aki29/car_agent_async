# ---------------- agent/rag/manual_rag.py ---------------------
from pathlib import Path
import json
from typing import List
from pydantic import BaseModel
from langchain.schema import Document
from .base import BaseRAG
import time

class ManualFAQ(BaseModel):
    type: str  # always "車主手冊"
    title: str
    content: str


class ManualRAG(BaseRAG):
    """車主手冊 FAQ 檢索"""

    domain = "manual"

    def _load_docs(self) -> List[Document]:
        start = time.time()
        faq_file = self.data_dir / "manual.jsonl"
        docs: List[Document] = []
        with faq_file.open(encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                    faq = ManualFAQ(**raw)
                except Exception as e:
                    raise ValueError(f"[manual.jsonl] line {lineno} parse error: {e}") from e

                docs.append(
                    Document(
                        page_content=f"{faq.title}\n{faq.content}",
                        metadata=faq.dict(),
                    )
                )
        print(f"[{self.domain}] loaded {len(docs)} docs in {time.time() - start:.2f}s")
        return docs

    async def aretrieve(self, query: str, k: int = 6):
        return await super().aretrieve(query, k)
