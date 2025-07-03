from pathlib import Path
from typing import List, Optional
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
import faiss
import asyncio
from concurrent.futures import ThreadPoolExecutor


class BaseRAG:
    domain: str = "base"

    def __init__(self, embed_model: Embeddings, data_dir: Path, store_dir: Path):
        self.embed_model = embed_model
        self.data_dir = data_dir
        self.store_dir = store_dir / self.domain
        self.faiss_index_path = self.store_dir / (self.domain + ".faiss")
        self.faiss_doc_path = self.store_dir / (self.domain + ".pkl")
        self.vs: FAISS | None = None
        self.executor = ThreadPoolExecutor()

    # --------- public API ---------
    async def ainit(self):

        if self.faiss_index_path.exists() and self.faiss_doc_path.exists():
            self.vs = await asyncio.get_event_loop().run_in_executor(
                self.executor, self._load_vectorstore
            )
        else:
            self._build_vectorstore()

    async def aretrieve(self, query: str, k: int = 6):
        if self.vs is None:
            await self.ainit()
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, lambda: self.vs.similarity_search(query, k=k)
        )

    # --------- helpers ---------
    def _build_vectorstore(self):

        docs = self._load_docs()
        self.vs = FAISS.from_documents(docs, self.embed_model)
        res = faiss.StandardGpuResources()
        self.vs.index = faiss.index_cpu_to_gpu(res, 0, self.vs.index)

        cpu_index = faiss.index_gpu_to_cpu(self.vs.index)
        self.vs.index = cpu_index

        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.vs.save_local(str(self.store_dir), self.domain)

    def _load_vectorstore(self):
        vs = FAISS.load_local(
            str(self.store_dir),
            self.embed_model,
            index_name=self.domain,
            allow_dangerous_deserialization=True,
        )
        res = faiss.StandardGpuResources()
        vs.index = faiss.index_cpu_to_gpu(res, 0, vs.index)
        return vs

    def _load_docs(self) -> List[Document]:
        raise NotImplementedError
