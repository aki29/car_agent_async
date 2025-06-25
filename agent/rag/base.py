from pathlib import Path
from typing import List
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain.embeddings.base import Embeddings


class BaseRAG:
    domain: str = "base"

    def __init__(self, embed_model: Embeddings, data_dir: Path, store_dir: Path):
        self.embed_model = embed_model
        self.data_dir = data_dir
        self.store_dir = store_dir / self.domain
        self.vs: Chroma | None = None

    # --------- public API ---------
    async def ainit(self):
        if not self.store_dir.exists():
            self._build_vectorstore()
        else:
            self.vs = Chroma(
                embedding_function=self.embed_model, persist_directory=str(self.store_dir)
            )

    async def aretrieve(self, query: str, k: int = 6):
        if self.vs is None:
            await self.ainit()

        retriever = self.vs.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "score_threshold": 0.8},
        )

        if hasattr(retriever, "ainvoke"):
            return await retriever.ainvoke(query)
        else:  #
            return await retriever.aget_relevant_documents(query)

    # --------- helpers ---------
    def _build_vectorstore(self):
        docs = self._load_docs()
        self.vs = Chroma.from_documents(
            docs, embedding=self.embed_model, persist_directory=str(self.store_dir)
        )
        if hasattr(self.vs, "persist"):
            self.vs.persist()

    def _load_docs(self) -> List[Document]:
        raise NotImplementedError
