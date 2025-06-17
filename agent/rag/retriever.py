"""Async helper to build a Chroma retriever from local docs."""
import asyncio
from functools import partial
from pathlib import Path
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOC_PATH = Path(__file__).parent.parent.parent / "docs"
VECTOR_PATH = Path(__file__).parent.parent.parent / "data" / "vector_db"
VECTOR_PATH.mkdir(parents=True, exist_ok=True)

def _load_documents():
    docs = []
    for p in DOC_PATH.rglob("*.txt"):
        docs.extend(TextLoader(str(p)).load())
    for p in DOC_PATH.rglob("*.pdf"):
        docs.extend(PyMuPDFLoader(str(p)).load())
    return docs

async def build_retriever_async(embedding_model: Embeddings) -> Optional[Chroma]:
    docs = await asyncio.to_thread(_load_documents)
    if not docs:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=120
    )
    splits = await asyncio.to_thread(splitter.split_documents, docs)

    vectorstore = await asyncio.to_thread(
        partial(
            Chroma.from_documents,
            documents=splits,
            embedding=embedding_model,
            persist_directory=str(VECTOR_PATH),
            collection_name="ctk_rag",
        )
    )
    return vectorstore.as_retriever(
        search_type="mmr", search_kwargs={"k": 4, "score_threshold": 0.7}
    )
