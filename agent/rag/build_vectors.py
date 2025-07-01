# import argparse, asyncio
# from pathlib import Path
# from langchain_ollama import OllamaEmbeddings
# from . import RAGManager
# import shutil
# from dotenv import load_dotenv, find_dotenv
# import os

# load_dotenv(find_dotenv())


# async def _build(domain: str, force: bool = False):
#     embed = OllamaEmbeddings(
#         model=os.getenv("EMBEDDINGS_MODEL_NAME", "nomic-embed-text:latest"),
#         base_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
#         num_thread=2,
#         keep_alive=-1,
#     )
#     rm = RAGManager(embed)

#     if domain in ("poi", "all"):
#         poi_dir = rm.poi.store_dir  #

#         if force and poi_dir.exists():
#             print("remove old POI store ...")
#             shutil.rmtree(poi_dir)

#         if force:
#             print("(Re)building POI vectore ...")
#             rm.poi._build_vectorstore()
#         else:
#             await rm.poi.ainit()
#     print("Done!")


# def main():
#     parser = argparse.ArgumentParser(description="Build RAG vectorstores")
#     parser.add_argument(
#         "--domain", default="poi", choices=["poi", "all"], help="The RAG domain to rebuild"
#     )
#     parser.add_argument(
#         "--force", action="store_true", help="Force rebuild (delete existing directory first)"
#     )
#     args = parser.parse_args()
#     asyncio.run(_build(args.domain, force=args.force))


# if __name__ == "__main__":
#     main()

import argparse, asyncio, shutil, os
from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv, find_dotenv
from . import RAGManager

load_dotenv(find_dotenv())

DOMAINS = ["poi", "manual", "parking", "food"]


async def _build(domain: str, force: bool):
    embed = OllamaEmbeddings(
        model=os.getenv("EMBEDDINGS_MODEL_NAME", "nomic-embed-text:latest"),
        base_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        num_thread=6,
        keep_alive=-1,
    )
    rm = RAGManager(embed)

    targets = DOMAINS if domain == "all" else [domain]
    for d in targets:
        rag = getattr(rm, d)
        vec_dir = rag.store_dir
        if force and vec_dir.exists():
            print(f"[rebuild] removing {vec_dir} …")
            shutil.rmtree(vec_dir)
        if force:
            print(f"building {d} …")
            rag._build_vectorstore()
        else:
            await rag.ainit()
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="all", choices=DOMAINS + ["all"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    asyncio.run(_build(args.domain, args.force))