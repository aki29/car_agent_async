import argparse, asyncio
from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from . import RAGManager
import shutil


async def _build(domain: str, force: bool = False):
    embed = OllamaEmbeddings(
        model="nomic-embed-text:latest",
        base_url="http://localhost:11434",
        num_thread=2,
        keep_alive=-1,
    )
    rm = RAGManager(embed)

    if domain in ("poi", "all"):
        poi_dir = rm.poi.store_dir  #

        if force and poi_dir.exists():
            print("remove old POI store ...")
            shutil.rmtree(poi_dir)

        if force:
            print("(Re)building POI vectore ...")
            rm.poi._build_vectorstore()
        else:
            await rm.poi.ainit()
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Build RAG vectorstores")
    parser.add_argument(
        "--domain", default="poi", choices=["poi", "all"], help="The RAG domain to rebuild"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force rebuild (delete existing directory first)"
    )
    args = parser.parse_args()
    asyncio.run(_build(args.domain, force=args.force))


if __name__ == "__main__":
    main()
