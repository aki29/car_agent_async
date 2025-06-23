from pathlib import Path
from .poi_rag import POIRAG


class RAGManager:
    def __init__(
        self,
        embed,
        base_data=Path(__file__).parent / "rawdata",
        store_dir=Path(__file__).parent.parent / "data" / "rag",
    ):
        self.poi = POIRAG(embed, base_data, store_dir)

    async def ainit(self):
        await self.poi.ainit()

    async def query_poi(self, query: str, user_loc=None):
        return await self.poi.aretrieve(query, k=6, user_loc=user_loc)


rag_manager: RAGManager | None = None
