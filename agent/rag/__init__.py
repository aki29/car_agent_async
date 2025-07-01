# from pathlib import Path
# from .poi_rag import POIRAG


# class RAGManager:
#     def __init__(
#         self,
#         embed,
#         base_data=Path(__file__).parent / "rawdata",
#         store_dir=Path(__file__).parent.parent / "data" / "rag",
#     ):
#         self.poi = POIRAG(embed, base_data, store_dir)

#     async def ainit(self):
#         await self.poi.ainit()

#     async def query_poi(self, query: str, user_loc=None):
#         return await self.poi.aretrieve(query, k=6, user_loc=user_loc)

from pathlib import Path
from .poi_rag import POIRAG
from .manual_rag import ManualRAG
from .parking_rag import ParkingRAG
from .food_rag import FoodRAG


class RAGManager:
    def __init__(
        self,
        embed,
        base_data: Path = Path(__file__).parent / "rawdata",
        store_dir: Path = Path(__file__).parent.parent / "data" / "rag",
    ):
        self.poi = POIRAG(embed, base_data, store_dir)
        self.manual = ManualRAG(embed, base_data, store_dir)
        self.parking = ParkingRAG(embed, base_data, store_dir)
        self.food = FoodRAG(embed, base_data, store_dir)

    async def ainit(self):
        await self.poi.ainit()
        await self.manual.ainit()
        await self.parking.ainit()
        await self.food.ainit()

    # ---------- query helpers ----------
    async def query_poi(self, q: str, user_loc=None):
        return await self.poi.aretrieve(q, user_loc=user_loc)

    async def query_manual(self, q: str):
        return await self.manual.aretrieve(q)

    async def query_parking(self, q: str, user_loc=None):
        return await self.parking.aretrieve(q, user_loc=user_loc)

    async def query_food(self, q: str, user_loc=None):
        return await self.food.aretrieve(q, user_loc=user_loc)


rag_manager: RAGManager | None = None
