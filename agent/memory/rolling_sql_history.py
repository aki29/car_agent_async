from __future__ import annotations
import asyncio
from typing import List
from langchain_community.chat_message_histories.sql import SQLChatMessageHistory
from sqlalchemy import delete, select


class RollingSQLHistory(SQLChatMessageHistory):
    def __init__(self, *, window_size: int = 40, **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size

    def get_messages(self) -> List:
        all_msgs: List = super().get_messages()
        return all_msgs[-self.window_size :]

    async def aget_messages(self) -> List:
        return await asyncio.to_thread(self.get_messages)

    def add_message(self, message) -> None:
        super().add_message(message)
        with self.session_maker() as s:
            subq = (
                select(self.sql_model_class.id)
                .filter_by(session_id=self.session_id)
                .order_by(self.sql_model_class.id.desc())
                .offset(self.window_size)
            )
            s.execute(delete(self.sql_model_class).where(self.sql_model_class.id.in_(subq)))
            s.commit()

    async def aadd_message(self, message) -> None:
        await asyncio.to_thread(self.add_message, message)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False
