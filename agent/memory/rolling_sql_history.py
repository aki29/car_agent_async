from __future__ import annotations
import asyncio, threading, json
from typing import List
from langchain_community.chat_message_histories.sql import SQLChatMessageHistory
from langchain.schema import messages_from_dict
from sqlalchemy import delete, select


class RollingSQLHistory(SQLChatMessageHistory):
    def __init__(self, *, window_size: int = 40, **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size
        self._cache: List | None = None
        self._lock = threading.Lock()
        self._alock = asyncio.Lock()

    def _load_cache(self):
        def _row_to_msg(row):
            if hasattr(row, "to_message"):
                return row.to_message()
            if hasattr(row, "to_langchain_message"):
                return row.to_langchain_message()
            if hasattr(row, "message"):
                msg_dict = json.loads(row.message)
                return messages_from_dict([msg_dict])[0]
            raise AttributeError("Un-recognized SQL row schema")

        with self.session_maker() as s:
            rows = (
                s.query(self.sql_model_class)
                .filter_by(session_id=self.session_id)
                .order_by(self.sql_model_class.id.desc())
                .limit(self.window_size)
                .all()
            )
            self._cache = [_row_to_msg(r) for r in reversed(rows)] if rows else []

    def get_messages(self) -> List:
        with self._lock:
            if self._cache is None:
                self._load_cache()
            return list(self._cache)

    async def aget_messages(self) -> List:
        async with self._alock:
            if self._cache is None:
                await asyncio.to_thread(self._load_cache)
            return list(self._cache)

    def add_message(self, message) -> None:
        with self._lock:
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
        async with self._alock:
            if self._cache is None:
                await asyncio.to_thread(self._load_cache)
            self._cache.append(message)
            if len(self._cache) > self.window_size:
                self._cache = self._cache[-self.window_size :]
        await asyncio.to_thread(self.add_message, message)

    def add_messages(self, messages) -> None:
        for m in messages:
            self.add_message(m)

    async def aadd_messages(self, messages) -> None:
        for m in messages:
            await self.aadd_message(m)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False
