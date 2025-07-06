import asyncio
import sqlite3
import os
import pickle
import base64
from typing import Optional, Any
from concurrent.futures import ThreadPoolExecutor
from langchain_core.caches import BaseCache
import hashlib

def _normalize_prompt(prompt: str) -> str:
    return " ".join(prompt.strip().split())

def _prompt_key(prompt: str, llm_string: str) -> str:
    prompt = _normalize_prompt(prompt)
    h = hashlib.sha256()
    h.update(prompt.encode())
    h.update(llm_string.encode())
    return h.hexdigest()

# def _prompt_key(prompt: str, llm_string: str) -> str:
#     h = hashlib.sha256()
#     h.update(prompt.encode())
#     h.update(llm_string.encode())
#     return h.hexdigest()


class AsyncSQLiteCache(BaseCache):
    def __init__(self, db_path: str = ".cache/langchain_async.db", *, max_workers: int = 1):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._ensure_table()

    # ---------------- internal loop helper ----------------
    @staticmethod
    def _run(coro):
        try:
            loop = asyncio.get_running_loop()
            return loop.run_until_complete(coro)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()

    # ---------------- BaseCache sync API ----------------
    # def lookup(self, prompt: str, llm_string: str) -> Optional[Any]:
    #     return self._run(self.alookup(prompt, llm_string))

    def lookup(self, prompt: str, llm_string: str) -> Optional[Any]:
        key = _prompt_key(prompt, llm_string)
        def _query():
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute(
                    "SELECT llm_output FROM langchain_cache WHERE prompt_key = ?",
                    (key,),
                )
                row = cur.fetchone()
                if not row:
                    return None
                try:
                    return pickle.loads(base64.b64decode(row[0]))
                except Exception:
                    return None
        return self._executor.submit(_query).result()

    # def lookup(self, prompt: str, llm_string: str) -> Optional[Any]:
    #     key = _prompt_key(prompt, llm_string)
    #     with sqlite3.connect(self.db_path) as conn:
    #         cur = conn.execute(
    #             "SELECT llm_output FROM langchain_cache WHERE prompt_key = ?",
    #             (key,),
    #         )
    #         row = cur.fetchone()
    #         if not row:
    #             return None
    #         try:
    #             return pickle.loads(base64.b64decode(row[0]))
    #         except Exception:
    #             return None

    def update(self, prompt: str, llm_string: str, result: Any) -> None:
        self._run(self.aupdate(prompt, llm_string, result))

    def clear(self) -> None:
        self._run(self.aclear())

    # ---------------- Async API ----------------
    async def alookup(self, prompt: str, llm_string: str) -> Optional[Any]:
        key = _prompt_key(prompt, llm_string)

        def _query():
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute(
                    "SELECT llm_output FROM langchain_cache WHERE prompt_key = ?",
                    (key,),
                )
                row = cur.fetchone()
                if not row:
                    return None
                try:
                    return pickle.loads(base64.b64decode(row[0]))
                except Exception:
                    return None

        return await asyncio.get_running_loop().run_in_executor(self._executor, _query)

    async def aupdate(self, prompt: str, llm_string: str, result: Any) -> None:
        key = _prompt_key(prompt, llm_string)
        payload = base64.b64encode(pickle.dumps(result)).decode()

        def _write():
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO langchain_cache (prompt_key, llm_output) VALUES (?, ?)",
                    (key, payload),
                )
                conn.commit()

        await asyncio.get_running_loop().run_in_executor(self._executor, _write)

    async def aclear(self) -> None:
        def _truncate():
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM langchain_cache")
                conn.commit()

        await asyncio.get_running_loop().run_in_executor(self._executor, _truncate)

    # ---------------- internal ----------------
    def _ensure_table(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS langchain_cache (prompt_key TEXT PRIMARY KEY, llm_output TEXT)"
            )
            conn.commit()

    def close(self) -> None:
        self._executor.shutdown(wait=False)
