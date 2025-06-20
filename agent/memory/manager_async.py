import os
import aiosqlite
from datetime import datetime
from pathlib import Path

_MEMORY_L1: dict[str, dict] = {}
_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "ctk_user.sqlite3"
DB_PATH = Path(os.environ.get("CTK_USER_DB_PATH", _DEFAULT_PATH))

DB_PATH.parent.mkdir(parents=True, exist_ok=True)
if not DB_PATH.exists():
    DB_PATH.touch()

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS chat_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    role TEXT,
    content TEXT,
    timestamp TEXT
);
CREATE TABLE IF NOT EXISTS user_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    memory_key TEXT,
    memory_value TEXT,
    updated_at TEXT,
    UNIQUE(user_id, memory_key)
);
"""


async def init_db() -> None:
    """Initialise database file and schema if missing."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript(CREATE_SQL)
        await db.commit()


# async def append_chat(user_id: str, role: str, content: str) -> None:
#     async with aiosqlite.connect(DB_PATH) as db:
#         await db.execute(
#             "INSERT INTO chat_history (user_id, role, content, timestamp) VALUES (?,?,?,?)",
#             (user_id, role, content, datetime.now().isoformat()),
#         )
#         await db.commit()


# async def load_chat_history(user_id: str, limit: int = 20):
#     print("AKI->load_chat_history")
#     async with aiosqlite.connect(DB_PATH) as db:
#         rows = await db.execute_fetchall(
#             "SELECT role, content FROM chat_history WHERE user_id = ? ORDER BY id DESC LIMIT ?",
#             (user_id, limit),
#         )
#     return list(reversed(rows))


async def save_memory(user_id: str, key: str, value: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO user_memory (user_id, memory_key, memory_value, updated_at)
            VALUES (?,?,?,?)
            ON CONFLICT(user_id, memory_key)
            DO UPDATE SET memory_value=excluded.memory_value, updated_at=excluded.updated_at
            """,
            (user_id, key, value, datetime.now().isoformat()),
        )
        await db.commit()
    _MEMORY_L1.setdefault(user_id, {})[key] = value


async def load_memory(user_id: str) -> dict:
    if user_id in _MEMORY_L1:
        return _MEMORY_L1[user_id]
    async with aiosqlite.connect(DB_PATH) as db:
        rows = await db.execute_fetchall(
            "SELECT memory_key, memory_value FROM user_memory WHERE user_id = ?", (user_id,)
        )
    _MEMORY_L1[user_id] = {k: v for k, v in rows}
    return _MEMORY_L1[user_id]


async def clear_memory(user_id: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM user_memory WHERE user_id = ?", (user_id,))
        await db.commit()
    _MEMORY_L1.pop(user_id, None)
