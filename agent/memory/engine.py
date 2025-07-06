import os
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy import text
from pathlib import Path
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError


_DEFAULT_PATH = Path(__file__).resolve().parent.parent.parent / ".cache" / "ctk_memory.sqlite3"
DB_PATH = Path(os.environ.get("CTK_MEM_DB_PATH", _DEFAULT_PATH))

# DB_PATH.parent.mkdir(parents=True, exist_ok=True)
# if not DB_PATH.exists():
#     DB_PATH.touch()

import sqlite3


def checkpoint_db(path=DB_PATH):
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
    conn.close()


ASYNC_DSN = f"sqlite+aiosqlite:///{DB_PATH}"
SYNC_DSN = f"sqlite:///{DB_PATH}"

async_engine = create_async_engine(
    ASYNC_DSN,
    echo=False,
    pool_size=5,
    max_overflow=10,
    future=True,
)
async_session = async_sessionmaker(async_engine, expire_on_commit=False)

sync_engine = create_engine(
    SYNC_DSN,
    future=True,
    connect_args={"check_same_thread": False},
)


def init_sync_db() -> None:
    with sync_engine.begin() as conn:
        conn.execute(text("PRAGMA journal_mode=WAL;"))
        try:
            conn.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS ix_messages_session_id_id
                    ON messages (session_id, id DESC);
                    """
                )
            )
        except OperationalError:
            pass


init_sync_db()
