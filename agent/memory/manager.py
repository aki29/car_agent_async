# agent/memory/manager.py
# from langchain.memory import CombinedMemory
from agent.memory.rolling_sql_history import RollingSQLHistory
from langchain.memory import ConversationBufferMemory
from agent.memory.summary_memory import create_summary_memory
from agent.memory.engine import sync_engine

import sys, os


class SuppressStdout:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr


class MemoryManager:
    def __init__(
        self,
        llm,
        session_id: str,
        max_messages: int = 20,
        token_limit: int = 1500,
    ):
        with SuppressStdout():
            self._store = RollingSQLHistory(
                session_id=session_id, connection=sync_engine, window_size=max_messages
            )
            self.history = ConversationBufferMemory(
                chat_memory=self._store,
                return_messages=True,
                memory_key="chat_history",
            )
            self.summary = create_summary_memory(llm, token_limit)
            # self.summary = create_summary_memory(llm, chat_memory=self._store, token_limit=1500)
            # self.memory = CombinedMemory(memories=[self.history, self.summary])

    # def get(self):
    #     return self.memory

    async def save_turn(self, user_text: str, ai_text: str):
        # with SuppressStdout():
        content = ai_text.content if hasattr(ai_text, "content") else str(ai_text)
        self._store.add_user_message(user_text)
        self._store.add_ai_message(content)
        await self.summary.asave_context(
            {"input": user_text},
            {"output": content},
        )
