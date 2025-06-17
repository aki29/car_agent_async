import asyncio
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    RunnableWithMessageHistory,
)
from langchain_community.chat_message_histories.sql import (
    SQLChatMessageHistory,
)
from sqlalchemy.ext.asyncio import create_async_engine

from agent.memory.manager_async import (
    append_chat,
    load_memory,
    save_memory,
    clear_memory,
)
from agent.memory.extractor import extract_memory_kv_chain

DB_PATH = Path(__file__).parent.parent.parent / "data" / "ctk_memory.sqlite3"

def _chat_prompt():
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an intelligent and friendly in‑car voice assistant.
Respond in the same language as the user's input.""",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

def get_chat_chain(user_id: str, model, retriever=None):
    engine = create_async_engine(f"sqlite+aiosqlite:///{DB_PATH}")
    message_history = SQLChatMessageHistory(
        session_id=user_id, connection=engine
    )

    prompt = _chat_prompt()
    extractor = extract_memory_kv_chain(model)

    async def preprocess(inputs):
        question = inputs["question"].strip()

        # slash commands
        if question == "/memory":
            mem = await load_memory(user_id)
            return {
                "question": f"目前記憶: {mem or '尚無資料'}",
                "chat_history": [],
            }
        if question == "/clear":
            await clear_memory(user_id)
            return {"question": "已清除使用者記憶", "chat_history": []}
        if question == "/exit":
            return {"question": "[exit]", "chat_history": []}

        await append_chat(user_id, "user", question)

        async def _extract():
            try:
                ext = await extractor.ainvoke({"text": question})
                if isinstance(ext, dict):
                    await asyncio.gather(
                        *[
                            save_memory(user_id, k, v)
                            for k, v in ext.items()
                        ]
                    )
            except Exception as exc:
                print(f"[extractor] {exc}")

        asyncio.create_task(_extract())

        chats = await message_history.aget_messages()
        return {"question": question, "chat_history": chats}

    chain = (
        RunnableLambda(preprocess)
        | prompt
        | model
        | StrOutputParser()
    )

    return RunnableWithMessageHistory(
        chain,
        lambda _: message_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )
