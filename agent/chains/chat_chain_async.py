import asyncio
from pathlib import Path
from sqlalchemy.ext.asyncio import create_async_engine
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories.sql import SQLChatMessageHistory

from agent.memory.manager_async import append_chat, load_memory, save_memory, clear_memory
from agent.memory.extractor import extract_memory_kv_chain

DB_PATH = Path(__file__).parent.parent / "data" / "ctk_memory.sqlite3"


def get_chat_prompt():
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an intelligent and friendly in-car voice assistant. You can understand and automatically respond in the language the user uses—Chinese, English, or Japanese. Your response must always match the user's language and must never switch languages.                Your tone should be warm, concise, and emotionally aware, like a thoughtful companion sitting beside the driver and speaking gently.
Avoid using any emojis or emoticons. When the user shares something meaningful—such as personal preferences, life events, or important information—acknowledge it kindly and store it using a unique key. Use this information in the future to provide personalized responses that meet the user's needs.
If you are unsure about something, ask politely and gently. Avoid repeating words or phrases. Keep your language clear, natural, supportive, sincere, and friendly.""",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )


def get_chat_chain(user_id: str, model):
    # use async engine
    engine = create_async_engine(f"sqlite+aiosqlite:///{DB_PATH}")
    message_history = SQLChatMessageHistory(session_id=user_id, connection=engine)

    prompt = get_chat_prompt()
    extract_chain = extract_memory_kv_chain(model)

    async def store_and_extract(input_dict):
        user_input = input_dict["question"]

        cmd = user_input.strip()
        if cmd == "/memory":
            mem = await load_memory(user_id)
            return {"question": f"目前記憶: {mem or '尚無資料'}", "chat_history": []}
        if cmd == "/clear":
            await clear_memory(user_id)
            return {"question": "已清除使用者記憶", "chat_history": []}
        if cmd == "/exit":
            return {"question": "[exit]", "chat_history": []}

        await append_chat(user_id, "user", user_input)

        async def _extract():
            try:
                ext = await extract_chain.ainvoke({"text": user_input})
                if isinstance(ext, dict):
                    tasks = [save_memory(user_id, k.strip(), v.strip()) for k, v in ext.items()]
                    await asyncio.gather(*tasks)
            except Exception as e:
                print(f"[!] Memory extraction failed: {e}")

        asyncio.create_task(_extract())

        history_msgs = await message_history.aget_messages()
        return {"question": user_input, "chat_history": history_msgs}

    chain = RunnableLambda(store_and_extract) | prompt | model

    return (
        RunnableWithMessageHistory(
            chain,
            lambda _: message_history,
            input_messages_key="question",
            history_messages_key="chat_history",
        )
        | StrOutputParser()
    ).with_config(
        {"run_name": "chat_pipeline", "verbose": True}  # 只顯示「重要事件」
        # 如果想看所有子步驟，把 "verbose": True 改成 "debug": True
    )
