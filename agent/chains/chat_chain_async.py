import asyncio
from pathlib import Path
from sqlalchemy.ext.asyncio import create_async_engine
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.chat_message_histories.sql import SQLChatMessageHistory
from agent.memory.rolling_history import RollingSQLHistory
from agent.memory.engine import async_session, async_engine, sync_engine
from operator import itemgetter
from agent.memory.extractor import extract_memory_kv_chain
from agent.memory.manager_async import append_chat, load_memory, save_memory, clear_memory, DB_PATH


def get_chat_prompt():
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                【語言】偵測並使用用戶語言（繁中／英／日），依使用者輸入語言回應不得切換。
                【語氣】溫暖、精簡、體貼，如同坐在駕駛旁的夥伴。
                【標點】禁止 emoji、顏文字、裝飾符號、標點符號。
                【思考】隱藏推理，只輸出最終答案。
                【輸出限制】全文 ≤50 字；若需步驟，列 ≤3 步，每步 ≤15 字。
                【記憶】用戶透露偏好或重要資訊時，親切回應並以唯一key存檔，後續直接個性化運用。
                【釐清】有疑慮時，可禮貌提問確認。
                """,
                # """
                # You are an intelligent and friendly in-car voice assistant. You can understand and automatically respond in the language the user uses—Chinese, English, or Japanese. Your response must always match the user's language and must never switch languages.
                # Your tone should be warm, concise, and emotionally aware, like a thoughtful companion sitting beside the driver and speaking gently.
                # Avoid using any emojis or emoticons.
                # When the user shares something meaningful—such as personal preferences, life events, or important information—acknowledge it kindly and store it using a unique key. Use this information in the future to provide personalized responses that meet the user's needs.
                # If you are unsure about something, ask politely and gently.
                # Avoid repeating words or phrases. Keep your language clear, natural, supportive, sincere, and friendly.""",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )


def get_chat_chain(user_id: str, model):

    message_history = RollingSQLHistory(
        session_id=user_id,
        connection=sync_engine,
        window_size=4,
    )

    prompt = get_chat_prompt()
    extract_chain = extract_memory_kv_chain(model)

    def is_memory(d):
        return d.get("command") == "memory"

    def is_clear(d):
        return d.get("command") == "clear"

    def is_exit(d):
        return d.get("command") == "exit"

    memory_node = RunnableLambda(lambda d: f"[Memory] {d['mem'] or 'No Data.'}")
    clear_node = RunnableLambda(lambda d: d["__payload"])
    exit_node = RunnableLambda(lambda d: "")

    async def store_and_extract(input_dict):
        user_input = input_dict["question"].strip()

        if user_input == "/memory":
            mem = await load_memory(user_id)
            if mem:
                lines = [f"{k}: {v}" for k, v in mem.items()]
                return "[Memory]\n" + "\n".join(lines)
            return "[Memory] No data."

        if user_input == "/clear":
            await clear_memory(user_id)
            return "[Memory] Cleared."

        if user_input == "/exit":
            return ""

        await append_chat(user_id, "user", user_input)

        async def _extract():
            try:
                ext = await extract_chain.ainvoke({"content": user_input})
                if isinstance(ext, dict):
                    await asyncio.gather(
                        *[save_memory(user_id, k.strip(), v.strip()) for k, v in ext.items()]
                    )
            except Exception as e:
                print("[!] memory extract err:", e)

        asyncio.create_task(_extract())

        history = await message_history.aget_messages()

        reply = await router.ainvoke({"question": user_input, "chat_history": history})

        message_history.add_user_message(user_input)
        message_history.add_ai_message(reply)
        return reply

    llm_part = prompt | model
    router = RunnableBranch(
        (is_memory, memory_node),
        (is_clear, clear_node),
        (is_exit, exit_node),
        RunnablePassthrough() | llm_part,
    )

    chain = RunnableLambda(store_and_extract)
    return chain | StrOutputParser()
