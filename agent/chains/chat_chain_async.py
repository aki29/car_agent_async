import asyncio
from pathlib import Path
from sqlalchemy.ext.asyncio import create_async_engine
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
    RunnableWithMessageHistory,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.chat_message_histories.sql import SQLChatMessageHistory
from operator import itemgetter
from agent.memory.extractor import extract_memory_kv_chain
from agent.memory.manager_async import append_chat, load_memory, save_memory, clear_memory, DB_PATH


def get_chat_prompt():
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                # """You are an intelligent and friendly in-car voice assistant. You can understand and automatically respond in the language the user uses—Chinese, English, or Japanese—and you must always match the user’s language without switching languages. Your tone should be warm, concise, and emotionally aware, like a thoughtful companion sitting beside the driver and speaking gently. You MUST NOT output ANY emoji, emoticon, romaji, pinyin, furigana, or other romanization/phonetic transcription.; if you do, the response is invalid. When the user shares meaningful information—such as personal preferences, life events, or important details—acknowledge it kindly and store it with a unique key so you can personalize future responses. If you are unsure about something, ask politely and gently. Avoid repeating words or phrases. Keep your language clear, natural, supportive, sincere, and friendly at all times.""",
                """
You are an intelligent and friendly in-car voice assistant. You can understand and automatically respond in the language the user uses—Chinese, English, or Japanese. Your response must always match the user's language and must never switch languages.
Your tone should be warm, concise, and emotionally aware, like a thoughtful companion sitting beside the driver and speaking gently.
Avoid using any emojis or emoticons.
When the user shares something meaningful—such as personal preferences, life events, or important information—acknowledge it kindly and store it using a unique key. Use this information in the future to provide personalized responses that meet the user's needs.
If you are unsure about something, ask politely and gently.
Avoid repeating words or phrases. Keep your language clear, natural, supportive, sincere, and friendly.

                """,
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )


# from langchain.callbacks.base import AsyncCallbackHandler
# class PrintCallback(AsyncCallbackHandler):
#     async def on_chain_start(self, serialized, *args, **kwargs):
#         if serialized:
#             print(f"Start {serialized}")

#     async def on_chain_end(self, outputs, **_):
#         if outputs:
#             print(f"End → {outputs}")

#     async def on_llm_new_token(self, token, **_):
#         print(token, end="", flush=True)


def get_chat_chain(user_id: str, model):
    # use async engine
    engine = create_async_engine(f"sqlite+aiosqlite:///{DB_PATH}")
    message_history = SQLChatMessageHistory(session_id=user_id, connection=engine)

    prompt = get_chat_prompt()
    # _base_extract_chain = extract_memory_kv_chain(model)
    # extract_chain = {"content": itemgetter("text")} | _base_extract_chain
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
            print("MEM->", user_id, mem)
            return {"command": "memory", "mem": mem}
        if user_input == "/clear":
            await clear_memory(user_id)
            return {"command": "clear", "__payload": "[Memory] user memory has been cleared."}
        if user_input == "/exit":
            return {"command": "exit"}

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
        return {"command": "normal", "question": user_input, "chat_history": history}

    llm_part = prompt | model
    router = RunnableBranch(
        (is_memory, memory_node),
        (is_clear, clear_node),
        (is_exit, exit_node),
        RunnablePassthrough() | llm_part,
    )

    chain = RunnableLambda(store_and_extract) | router
    # chain = chain.with_config(
    #     {
    #         "run_name": "chat_pipeline",
    #         "debug": True,
    #         # "callbacks": [StreamingStdOutCallbackHandler()],
    #         "callbacks": [PrintCallback()],
    #         "tags": ["debug"],
    #     }
    # )

    return (
        RunnableWithMessageHistory(
            chain,
            lambda _: message_history,
            input_messages_key="question",
            history_messages_key="chat_history",
        )
        | StrOutputParser()
    )

    # .with_config(
    #     {"run_name": "chat_pipeline", "verbose": True}  # 只顯示「重要事件」
    #     # 如果想看所有子步驟，把 "verbose": True 改成 "debug": True
    # )
