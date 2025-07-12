import re, pytz, asyncio, time
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
    RunnableParallel,
)
from agent.memory.extractor import extract_memory_kv_chain
from agent.memory.manager_async import load_memory, save_memory, clear_memory
from agent.memory.engine import checkpoint_db
from datetime import datetime
from agent.chains.classify_chain import classify_chain
from operator import itemgetter
from pprint import pprint
from agent.utils.lang import LangDetector

lang_det = LangDetector()


def detect_lang(text: str) -> str:
    return lang_det.detect(text)


tz = pytz.timezone("Asia/Taipei")  # 若未來要自動偵測車機時區，可改成 time.tznamei
LANG_HINT = {
    "zh-tw": "以下回答必須使用【繁體中文】，不得包含任何英文或日文。",
    "zh-cn": "接下來所有回答請使用【简体中文】，不得包含任何英文或日文。",
    "en": "From now on, answer strictly in **English**. Do NOT output any Chinese or Japanese.",
    "ja": (
        "これ以降の回答は必ず【日本語】で書いてください。"
        "中国語や英語の語句を一切含めてはいけません。"
    ),
}


def get_chat_prompt(lang_hint: str = ""):
    SYSTEM_CORE = f"""{lang_hint}
你是智慧且友善的AI車載語音助理.
【語氣】溫暖、精簡、體貼，如同坐在駕駛旁的夥伴。
【思考】隱藏推理，只輸出最終答案。
【親切回應】用戶透露偏好或重要資訊時，直接個性化運用。
【使用者資料使用】僅在與當前問題明確相關時才引用使用者資料；否則忽略。
【歷史使用】僅在與當前問題明確相關時才引用歷史；否則忽略。
"""
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_CORE),
            ("system", "【使用者資料】\n<<<\n{user_profile}\n>>>"),
            ("system", "【歷史】\n<<<\n{chat_history}\n>>>"),
            ("human", "{question}"),
        ]
    )


def get_chat_chain(user_id: str, model, mem_mgr, rag):
    extract_chain = extract_memory_kv_chain(model)

    def make_rag_node(domain_name: str):
        async def _answer(d):
            q = d["question"]
            if domain_name == "manual":
                docs = await rag.query_manual(q)
            elif domain_name == "parking":
                docs = await rag.query_parking(q, user_loc=d.get("user_loc"))
            elif domain_name == "food":
                docs = await rag.query_food(q, user_loc=d.get("user_loc"))
            else:  # poi
                docs = await rag.query_poi(q, user_loc=d.get("user_loc"))
            # print(docs)
            top_docs = docs[:4]
            context = "\n---\n".join(x.page_content for x in top_docs)
            prompt_tmpl = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        f"你是車載助手 以下是找到的 {domain_name} 資訊 "
                        "請用最相關 3 條結果總結回答 每條 ≤25 字 不得使用標點符號和換行 "
                        "用空格分隔條目 最後加 想了解哪一項？",
                    ),
                    ("system", "{context}"),
                    ("human", "{question}"),
                ]
            )
            return await (prompt_tmpl | model).ainvoke({"context": context, "question": q})

        return RunnableLambda(_answer)

    def _fmt_time(lang: str) -> str:
        now = datetime.now(tz)
        hour = now.strftime("%-I")
        minute = now.strftime("%M")
        ampm = now.strftime("%p")
        if lang == "zh-tw":
            ampm = "上午" if ampm == "AM" else "下午"
            return f"現在是{ampm} {hour}:{minute}"
        if lang == "ja":
            ampm = "午前" if ampm == "AM" else "午後"
            return f"現在は{ampm}{hour}:{minute}です"
        # default English
        return f"It is {hour}:{minute} {ampm}"

    def _fmt_date(lang: str) -> str:
        now = datetime.now(tz)
        if lang == "zh-tw":
            return now.strftime("今天是 %Y/%m/%d (%A)")
        if lang == "ja":
            return now.strftime("今日は %Y年%m月%d日 (%A) です")
        return now.strftime("Today is %A, %Y-%m-%d")

    def strip_punct(text: str) -> str:
        return re.sub(r"[，,。.！!？?:：;；…\-—「」『』‘’“”]", "", text).strip()

    time_node = RunnableLambda(lambda d: _fmt_time(detect_lang(d["question"])))
    date_node = RunnableLambda(lambda d: _fmt_date(detect_lang(d["question"])))
    clean_node = RunnableLambda(strip_punct)

    manual_node = make_rag_node("manual")
    parking_node = make_rag_node("parking")
    food_node = make_rag_node("food")
    poi_node = make_rag_node("poi")

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

        if user_input == "/checkpoint":
            checkpoint_db()
            return "[Memory] WAL has been synced to the main database file."

        if user_input == "/exit":
            return ""

        def compress_history(msgs, max_pairs=3, max_len=40):
            keep = msgs[-max_pairs * 2 :]
            out = []
            for m in keep:
                role = "Human" if m.type == "human" else "AI"
                text = m.content.strip().replace("\n", " ")[:max_len]
                if len(m.content) > max_len:
                    text += "…"
                out.append(f"{role}: {text}")
            return out

        async def _extract():
            try:
                ext = await extract_chain.ainvoke({"content": user_input})
                if isinstance(ext, dict):
                    tasks = [
                        save_memory(user_id, str(k).strip(), str(v).strip())
                        for k, v in ext.items()
                        if k and v
                    ]
                if tasks:
                    await asyncio.gather(*tasks)
            except Exception as e:
                print("[!] memory extract err:", e)

        asyncio.create_task(_extract())

        mem_vars = await mem_mgr.summary.aload_memory_variables({})
        raw_msgs = mem_vars.get("history", [])
        compact_history = compress_history(raw_msgs, max_pairs=4, max_len=60)

        # raw_sags = await mem_mgr._store.aget_messages()
        # compact_history = compress_history(raw_msgs, max_pairs=4, max_len=60)

        # mem_vars = mem_mgr.history.load_memory_variables({})
        # chat_history = mem_vars.get("chat_history", [])
        # print("CCC", type(chat_history), chat_history)

        profile_dict = await load_memory(user_id)
        profile_text = "\n".join(f"{k}: {v}" for k, v in profile_dict.items())

        reply = await router.ainvoke(
            {
                "question": user_input,
                "chat_history": compact_history,
                "user_profile": profile_text,
            }
        )

        # if isinstance(reply, str):
        # await mem_mgr.save_turn(user_input, reply)
        # await mem_mgr.save_turn(user_input, reply)

        if hasattr(reply, "content"):
            return reply.content
        else:
            return reply

    from termcolor import colored

    def get_message(input):
        print(colored(str(type(input)) + ": " + f"{input}", "yellow", attrs=["bold"]))
        return input

    async def _run_llm(d):
        lang_code = detect_lang(d["question"])
        print("lang_code", lang_code)
        lang_hint = LANG_HINT[lang_code]
        dyn_prompt = get_chat_prompt(lang_hint)

        # return (dyn_prompt).bind(**d) | get_message
        return (dyn_prompt).bind(**d)

    llm_part = RunnableLambda(_run_llm)

    CHAIN_MAP = {
        "time": time_node,
        "date": date_node,
        "chat": llm_part,
        "manual": manual_node,
        "parking": parking_node,
        "food": food_node,
        "poi": poi_node,
        "music": llm_part,
        "navigation": llm_part,
        "news": llm_part,
    }

    parallel = RunnableParallel(
        route={"input": itemgetter("question")} | classify_chain
        # | get_message
        | RunnableLambda(lambda r: r if isinstance(r, dict) else r.model_dump()),
        payload=RunnablePassthrough(),
    )

    STREAMABLE = {"chat", "music", "navigation", "news"}

    async def _dispatch(d):
        dest = d["route"]["destination"]
        pprint(d)
        chain = CHAIN_MAP[dest]
        if dest in STREAMABLE:
            return await chain.ainvoke(d["payload"])
        else:
            return await chain.ainvoke(d["payload"])

    router = parallel | RunnableLambda(_dispatch)

    is_final_str = lambda x: isinstance(x, str)

    branch_tail = RunnableBranch((is_final_str, RunnablePassthrough()), model)
    chain = RunnableLambda(store_and_extract)
    stream_chain = chain | branch_tail
    invoke_chain = chain | StrOutputParser() | clean_node
    return {"stream": stream_chain, "invoke": invoke_chain}
