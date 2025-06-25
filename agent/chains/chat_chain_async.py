import re, pytz, asyncio
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

# from langchain_core.runnables import RunnableWithMessageHistory
from operator import itemgetter
from pprint import pprint

tz = pytz.timezone("Asia/Taipei")  # 若未來要自動偵測車機時區，可改成 time.tznamei
_RE_JP = re.compile(r"[ぁ-んァ-ン一-龯]")
_RE_KANA = re.compile(r"[ぁ-んァ-ン]")
_RE_ZH = re.compile(r"[\u4e00-\u9fff]")


def detect_lang(text: str) -> str:  # 'ja' | 'zh' | 'en'
    # if _RE_JP.search(text):
    if _RE_KANA.search(text):
        return "ja"
    return "zh" if _RE_ZH.search(text) else "en"


LANG_HINT = {
    "zh": "以下回答必須使用【繁體中文】，不得包含任何英文或日文。",
    "en": "From now on, answer strictly in **English**. Do NOT output any Chinese or Japanese.",
    "ja": (
        "これ以降の回答は必ず【日本語】で書いてください。"
        "中国語や英語の語句を一切含めてはいけません。"
    ),
}


def get_chat_prompt(lang_hint: str = ""):
    return ChatPromptTemplate.from_messages(
        [
            ("system", lang_hint),
            (
                "system",
                """
                 You are an intelligent and friendly in-car voice assistant.
                【語言】偵測並使用用戶語言（繁中／英／日），依使用者輸入語言回應不得切換。
                【語氣】溫暖、精簡、體貼，如同坐在駕駛旁的夥伴。   
                【思考】隱藏推理，只輸出最終答案。               
                【記憶】用戶透露偏好或重要資訊時，親切回應並以唯一key存檔，後續直接個性化運用。
                【釐清】有疑慮時，可禮貌提問確認。
                """,
            ),
            (
                "system",
                """
【禁止符號】, ， . 。 ！ ! ？ ? ： : ； ; 「 」 『 』 ‘ ’ “ ” - — … 、請完全不要輸出以上任何符號，保留空格即可。

【示例】
ユーザー: おはよう
アシスタント: おはよう 素敵な一日を

使用者: 早安
助手輸出: 早安 祝您有個美好的一天

User: Good morning
Assistant: Good morning Have a wonderful day
""",
            ),
            ("system", "【使用者資料】\n{user_profile}"),
            ("system", "【歷史】{chat_history}"),
            # MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )


def get_chat_chain(user_id: str, model, mem_mgr, rag):
    prompt = get_chat_prompt()
    extract_chain = extract_memory_kv_chain(model)

    async def _answer_with_poi(input_dict):
        q = input_dict["question"]
        user_profile = input_dict.get("user_profile", "")
        docs = await rag.query_poi(q, user_loc=None)
        # context = "\n".join(d.page_content for d in docs)
        context = "\n".join(str(d.metadata) for d in docs)

        prompt_tmpl = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are an in-car assistant. Here are the POI(Point of Interest)s found from RAG; reply warmly and concisely (>30 characters, ≤ 100 characters) to match user preferences.\n
                    [Found POIs]\n{context}\n\n
                    """,
                ),
                ("system", "[User Data]\n{user_profile}\n\n"),
                ("human", "{question}"),
            ]
        )
        result = await (prompt_tmpl | model).ainvoke(
            {
                "user_profile": user_profile,
                "context": context,
                "question": q,
            }
        )
        return result

    def _fmt_time(lang: str) -> str:
        now = datetime.now(tz)
        hour = now.strftime("%-I")
        minute = now.strftime("%M")
        ampm = now.strftime("%p")
        if lang == "zh":
            ampm = "上午" if ampm == "AM" else "下午"
            return f"現在是{ampm} {hour}:{minute}"
        if lang == "ja":
            ampm = "午前" if ampm == "AM" else "午後"
            return f"現在は{ampm}{hour}:{minute}です"
        # default English
        return f"It is {hour}:{minute} {ampm}"

    def _fmt_date(lang: str) -> str:
        now = datetime.now(tz)
        if lang == "zh":
            return now.strftime("今天是 %Y/%m/%d (%A)")
        if lang == "ja":
            return now.strftime("今日は %Y年%m月%d日 (%A) です")
        return now.strftime("Today is %A, %Y-%m-%d")

    time_node = RunnableLambda(lambda d: _fmt_time(detect_lang(d["question"])))
    date_node = RunnableLambda(lambda d: _fmt_date(detect_lang(d["question"])))
    poi_node = RunnableLambda(_answer_with_poi)

    async def store_and_extract(input_dict):
        user_input = input_dict["question"].strip()
        # lang_code  = detect_lang(user_input)
        # lang_hint  = LANG_HINT[lang_code]

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
        compact_history = compress_history(raw_msgs, max_pairs=8, max_len=80)

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
        # print(reply)
        await mem_mgr.save_turn(user_input, reply)
        return reply

    # from termcolor import colored

    # def get_message(input):
    #     print(colored(str(type(input)) + ": " + f"{input}", "yellow", attrs=["bold"]))
    #     return input

    # llm_part = prompt | model
    async def _run_llm(d):
        lang_code = detect_lang(d["question"])
        lang_hint = LANG_HINT[lang_code]
        dyn_prompt = get_chat_prompt(lang_hint)
        chain = dyn_prompt | model
        return await chain.ainvoke(d)

    llm_part = RunnableLambda(_run_llm)

    CHAIN_MAP = {
        "time": time_node,
        "date": date_node,
        "poi": poi_node,
        "chat": llm_part,
        "music": llm_part,
        "navigation": llm_part,
        "guide": llm_part,
        "news": llm_part,
    }

    parallel = RunnableParallel(
        route={"input": itemgetter("question")} | classify_chain
        # | get_message
        | RunnableLambda(lambda r: r if isinstance(r, dict) else r.model_dump()),
        payload=RunnablePassthrough(),
    )
    # | (lambda r: r.model_dump())

    async def _dispatch(d):
        dest = d["route"]["destination"]
        # pprint(d)
        chain = CHAIN_MAP[dest]
        return await chain.ainvoke(d["payload"])

    router = parallel | RunnableLambda(_dispatch)

    chain = RunnableLambda(store_and_extract)
    return chain | StrOutputParser()
