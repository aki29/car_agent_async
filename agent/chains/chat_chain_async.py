import asyncio
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
)
from agent.memory.extractor import extract_memory_kv_chain
from agent.memory.manager_async import load_memory, save_memory, clear_memory
from agent.memory.engine import checkpoint_db
from agent.rag import rag_manager
import re
from datetime import datetime
import pytz

tz = pytz.timezone("Asia/Taipei")  # 若未來要自動偵測車機時區，可改成 time.tznamei
_RE_JP = re.compile(r"[ぁ-んァ-ン一-龯]")
_RE_KANA = re.compile(r"[ぁ-んァ-ン]")
_RE_ZH = re.compile(r"[\u4e00-\u9fff]")


def detect_lang(text: str) -> str:  # 'ja' | 'zh' | 'en'
    # if _RE_JP.search(text):
    if _RE_KANA.search(text):
        return "ja"
    return "zh" if _RE_ZH.search(text) else "en"


def get_chat_prompt():
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                #                 """You are an in-car AI assistant. Detect user language (zh-Hant, en, ja) and answer in that language. Be concise (≤50 chars). No emoji.
                # """,
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
            ("system", "【使用者資料】\n{user_profile}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )


def get_chat_chain(user_id: str, model, mem_mgr, rag):
    prompt = get_chat_prompt()
    extract_chain = extract_memory_kv_chain(model)

    _TIME_PAT = re.compile(
        r"( ?(?:what(?:'s|\s+is)?\s+the\s+)?time\b|幾點|現在.*幾點|今何時)", re.I
    )
    _DATE_PAT = re.compile(r"( ?(?:what(?:'s|\s+is)?\s+the\s+)?date\b|今天幾號|日期|何日)", re.I)
    _POI_PAT = re.compile(r"(餐廳|咖啡|加油站|停車場|poi|附近)", re.I)

    async def _answer_with_poi(input_dict):
        q = input_dict["question"]
        user_profile = input_dict.get("user_profile", "")
        docs = await rag.query_poi(q, user_loc=None)
        context = "\n".join(d.page_content for d in docs)

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
        return await (prompt_tmpl | model).ainvoke(
            {
                "user_profile": user_profile,
                "context": context,
                "question": q,
            }
        )

    def is_poi(d):
        return bool(_POI_PAT.search(d["question"]))

    def is_time(d):
        return bool(_TIME_PAT.search(d["question"]))

    def is_date(d):
        return bool(_DATE_PAT.search(d["question"]))

    def _fmt_time(lang: str) -> str:
        now = datetime.now(tz)
        hour = now.strftime("%-I")  # 去掉前導 0
        minute = now.strftime("%M")
        ampm = now.strftime("%p")  # 'AM' / 'PM'
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

        if user_input == "/checkpoint":
            checkpoint_db()
            return "[Memory] WAL has been synced to the main database file."

        if user_input == "/exit":
            return ""

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
        chat_history = mem_vars.get("history", [])

        # mem_vars = mem_mgr.history.load_memory_variables({})
        # chat_history = mem_vars.get("chat_history", [])
        profile_dict = await load_memory(user_id)
        profile_text = "\n".join(f"{k}: {v}" for k, v in profile_dict.items())
        # print("AKI-->", profile_text)
        reply = await router.ainvoke(
            {"question": user_input, "chat_history": chat_history, "user_profile": profile_text}
        )
        await mem_mgr.save_turn(user_input, reply)
        return reply

    llm_part = prompt | model
    router = RunnableBranch(
        (is_time, time_node),
        (is_date, date_node),
        (is_poi, poi_node),
        (is_memory, memory_node),
        (is_clear, clear_node),
        (is_exit, exit_node),
        RunnablePassthrough() | llm_part,
    )

    chain = RunnableLambda(store_and_extract)
    return chain | StrOutputParser()
