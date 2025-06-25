from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from enum import Enum
import os, re
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


class DestEnum(str, Enum):
    time = "time"
    date = "date"
    poi = "poi"
    chat = "chat"
    music = "music"
    navigation = "navigation"
    guide = "guide"
    news = "news"


class RouteQuery(BaseModel):
    destination: DestEnum = Field(...)
    # justification: str = Field(default="", description="為什麼是這個答案")


llm_opts = dict(
    model=os.getenv("ROUTER_MODEL", "phi3:instruct"),
    base_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
    temperature=0.0,
    top_p=0.15,
    top_k=30,
    repeat_penalty=1.15,
    num_predict=12,
    num_ctx=1024,
    stop=["}"],
    format="json",
    seed=42,
    num_thread=6,
    keep_alive=-1,
)
_llm = ChatOllama(**llm_opts)


_router_prompt = PromptTemplate.from_template(
    """你是一個意圖分類器，僅回傳{{"destination": "<time|date|poi|chat|music|navigation|guide|news>"}}。

**分類定義**：
• time       ：問現在時間 (例："現在幾點?", "what time is it?")
• date       ：問今天日期 (例："今天幾號?", "what's the date today?")
• poi        ：問附近地點 (例："附近有加油站嗎?", "any coffee shop nearby?", "近くのレストランは？")
• chat       ：一般對話或問候 (例："hi", "你好", "元気ですか？")
• music      ：跟音樂、歌曲、歌單相關 (例："播音樂", "我想聽點輕快的歌", "play some jazz")
• navigation：跟導航、路線、方向相關 (例："怎麼去火車站?", "navigate to airport", "空港への行き方を教えて")
• guide     ：導覽、旅遊、介紹景點 (例："介紹一下這個城市", "有什麼景點?", "観光ガイドして")
• news      ：跟新聞、時事、最新消息相關 (例："有什麼最新新聞?", "給我最新國際消息")

**規則**
1. 先比對關鍵詞 (中/英/日)；若無明確關鍵詞就選 chat。
2. 只能輸出 destination；不得輸出多餘文字。

**示例**
使用者: hi
輸出: {{"destination":"chat"}}

使用者: navigate to airport
輸出: {{"destination":"navigation"}}

使用者: 你要去哪裡 日文如何說
輸出: {{"destination":"chat"}}

使用者訊息：
{input}"""
)


_llm_chain = _router_prompt | _llm.with_structured_output(RouteQuery)


classify_chain = _llm_chain
