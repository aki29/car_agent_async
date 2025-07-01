from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv, find_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field

load_dotenv(find_dotenv())


class DestEnum(str, Enum):
    time = "time"
    date = "date"
    manual = "manual"
    parking = "parking"
    food = "food"
    poi = "poi"
    chat = "chat"
    music = "music"
    navigation = "navigation"
    news = "news"


class RouteQuery(BaseModel):
    destination: DestEnum = Field(..., description="意圖分類結果")
    justification: str = Field("", description="簡短理由，10-20字")


_llm = ChatOllama(
    model=os.getenv("ROUTER_MODEL", "phi3:instruct"),
    base_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
    temperature=0.0,
    format="json",
    num_predict=10, 
    num_thread=int(os.getenv("OLLAMA_THREADS", "4")),
    keep_alive=-1,
)


_prompt_tmpl = PromptTemplate.from_template(
    r"""
You are an **in‑car voice assistant intent classifier** that supports **Chinese, English, and Japanese**.
Read the USER message and reply with **exactly one line JSON** only:
{{"destination":"<time|date|manual|parking|food|poi|chat|music|navigation|news>","justification":"<10‑20字/words>"}}
No other characters or line breaks.
================ CATEGORY GUIDE (ZH / EN / JA) =================
time – current time / world clock / countdown  
  中: 現在幾點 倒數五分鐘│EN: what time is it│日: 今何時 カウントダウン
date – date / weekday / holiday  
  中: 今天幾號│EN: what’s today’s date│日: 今日は何日
manual – vehicle function / troubleshooting / maintenance  
  中: 雨刷故障 胎壓燈亮 鑰匙沒電│EN: wiper not working│日: ワイパー故障
parking – parking lot / charger / fee / hours  
  中: 附近停車場 充電站│EN: nearest parking│日: 近くの駐車場
food – restaurant / cafe / cuisine  
  中: 想吃拉麵│EN: good sushi restaurant│日: カレーが食べたい
poi – tourist spot / park / museum  
  中: 台北景點有哪些│EN: attractions in Kyoto│日: 観光スポット
chat – greetings / small talk / joke  
  中: 你好呀│EN: hi there│日: こんにちは
music – play / pause / next / volume  
  中: 播放爵士樂│EN: play some jazz│日: 音楽を再生
navigation – route / traffic / ETA  
  中: 導航到機場│EN: navigate to station│日: 東京タワーまで案内
news – news / weather / finance / sports  
  中: 最新國際新聞│EN: tech news│日: 今日のニュース
================ FEW‑SHOT EXAMPLES ================
hi → {{"destination":"chat","justification":"打招呼"}}
おはよう → {{"destination":"chat","justification":"挨拶"}}
I’m hungry any ramen? → {{"destination":"food","justification":"找吃"}}
雨刷有點故障 → {{"destination":"manual","justification":"車輛故障"}}
navigate to Tokyo Tower → {{"destination":"navigation","justification":"路線"}}
最新国際ニュース → {{"destination":"news","justification":"want news"}}
USER MESSAGE:
{input}
"""
)


_llm_chain = _prompt_tmpl | _llm.with_structured_output(RouteQuery)
classify_chain = RunnableLambda(lambda d: _llm_chain.invoke({"input": d["input"]}))
