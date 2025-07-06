from __future__ import annotations

import ast
import json
from typing import Any, Dict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.chat_history import HumanMessage

# SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel


def extract_memory_kv_chain(model: BaseChatModel): 
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an information extractor. Your task is to find any user-profile facts in the input "
                "and return **one-line valid JSON**. "
                "Use the following English keys when possible: "
                "name, age, location, favorite_music, favorite_food, profession, hobby. "
                "If nothing can be extracted, return an empty JSON object {{}}."
                "MUST NOT output ANY emoji, emoticon, romaji, pinyin, furigana, or other romanization/phonetic transcription.",
            ),
            (
                "human",
                "我叫小明，住台北，最愛聽爵士樂, 30 years,性別男,電話0988",
            ),
            (
                "ai",
                '{{"name":"小明","location":"台北","favorite_music":"爵士樂","age":"30 years","sex":"Male","phone":"0988"}}',
            ),
            (
                "human",
                "Hi, I'm Akira from Osaka, 34 years old. Love listening to rock.",
            ),
            (
                "ai",
                '{{"name":"Akira","location":"Osaka","age":"34 years old","favorite_music":"rock"}}',
            ),
            ("human", "{content}"),
        ]
    )

    async def _extract(inputs: Dict[str, Any]) -> Dict[str, Any]:
        user_text = inputs.get("content") or inputs.get("text") or ""
        messages = prompt.format_messages(content=user_text)
        response = await model.ainvoke(messages)
        raw_content = (response.content if hasattr(response, "content") else str(response)).strip()
        for parse in (_safe_json_loads, _safe_ast_literal):
            parsed = parse(raw_content)
            if isinstance(parsed, dict):
                return parsed
        return {}

    return RunnableLambda(_extract)


def _safe_json_loads(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None


def _safe_ast_literal(s: str):
    try:
        return ast.literal_eval(s)
    except Exception:
        return None
