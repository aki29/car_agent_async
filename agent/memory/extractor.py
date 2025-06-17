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
    """
    回傳一個 Runnable，可接受形如 {"content": <user_text>} 的 dict，
    非同步叫用 LLM 抽取「鍵-值」資訊並以 dict 形式輸出。

    Example
    -------
    >>> chain = extract_memory_kv_chain(llm)
    >>> result = await chain.ainvoke({"content": "我叫小明，住台北，最愛聽爵士樂"})
    >>> print(result)   # {"name": "小明", "location": "台北", "favorite_music": "爵士樂"}
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一個資訊抽取器，工作是把使用者句子裡的『屬性: 值』資訊找出來。\n"
                "請只回傳 **單行**、有效的 JSON 物件，鍵用英文字，值保留原語意；"
                "若沒有可抽取的鍵值，請回傳空的 JSON 物件 {{}}。",
            ),
            ("human", "{content}"),
        ]
    )

    async def _extract(inputs: Dict[str, Any]) -> Dict[str, Any]:
        user_text = inputs.get("content", "")
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
