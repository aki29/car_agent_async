# agent/memory/summary_memory.py
from langchain.memory import ConversationSummaryBufferMemory
from langchain.llms.base import BaseLanguageModel
from langchain.prompts import PromptTemplate
from agent.memory.rolling_sql_history import RollingSQLHistory


def create_summary_memory(
    llm: BaseLanguageModel,
    token_limit: int = 1500,
    chat_memory: RollingSQLHistory | None = None,
) -> ConversationSummaryBufferMemory:
    summary_prompt = PromptTemplate(
        input_variables=["summary", "new_lines"],
        template=(
            "Summarize the key points in ≤30 English words.\n\n"
            "Previous summary:\n{summary}\n\n"
            "Recent lines:\n{new_lines}"
        ),
    )
    return ConversationSummaryBufferMemory(
        llm=llm,
        chat_memory=chat_memory,
        prompt=summary_prompt,
        max_token_limit=token_limit,
        return_messages=True,
        memory_key="history",
        summary_key="summary",
    )
