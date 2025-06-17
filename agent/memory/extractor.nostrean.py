"""A toy memory key‑value extractor chain.

Replace with your own schema extractor. The default implementation returns
an empty dict so the assistant still works even without extra dependencies.
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


def extract_memory_kv_chain(model):
    """Return a LangChain Runnable that extracts user memory kv.

    The Runnable's .ainvoke() returns a Python dict like {key: value, ...}
    """
    parser = JsonOutputParser()
    prompt = PromptTemplate.from_template(
        """請從使用者輸入中提取固定 key-value 格式的個人資訊，以 JSON 輸出。
        {format_instructions}

        文字: {text}""",
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    return prompt | model | parser
