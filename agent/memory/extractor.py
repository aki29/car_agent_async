from langchain_core.runnables import RunnableLambda

def extract_memory_kv_chain(model):
    """Placeholder extractor that returns an empty dict."""
    async def _noop(inputs):
        return {}
    return RunnableLambda(_noop)
