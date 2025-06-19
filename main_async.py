import asyncio, os, time, uuid, signal, pytz
from termcolor import colored
from dotenv import load_dotenv, find_dotenv
from aioconsole import ainput
from langchain_ollama import ChatOllama, OllamaEmbeddings

from agent.chains.chat_chain_async import get_chat_chain
from agent.memory.manager_async import init_db
from langchain.globals import set_debug, set_verbose, set_llm_cache
from langchain_core.caches import InMemoryCache

# set_debug(True)
# set_verbose(True)
# set_llm_cache(None)           # stop cache
set_llm_cache(InMemoryCache())  # new cache everytime

load_dotenv(find_dotenv())

timezone = pytz.timezone("Asia/Taipei")

# import re
# PINYIN_RE = re.compile(r'\(([A-Za-zāáǎàēéěèīíǐìōóǒòūúǔùüǖǘǚǜĀÁǍÀĒÉĚÈĪÍǏÌŌÓǑÒŪÚǓÙÜǕǗǙǛ\s,]+)\)')
# def strip_romanization(text: str) -> str:
#     return PINYIN_RE.sub("", text)


def signal_handler(sig, frame):
    print("\nInterrupted!")
    raise SystemExit


signal.signal(signal.SIGINT, signal_handler)


def init_models():
    critical_cfg = dict(
        model=os.getenv("LLM_MODEL_NAME", "gemma3:1b"),
        base_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        keep_alive=-1,
        num_ctx=768,
        num_predict=48,
        num_thread=2,
        temperature=0.3,
        top_k=15,
        top_p=0.8,
        stop=["\n\n", "<END>"],
    )
    causal_cfg = dict(
        model=os.getenv("LLM_MODEL_NAME", "gemma3:1b"),
        base_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        keep_alive=-1,
        num_ctx=1024,
        num_predict=96,
        num_thread=2,
        temperature=0.7,
        top_k=40,
        top_p=0.9,
        stop=["\n\n", "<END>"],
    )
    llm = ChatOllama(**causal_cfg, cache=True)
    embed = OllamaEmbeddings(
        base_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        model=os.getenv("EMBEDDINGS_MODEL_NAME", "nomic-embed-text:latest"),
        keep_alive=-1,
        num_thread=4,
        # num_ctx=512,
    )
    mem_cfg = dict(
        model=os.getenv("MEM_MODEL_NAME", "gemma3:1b"),
        base_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        keep_alive=-1,
        # num_ctx=768,
        # num_predict=48,
        num_thread=2,
        temperature=0,
        # top_k=15,
        # top_p=0.8,
        stop=["\n\n", "<END>"],
    )
    mem = ChatOllama(**mem_cfg, cache=True)
    return llm, embed, mem


async def main():
    model, embed, _ = init_models()
    await init_db()

    user_id = (await ainput("Please enter your user ID: ")).strip() or str(uuid.uuid4())
    chat = get_chat_chain(user_id, model)

    print("\n[In-Car Assistant STREAMING mode. Type /exit to end.]")
    while True:
        query = (await ainput("\nQuery: ")).strip()
        if not query:
            continue

        start = time.perf_counter()
        response_text = ""
        try:
            async for chunk in chat.astream(
                # {"question": query}, config={"configurable": {"session_id": user_id}}
                {"question": query},
                config={
                    "configurable": {"session_id": user_id},
                    # "callbacks": [PrintCallback()],
                    # "tags": ["debug"],
                },
            ):
                print(colored(chunk, "green"), end="", flush=True)
                response_text += chunk
            print()  # newline after streaming

            if response_text.strip() == "":
                print("bye！")
                break

        except Exception as e:
            print(colored(f"[error] {e}", "red"))
        finally:
            elapsed = time.perf_counter() - start
            print(colored(f"({elapsed:.2f}s)", "blue"))


if __name__ == "__main__":
    asyncio.run(main())
