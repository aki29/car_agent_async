import asyncio, os, time, uuid, signal, pytz
from termcolor import colored
from dotenv import load_dotenv, find_dotenv
from aioconsole import ainput
from langchain_ollama import ChatOllama, OllamaEmbeddings
from agent.chains.chat_chain_async import get_chat_chain
from agent.memory.manager_async import init_db
from langchain.globals import set_debug, set_verbose, set_llm_cache
from langchain_core.caches import InMemoryCache
from agent.memory.engine import checkpoint_db
from agent.memory.manager import MemoryManager


# set_debug(True)
# set_verbose(True)
# set_llm_cache(None)           # stop cache
set_llm_cache(InMemoryCache())  # new cache everytime

# import warnings
# warnings.filterwarnings("ignore", category=UserWarning, module="langchain.memory")
# warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
# os.environ["TRANSFORMERS_NO_TF"] = "1"
# os.environ["TRANSFORMERS_NO_FLAX"] = "1"
# os.environ["TRANSFORMERS_NO_PYTORCH"] = "1"

load_dotenv(find_dotenv())


def signal_handler(sig, frame):
    print("\nInterrupted!!")
    checkpoint_db()
    raise SystemExit


signal.signal(signal.SIGINT, signal_handler)


async def periodic_checkpoint(interval_sec=600):
    while True:
        await asyncio.sleep(interval_sec)
        checkpoint_db()


def init_models():
    critical_cfg = dict(
        model=os.getenv("LLM_MODEL_NAME", "gemma3:4b"),
        base_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        keep_alive=-1,
        num_ctx=1536,
        num_predict=64,
        num_thread=6,
        temperature=0.4,
        top_k=50,
        top_p=0.9,
        repeat_penalty=1.2,
        presence_penalty=0.1,
        stop=["\n\n", "<END>"],
    )

    llm = ChatOllama(**critical_cfg, cache=True)
    embed = OllamaEmbeddings(
        base_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        model=os.getenv("EMBEDDINGS_MODEL_NAME", "nomic-embed-text:latest"),
        keep_alive=-1,
        num_thread=4,
        # num_ctx=512,
    )

    mem_cfg = dict(
        model=os.getenv("MEM_MODEL_NAME", "phi3:latest"),  # or gemma:1b
        base_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        keep_alive=-1,
        num_ctx=512,
        num_predict=64,
        num_thread=4,
        temperature=0.15,
        top_k=30,
        top_p=0.9,
        repeat_penalty=1.2,
        stop=["<END>"],
    )
    mem = ChatOllama(**mem_cfg, cache=True)
    return llm, embed, mem


async def main():
    model, embed, mem = init_models()
    await init_db()
    user_id = (await ainput("Please enter your user ID: ")).strip() or str(uuid.uuid4())
    mem_mgr = MemoryManager(mem, session_id=user_id, max_messages=12, token_limit=20)
    chat = get_chat_chain(user_id, model, mem_mgr)
    asyncio.create_task(periodic_checkpoint())
    print("\n[In-Car Assistant STREAMING mode. Type /exit to end.]")
    try:
        while True:
            query = (await ainput("\nQuery: ")).strip()
            if not query:
                continue
            # user_text = await listen_asr()

            start = time.perf_counter()
            response_text = ""
            try:
                async for chunk in chat.astream(
                    {"question": query},
                    config={
                        "configurable": {"session_id": user_id},
                    },
                ):
                    print(colored(chunk, "green"), end="", flush=True)
                    response_text += chunk
                print()  # newline after streaming

                if response_text.strip() == "":
                    print("bye！")
                    break
                else:
                    pass
                    # speak_tts(ai_text)

            except Exception as e:
                print(colored(f"[error] {e}", "red"))
            finally:
                elapsed = time.perf_counter() - start
                print(colored(f"({elapsed:.2f}s)", "blue"))
    except Exception as e:
        print(colored(f"[system error] {e}", "red"))
    finally:
        checkpoint_db()


if __name__ == "__main__":
    asyncio.run(main())
