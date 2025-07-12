import asyncio, os, time, uuid, signal, pytz
from termcolor import colored
from dotenv import load_dotenv, find_dotenv
from aioconsole import ainput
from langchain_ollama import ChatOllama, OllamaEmbeddings
from agent.chains.chat_chain_async import get_chat_chain
from agent.memory.manager_async import init_db, load_memory
from langchain.globals import set_debug, set_verbose, set_llm_cache
from langchain_core.caches import InMemoryCache
from langchain_community.cache import SQLiteCache
from agent.cache.async_and_fuzzy_cache import AsyncSQLiteCache
from agent.memory.engine import checkpoint_db
from agent.memory.manager import MemoryManager
from agent.rag import RAGManager
import agent.rag as rag_mod
import re
import emoji
from pathlib import Path

# set_debug(True)
# set_verbose(True)
# set_llm_cache(None)           # stop cache
# set_llm_cache(InMemoryCache())  # new cache everytime
use_cache = os.getenv("USE_LLM_CACHE", "false").lower() == "true"
if use_cache:
    os.makedirs(".cache", exist_ok=True)
    set_llm_cache(SQLiteCache(database_path=".cache/cache.db"))
    # print("[cache] SQLite LLM cache enabled.")
    # set_llm_cache(AsyncSQLiteCache(db_path=".cache/langchain.db"))
    # print("[cache] AsyncSQLite LLM cache enabled.")
else:
    set_llm_cache(InMemoryCache())
    # print("[cache] In-memory (ephemeral) LLM cache enabled.")

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
        # num_predict=128,
        num_thread=6,
        temperature=0.4,
        top_k=50,
        top_p=0.9,
        repeat_penalty=1.2,
        presence_penalty=0.1,
        # stop=["\n\n", "<END>"],
        stop=["<END>"],
        stream=True,
        # cache=True,
    )

    llm = ChatOllama(**critical_cfg, cache=True)
    embed = OllamaEmbeddings(
        model=os.getenv("EMBEDDINGS_MODEL_NAME", "nomic-embed-text:latest"),
        base_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        keep_alive=-1,
        num_thread=4,
        # num_ctx=512,
    )

    mem_cfg = dict(
        model=os.getenv("MEM_MODEL_NAME", "phi3:latest"),  # or gemma:1b
        base_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        keep_alive=-1,
        num_ctx=1024,
        num_predict=24,
        num_thread=6,
        temperature=0.0,
        top_k=30,
        top_p=0.15,
        repeat_penalty=1.15,
        seed=42,
        stop=["<END>"],
    )

    mem = ChatOllama(**mem_cfg, cache=True)

    # ChatOllama.get_token_ids = lambda self, text: text.split()

    import math

    def _cheap_tokenizer(self, text: str):

        n = max(1, math.ceil(len(text.encode('utf-8')) / 4))
        return [None] * n

    ChatOllama.get_token_ids = _cheap_tokenizer

    return llm, embed, mem


async def warmup_models(llm, rag_manager):
    async def _safe_warmup(rag, name: str):
        try:
            await rag.aretrieve("testing")
            # print(f"[warmup] {name} ready.")
        except Exception as e:
            print(f"[warmup] {name} failed: {e}")

    await llm.ainvoke("ping")
    await asyncio.gather(
        _safe_warmup(rag_manager.poi, "poi"),
        _safe_warmup(rag_manager.parking, "parking"),
        _safe_warmup(rag_manager.manual, "manual"),
        _safe_warmup(rag_manager.food, "food"),
    )
    # print("[warmup] All RAGs initialized.\n")


PUNCT_RE = re.compile(r"[，、,。.！!？?:：;；…\-—「」『』‘’“”*]")


async def main():
    model, embed, mem = init_models()
    global rag_manager
    rag_mod.rag_manager = RAGManager(embed, store_dir=Path(".cache/rag/"))
    await rag_mod.rag_manager.ainit()
    await init_db()  # ctk_user.sqlite3
    await asyncio.gather(
        warmup_models(model, rag_mod.rag_manager),
    )
    user_id = (await ainput("Please enter your user ID: ")).strip() or str(uuid.uuid4())
    await load_memory(user_id)
    mem_mgr = MemoryManager(mem, session_id=user_id, max_messages=12, token_limit=512)
    chains = get_chat_chain(user_id, model, mem_mgr, rag_mod.rag_manager)
    stream_chain = chains["stream"]
    # invoke_chain = chains["invoke"]
    asyncio.create_task(periodic_checkpoint(60))
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
                first_word = None
                async for chunk in stream_chain.astream(
                    {"question": query},
                    config={
                        # "stream": True,
                        "configurable": {"session_id": user_id},
                    },
                ):
                    if hasattr(chunk, "content"):
                        text = chunk.content
                    else:
                        text = str(chunk)
                    text = PUNCT_RE.sub("", text)
                    # text = re.sub(r"[，、,。.！!？?:：;；…\-—「」『』‘’“”\*]", "", text)
                    text = emoji.replace_emoji(text, replace="")
                    if not first_word:
                        first_word = time.perf_counter()
                    print(colored(text, "green"), end="", flush=True)
                    response_text += text
                    # response_text += chunk
                print()  # newline after streaming

                # full_reply = await invoke_chain.ainvoke(
                #     {"question": query},
                #     config={"configurable": {"session_id": user_id}},
                # )
                # response_text = full_reply
                # print(colored(response_text, "green"), end="", flush=True)

                if response_text.strip() == "":
                    print("bye！")
                    break
                else:
                    await mem_mgr.save_turn(query, response_text)
                    # speak_tts(ai_text)

            except Exception as e:
                print(colored(f"[error] {e}", "red"))
            finally:
                ttfb = (first_word or start) - start
                total = time.perf_counter() - start
                print(colored(f"(TTFB {ttfb:.2f}s, Total {total:.2f}s)", "blue"))

            # elapsed = time.perf_counter() - start
            # print(colored(f"({elapsed:.2f}s)", "blue"))
            # if first_word:
            #     elapsed = first_word - start
            #     print(colored(f"({elapsed:.2f}s)", "blue"))
            #     first_word = 0
    except Exception as e:
        print(colored(f"[system error] {e}", "red"))
    finally:
        checkpoint_db()


if __name__ == "__main__":
    asyncio.run(main())
