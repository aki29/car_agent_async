import asyncio, os, time, uuid, signal, pytz
from termcolor import colored
from dotenv import load_dotenv, find_dotenv
from aioconsole import ainput
from langchain_ollama import ChatOllama, OllamaEmbeddings

from agent.chains.chat_chain_async import get_chat_chain
from agent.memory.manager_async import init_db
from langchain.globals import set_debug, set_verbose

# set_debug(True)
# set_verbose(True)

load_dotenv(find_dotenv())

timezone = pytz.timezone("Asia/Taipei")


def signal_handler(sig, frame):
    print("\nInterrupted!")
    raise SystemExit


signal.signal(signal.SIGINT, signal_handler)


def init_models():
    llm = ChatOllama(
        model=os.getenv("LLM_MODEL_NAME", "gemma3:1b"),
        base_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        temperature=0.7,
        streaming=True,  # ★ Enable streaming
    )
    embed = OllamaEmbeddings(
        base_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        model=os.getenv("EMBEDDINGS_MODEL_NAME", "nomic-embed-text:latest"),
    )
    return llm, embed


# from langchain.callbacks.base import AsyncCallbackHandler
# class PrintCallback(AsyncCallbackHandler):
#     async def on_chain_start(self, serialized, *args, **kwargs):
#         if serialized:
#             print(f"Start {serialized}")

#     async def on_chain_end(self, outputs, **_):
#         if outputs:
#             print(f"End → {outputs}")

#     async def on_llm_new_token(self, token, **_):
#         print(token, end="", flush=True)


async def main():
    model, embed = init_models()
    await init_db()

    user_id = (await ainput("Please enter your user ID: ")).strip() or str(uuid.uuid4())
    chat = get_chat_chain(user_id, model)

    print("\n[In-Car Assistant STREAMING mode.  Type /exit to end.]")
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

            if response_text.strip() == "[exit]":
                print("bye！")
                break

        except Exception as e:
            print(colored(f"[error] {e}", "red"))
        finally:
            elapsed = time.perf_counter() - start
            print(colored(f"({elapsed:.2f}s)", "blue"))


if __name__ == "__main__":
    asyncio.run(main())
