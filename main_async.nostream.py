import asyncio, os, time, uuid, signal, pytz
from termcolor import colored
from dotenv import load_dotenv
from aioconsole import ainput

from agent.chains.chat_chain_async import get_chat_chain
from agent.memory.manager_async import init_db
from agent.rag.retriever import build_retriever_async

# Load .env if present
load_dotenv()

timezone = pytz.timezone("Asia/Taipei")

def signal_handler(sig, frame):
    print("\nInterrupted!!")
    raise SystemExit

signal.signal(signal.SIGINT, signal_handler)

# Dynamically import models from LangChain
from langchain_ollama import ChatOllama, OllamaEmbeddings

def init_model():
    return (
        ChatOllama(
            model=os.getenv("LLM_MODEL_NAME", "gemma3:1b"),
            base_url="http://localhost:11434",
            temperature=0.7,
        ),
        OllamaEmbeddings(
            model=os.getenv("EMBEDDINGS_MODEL_NAME", "nomic-embed-text:latest"),
            base_url="http://localhost:11434",
        ),
    )

async def main():
    model, embed = init_model()

    # Initialize database and vector retriever concurrently
    await init_db()
    retriever = await build_retriever_async(embed)

    user_id = (await ainput("User ID: ")).strip() or str(uuid.uuid4())
    chat = get_chat_chain(user_id, model, retriever)

    print("\n[In-Car Assistant async demo. /exit to quit]")
    while True:
        query = (await ainput("\nYou> ")).strip()
        if not query:
            continue
        start = time.perf_counter()
        result = await chat.ainvoke(
            {"question": query},
            config={"configurable": {"session_id": user_id}},
        )
        if result == "[exit]":
            break
        print(colored(f"Assistant> {result}", "green"))
        print(colored(f"({time.perf_counter()-start:.2f}s)", "blue"))

if __name__ == "__main__":
    asyncio.run(main())
