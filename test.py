from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    FunctionMessage,
    ToolMessage,
)
from langchain_ollama import OllamaLLM, ChatOllama, OllamaEmbeddings
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import OnlinePDFLoader

# from langchain.schema.runnable import RunnablePassthrough
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.schema.runnable import RunnableConfig
from operator import itemgetter
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages.base import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
import sys, os, time, signal
from termcolor import colored
from dotenv import load_dotenv, find_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from rich import print as pprint
import pytz
from datetime import datetime
import bs4
import uuid


timezone = pytz.timezone("Asia/Taipei")

_ = load_dotenv(find_dotenv())  # read local .env file
embeddings_model_name = os.environ.get(
    "EMBEDDINGS_MODEL_NAME", "nomic-embed-text:latest"
)
llm_model_name = os.environ.get(
    "LLM_MODEL_NAME", "gemma3:1b"
)
persist_directory = os.environ.get("PERSIST_DIRECTORY", "db")
target_source_chunks = int(os.environ.get("TARGET_SOURCE_CHUNKS", 4))

llm_config = RunnableConfig(callbacks=[StreamingStdOutCallbackHandler()])
from langchain_community.document_loaders import WebBaseLoader


def signal_handler(sig, frame):
    print("You pressed Ctrl+C!")
    # print(threads_running)
    sys.exit(0)


class SuppressStdout:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")        

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr


def does_vectorstore_exist(persist_directory: str) -> bool:
    """
    Checks if vectorstore exists
    """
    if os.path.exists(os.path.join(persist_directory, "chroma.sqlite3")):
        return True
    return False


def are_models_ready():
    with SuppressStdout():
        model = ChatOllama(  # ChatOllama // OllamaLLM
            model=llm_model_name,
            base_url="http://localhost:11434",
            temperature=0.7,
            # callbacks=[StreamingStdOutCallbackHandler()],
        )
        embed = OllamaEmbeddings(
            base_url="http://localhost:11434", model=embeddings_model_name
        )
        memory = ConversationBufferWindowMemory(
            input_key="query",
            memory_key="chat_history",
            # output_key="output",
            k=5,
            return_messages=False,
        )
    return model, embed, memory

    # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),


def get_datetime():
    now = datetime.now(timezone)
    return now.strftime("%Y%m%d, %H:%M:%S")


def get_question(input):
    if not input:
        return None
    elif isinstance(input, str):
        return input
    elif isinstance(input, dict) and "question" in input:
        return input["question"]
    elif isinstance(input, BaseMessage):
        return input.content
    else:
        raise Exception(
            "string or dict with 'question' key expected as RAG chain input."
        )


def get_message(input):
    #print(colored(str(type(input)) + ": " + f"{input}", "green", attrs=["bold"]))
    return input


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def make_rag_chain(model, retriever, rag_prompt=None):
    # We will use a prompt template from langchain hub.
    if not rag_prompt:
        rag_prompt = hub.pull("aki-rag-prompt")

    rag_chain = (
        {
            "context": RunnableLambda(get_question) | retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | rag_prompt
        | get_message
        | model
    )
    return rag_chain.with_config(llm_config)


def create_memory_chain(model, base_chain, chat_memory):
    if not base_chain:
        base_chain = model
   
    contextualize_q_system_prompt = """
You are an intelligent and friendly in-car assistant.  
You understand and automatically respond in the same language used by the user: **Chinese, English, or Japanese**.  
Your responses must always match the user's language. Never switch languages in your reply.

Respond in a **warm, concise, and emotionally considerate** tone, like a thoughtful companion sitting next to the driver.  
**Do not use any emojis or emoticons.**

When the user shares something meaningful—such as preferences, personal information, or events—acknowledge it kindly and remember it using a unique key for future personalization.

If you are unsure about something, ask politely and gently.

Always avoid repeating words or phrases.  
Write clearly, stay supportive, and keep your tone kind and natural.

"""

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    runnable = (
        get_message
        | contextualize_q_prompt
        | get_message
        # | (lambda output: {"question": output.messages})
        # | get_message
        | base_chain.with_config(llm_config)
    )

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        return chat_memory

    with_message_history = RunnableWithMessageHistory(
        runnable,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )
    return with_message_history


def create_full_chain(model, retriever, memory, rag_prompt=None):
    # chain = make_rag_chain(model, retriever, rag_prompt=rag_prompt)
    chain = None
    if memory:
        chain = create_memory_chain(model, chain, memory)
    return chain


def ensemble_retriever_from_docs(embed):
    if does_vectorstore_exist(persist_directory):
        # Update and store locally vectorstore
        # print(f"Existing vectorstore at {persist_directory}")
        vectorstore = Chroma(
            embedding_function=embed,
            persist_directory=persist_directory,
            collection_name="ctk_media",
            collection_metadata={"ctk_meta": "used for development"},
        )
    else:
        # Create and store locally vectorstore
        # print("Creating new vectorstore")
        # load the pdf and split it into chunks
        # loader = OnlinePDFLoader(
        #     "https://special.moe.gov.tw/_downfile.php?flag=3&fn=old_spc_upload_file/upload_file/all/bb5fc495b6cf4eba4214271f46446a41.pdf"
        # )
        loader = WebBaseLoader(
            "https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=N0030001"
        )
        # loader = WebBaseLoader(
        #     web_paths=("https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=N0030001",),
        #     bs_kwargs=dict(
        #         parse_only=bs4.SoupStrainer(
        #             class_=("post-content", "post-title", "post-header")
        #         )
        #     ),
        # )
        # loader = WebBaseLoader(
        #     "https://gist.githubusercontent.com/aki29/5a0d788e79150f9186b377cff4dba57c/raw/5f056979c753173e5b4989cbbf4ffcdd21d9386b/gistfile1.txt"
        # )
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            add_start_index=False,
        )
        all_splits = text_splitter.split_documents(data)
        vectorstore = Chroma.from_documents(
            documents=all_splits,
            embedding=embed,
            persist_directory=persist_directory,
            collection_name="ctk_media",
            collection_metadata={"ctk_meta": "used for development"},
        )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": target_source_chunks, "score_threshold": 0.7},
    )
    return retriever


def main():
    model, embed, memory = are_models_ready()
    session_id = str(uuid.uuid4())
    ############################################################################ Application Start
    ## sudo apt-get install tesseract-ocr #aki 20241022

    memory = ChatMessageHistory()
    # retriever = ensemble_retriever_from_docs(embed)
    qa_chain = (
        create_full_chain(
            model,
            None,
            memory,
            rag_prompt=ChatPromptTemplate.from_messages(
                (
                    "system",
                    "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. answer question by chinese.\nQuestion: {question} \nContext: {context} \nAnswer:",
                ),
            ),
        )
        | StrOutputParser()
    )
    # "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. answer question by chinese.\nQuestion: {question} \nContext: {context} \nAnswer:",
    # "You are the assistant for the question and answer task. Use the context retrieved below to answer the question. If you need more information you can ask back what information is needed. If you don't know the answer, just say you don't know. Use a maximum of three sentences and keep your answers concise. answer question by chinese.\nQuestion: {question} \nContext: {context} \nAnswer:",
    while True:
        query = input("\nQuery: ")
        if query == "exit":
            break
        if query.strip() == "":
            continue
        start = time.time()
        try:
            # with SuppressStdout():
            #     result = qa_chain.invoke(
            #         {"question": query},
            #         config={"configurable": {"session_id": "foo"}},
            #     )
            # print(result.pretty_print())

            for s in qa_chain.stream(
                {"question": query},
                config={"configurable": {"session_id": session_id}},
            ):
                pass
            print(s, end="", flush=True)
        except Exception as e:
            print(f"Execution Error: {e}")
            continue
        end = time.time()
        elapsed_time = end - start

        print(colored(f" ({elapsed_time:.3f} s)", "blue", attrs=["bold"]))
        # print(
        #     colored("\n" + result + f" ({elapsed_time:.3f} s)", "blue", attrs=["bold"])
        # )

        # print("\n\n> MEM:")
        # memory_content = memory.load_memory_variables({})  # 取得記憶中的變數
        # print(memory_content)
        # print("\n")

        # # Print the result
        # print("\n\n> Question:")
        # print(query)
        # print(answer)

        # # Print the relevant sources used for the answer
        # for document in docs:
        #     print("\n> " + document.metadata["source"] + ":")
        #     print(document.page_content)
        # for idx, entry in enumerate(history, 1):
        # print(f"查詢 {idx}: {entry['query']}\n結果: {entry['result']}\n")


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    main()

