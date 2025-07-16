import ctypes, ctypes.util, atexit
from ctypes import c_char_p, c_int, CFUNCTYPE

libasound = ctypes.cdll.LoadLibrary(ctypes.util.find_library("asound"))

ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)


def _alsa_error_silent(*args):
    pass


c_error_handler = ERROR_HANDLER_FUNC(_alsa_error_silent)

libasound.snd_lib_error_set_handler(c_error_handler)


@atexit.register
def _reset_alsa_handler():
    libasound.snd_lib_error_set_handler(None)


import uvloop, asyncio, os, time, uuid, signal, pytz
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
import emoji, re, sys, math
from pathlib import Path
import riva.client
from riva.client import ASRService, RecognitionConfig, StreamingRecognitionConfig, AudioEncoding
from riva.client import SpeechSynthesisService
from riva.client.proto.riva_audio_pb2 import AudioEncoding as TTSAudioEncoding
import riva.client.audio_io as audio_io

from agent.rag import RAGManager
from audio.asr import VADSource


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
tts_queue = asyncio.Queue()


def signal_handler(sig, frame):
    print("\nInterrupted!!")
    checkpoint_db()
    os.kill(os.getpid(), signal.SIGKILL)
    sys.exit(1)


signal.signal(signal.SIGINT, signal_handler)


async def periodic_checkpoint(interval_sec=600):
    while True:
        await asyncio.sleep(interval_sec)
        checkpoint_db()


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


def init_models():
    critical_cfg = dict(
        model=os.getenv("LLM_MODEL_NAME", "gemma3:4b"),
        base_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        keep_alive=-1,
        num_ctx=1536,
        num_predict=256,
        num_thread=6,
        temperature=0.5,
        top_k=50,
        top_p=0.9,
        repeat_penalty=1.2,
        presence_penalty=0.1,
        stop=["<END>"],
        stream=True,
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
        num_thread=4,
        temperature=0.0,
        top_k=30,
        top_p=0.15,
        repeat_penalty=1.15,
        # seed=42,
        stop=["<END>"],
    )

    mem = ChatOllama(**mem_cfg, cache=True)

    # ChatOllama.get_token_ids = lambda self, text: text.split()

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


async def main():
    # PUNCT_RE = re.compile(r"[，、,。.！!？?:：;；…\-—「」『』‘’“”*]")
    PUNCT_RE = re.compile(r"[，、,。.！!？?：;；…\-—「」『』‘’“”*]")
    model, embed, mem = init_models()
    global rag_manager
    rag_mod.rag_manager = RAGManager(embed, store_dir=Path(".cache/rag/"))
    await rag_mod.rag_manager.ainit()
    await init_db()  # ctk_user.sqlite3
    await asyncio.gather(
        warmup_models(model, rag_mod.rag_manager),
    )
    # ------ 初始化 Riva ASR ------
    auth = riva.client.Auth(
        ssl_cert=None, use_ssl=False, uri=os.getenv("RIVA_URI", "localhost:50051")
    )
    asr = ASRService(auth)

    recog_cfg = RecognitionConfig(
        encoding=AudioEncoding.LINEAR_PCM,
        language_code="zh-CN",
        # language_code="ja-JP",
        # language_code="en-US",
        sample_rate_hertz=16000,
        audio_channel_count=1,
        max_alternatives=1,
        enable_automatic_punctuation=True,
        # enable_voice_activity_events=True,
        # enable_noise_reduction=True,
    )
    stream_cfg = StreamingRecognitionConfig(
        config=recog_cfg,
        interim_results=True,
    )
    # ----------------------------------------
    auth_tts = riva.client.Auth(
        ssl_cert=None, use_ssl=False, uri=os.getenv("RIVA_URI", "localhost:50051")
    )
    tts = SpeechSynthesisService(auth_tts)
    RIVA_VOICE = "Mandarin-CN.Male-Happy"
    TTS_SR = 22050

    sound_stream = audio_io.SoundCallBack(
        output_device_index=None, sampwidth=2, nchannels=1, framerate=TTS_SR
    )
    # -----------------------------

    user_id = (await ainput("Please enter your user ID: ")).strip() or str(uuid.uuid4())
    await load_memory(user_id)
    mem_mgr = MemoryManager(mem, session_id=user_id, max_messages=12, token_limit=512)
    chains = get_chat_chain(user_id, model, mem_mgr, rag_mod.rag_manager)
    stream_chain = chains["stream"]
    # invoke_chain = chains["invoke"]
    asyncio.create_task(periodic_checkpoint(60))

    async def listen_asr(asr_service: ASRService, stream_cfg: StreamingRecognitionConfig) -> str:
        def sync_recognize():
            # with mute_alsa():
            vad = VADSource(rate=16000, frame_duration_ms=30, padding_duration_ms=300)
            responses = asr_service.streaming_response_generator(
                audio_chunks=vad,
                streaming_config=stream_cfg,
            )
            final_text = ""
            for resp in responses:
                for res in resp.results:
                    transcript = res.alternatives[0].transcript.strip()
                    if not res.is_final:
                        print(f"\r[Interim] {transcript}", end="", flush=True)
                    else:
                        if len(transcript) < 2:
                            continue
                        print(f"\r[Final]   {transcript}{' ' * 10}")
                        return transcript.strip()
            return ""

        try:
            return await asyncio.to_thread(sync_recognize)
        except KeyboardInterrupt:
            print("\n[ASR aborted]")
            return ""

    async def speak_tts(text: str):
        def _synth_and_play():
            for resp in tts.synthesize_online(
                text=text,
                voice_name=RIVA_VOICE,
                language_code="zh-CN",
                sample_rate_hz=TTS_SR,
                encoding=TTSAudioEncoding.LINEAR_PCM,
            ):
                sound_stream(resp.audio)

        await asyncio.to_thread(_synth_and_play)

    tts_queue = asyncio.Queue()

    async def tts_worker():
        while True:
            sentence = await tts_queue.get()
            # print("##", sentence)
            if sentence is None:
                break
            await speak_tts(sentence)

    tts_worker_task = asyncio.create_task(tts_worker())
    print("\n[In-Car Assistant STREAMING mode. Type /exit to end.]")
    try:
        while True:
            query = (await ainput("\nQuery: ")).strip()
            if not query:
                continue

            # print("Voice input started. Please speak…")
            # query = await listen_asr(asr, stream_cfg)

            if not query:
                print("[!️] No text or voice was received. Please try again.")
                continue

            start = time.perf_counter()
            response_text = ""
            try:
                first_word = None
                buf = ''
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
                    # text = PUNCT_RE.sub("", text)
                    # text = re.sub(r"[，、,。.！!？?:：;；…\-—「」『』‘’“”\*]", "", text)
                    text = emoji.replace_emoji(text, replace="")
                    if not first_word:
                        first_word = time.perf_counter()
                    print(colored(text, "green"), end="", flush=True)
                    response_text += text
                    buf += text
                    # if any(p in buf[-1:] for p in "。.!！?？") or len(buf) >= 20:
                    if any(p in buf[-1:] for p in "，,。.!！?？"):
                        # print("@@", buf, len(buf))
                        await tts_queue.put(buf)
                        buf = ""
                    # response_text += chunk
                print()  # newline after streaming

                # full_reply = await invoke_chain.ainvoke(
                #     {"question": query},
                #     config={"configurable": {"session_id": user_id}},
                # )
                # response_text = full_reply
                # print(colored(response_text, "green"), end="", flush=True)
                if buf.strip():
                    await tts_queue.put(buf)
                if response_text.strip() == "":
                    print("bye！")
                    break
                else:
                    await mem_mgr.save_turn(query, response_text)
                    # asyncio.create_task(speak_tts(response_text))
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
        await tts_queue.put(None)
        await tts_worker_task
        sound_stream.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting program by user request.")
        checkpoint_db()
        sys.exit(0)
