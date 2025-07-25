import atexit, asyncio, ctypes, ctypes.util, os, re, signal, sys, time, uuid, emoji, math
from pathlib import Path
from termcolor import colored

# from audio.io import synth_and_enqueue

ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(
    None, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p
)


def _alsa_error_silent(*_: object) -> None:
    pass


libasound = ctypes.cdll.LoadLibrary(ctypes.util.find_library("asound"))
_alsa_handler = ERROR_HANDLER_FUNC(_alsa_error_silent)
libasound.snd_lib_error_set_handler(_alsa_handler)


@atexit.register
def _cleanup_on_exit() -> None:
    try:
        libasound.snd_lib_error_set_handler(None)
        print("[ALSA] Error handler reset.")
    except Exception as e:
        print("[ALSA] Reset failed:", e)

    try:
        from audio.vad import VADSource

        for stream in list(VADSource._streams.values()):
            try:
                if stream.is_active():
                    stream.stop_stream()
                stream.close()
            except Exception:
                pass
        VADSource._streams.clear()
        VADSource._pya.terminate()
        print("[PyAudio] Terminated cleanly.")
        time.sleep(0.1)
    except Exception as e:
        print("[PyAudio] Terminate failed:", e)


import uvloop

uvloop.install()

from aioconsole import ainput
from dotenv import load_dotenv, find_dotenv
from langchain_community.cache import SQLiteCache
from langchain_core.caches import InMemoryCache
from langchain_ollama import ChatOllama, OllamaEmbeddings

from agent.chains.chat_chain_async import get_chat_chain
from agent.memory.engine import checkpoint_db
from agent.memory.manager import MemoryManager
from agent.memory.manager_async import init_db, load_memory
from agent.rag import RAGManager, rag_manager as global_rag_manager
from speech.service import SpeechService
from audio.io import mic_stream, play_wav, _PLAY_Q, _PLAYER_TASK, _ensure_player

import opencc

cc = opencc.OpenCC('s2tw.json')  # s2t=通用繁體, s2tw=台灣正體, s2hk=香港繁體…
# print(cc.convert("汉字转换工具how are you"))

load_dotenv(find_dotenv())


def _sigint_handler(sig, frame):
    print("Interrupted!!")
    checkpoint_db()
    if _PLAYER_TASK is not None:
        _PLAYER_TASK.cancel()
    sys.exit(0)
    # os.kill(os.getpid(), signal.SIGKILL)
    # raise KeyboardInterrupt


signal.signal(signal.SIGINT, _sigint_handler)

PUNCT_RE = re.compile(r"[，、,。.！!？?：;；…\-—「」『』‘’“”*]")
RIVA_URI = os.getenv("RIVA_URI", "localhost:50051")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL_NAME", "gemma3:4b")
EMBED_MODEL = os.getenv("EMBEDDINGS_MODEL_NAME", "nomic-embed-text:latest")
MEM_MODEL = os.getenv("MEM_MODEL_NAME", "phi3:latest")
VOICE_NAME = os.getenv("RIVA_VOICE", "Mandarin-CN.Male-Happy")
TTS_SR = 22050
VOICE_MODE = os.getenv("VOICE_MODE", "false").lower() == "true"
TTS_MODE = os.getenv("TTS_MODE", "true").lower() == "true"
USE_CACHE = os.getenv("USE_LLM_CACHE", "false").lower() == "true"
if USE_CACHE:
    os.makedirs(".cache", exist_ok=True)
    from langchain.globals import set_llm_cache

    set_llm_cache(SQLiteCache(database_path=".cache/cache.db"))
else:
    from langchain.globals import set_llm_cache

    set_llm_cache(InMemoryCache())

# from audio.webrtc_frontend import WebRTCAudioFrontend

# # apm = WebRTCAudioFrontend(rate=16000, channels=1)
# apm = WebRTCAudioFrontend(rate=16000, channels=1, aec=0, ns=True, agc=0, vad=False)
# apm.apm.set_ns_level(1)  # 噪音抑制 0‑3，建議 2
# # apm.apm.set_agc_level(2)  # AGC 目標 dBFS，數值大 → 輸出較小聲
# # apm.apm.set_agc_target(5)
# # apm.apm.set_aec_level(0)  # AEC 0=Low, 1=Moderate, 2=High
# # apm.apm.set_vad_level(0)  # VAD 0=敏感，3=嚴格


def init_models():
    llm_cfg = dict(
        model=LLM_MODEL,
        base_url=OLLAMA_URL,
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
    llm = ChatOllama(**llm_cfg, cache=True)
    embed = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL, keep_alive=-1, num_thread=4)
    mem_cfg = dict(
        model=MEM_MODEL,
        base_url=OLLAMA_URL,
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
    mem_llm = ChatOllama(**mem_cfg, cache=True)

    def _cheap(self, text: str):
        return [None] * max(1, math.ceil(len(text.encode("utf-8")) / 4))

    ChatOllama.get_token_ids = _cheap
    return llm, embed, mem_llm


async def main() -> None:
    llm, embed, mem_llm = init_models()
    global_rag_manager = RAGManager(embed, store_dir=Path(".cache/rag/"))
    await global_rag_manager.ainit()
    await init_db()

    user_id = (await ainput("User ID: ")).strip() or str(uuid.uuid4())
    await load_memory(user_id)
    mem_mgr = MemoryManager(mem_llm, session_id=user_id, max_messages=12, token_limit=512)

    chains = get_chat_chain(user_id, llm, mem_mgr, global_rag_manager)
    stream_chain = chains["stream"]

    _ensure_player(asyncio.get_running_loop())
    speech = SpeechService(host=RIVA_URI)
    # from audio.io import stop_playing

    # async def detect_barge_in(speech: SpeechService):
    #     async for chunk in speech.transcribe(mic_stream()):
    #         if len(chunk.strip()) > 2:
    #             stop_playing()
    #             return chunk

    print("\n[Car‑Agent READY]")

    # async def wait_for_play_queue():
    #     if _PLAY_Q is not None:
    #         await _PLAY_Q.join()

    async def tts_worker(speech, voice, sr):
        while True:
            text = await tts_q.get()
            if text is None:
                break
            wav = await speech.synth(text.strip(), voice=voice, sr=sr)
            await play_wav(wav)
            tts_q.task_done()

    tts_q = asyncio.Queue()
    asyncio.create_task(tts_worker(speech, VOICE_NAME, TTS_SR))

    async def wait_tts_done():
        if tts_q is not None:
            await tts_q.join()
        if _PLAY_Q is not None:
            await _PLAY_Q.join()

    SENT_END = "，,。.!！?？、"
    MAX_BUF = 40
    try:
        while True:
            if VOICE_MODE:
                print(colored("\nVoice input started…", "cyan"))
                query = await speech.listen_and_transcribe(lang="zh-CN", sr=16000, device=None)
                # print("ASR:", query)
                query = cc.convert(query)
                print(colored(f"\n{query}", "white"))
                # continue
            else:
                query = (await ainput(colored("\nQuery: ", "cyan"))).strip()

            if not query:
                continue

            if query.lower() in {"/exit", "退出", "離開", "离开"}:
                break

            first_token = time.perf_counter()
            buf, full_reply = "", ""
            async for chunk in stream_chain.astream(
                {"question": query}, config={"configurable": {"session_id": user_id}}
            ):
                text = chunk.content if hasattr(chunk, "content") else str(chunk)
                if text:
                    text = emoji.replace_emoji(text, replace="")
                    need_flush = False
                    for ch in text:
                        if ch:
                            buf += ch
                            print(colored(ch, "green"), end="", flush=True)
                        if ch in SENT_END:
                            need_flush = True
                            # print("#T#")
                        elif len(buf) >= MAX_BUF:
                            # print("#L#")
                            need_flush = True
                        # if need_flush and buf.strip():
                        if need_flush:
                            full_reply += buf
                            # wav = await speech.synth(buf.strip(), voice=VOICE_NAME, sr=TTS_SR)
                            # await play_wav(wav)
                            # asyncio.create_task(
                            #     synth_and_enqueue(buf.strip(), speech, VOICE_NAME, TTS_SR)
                            # )
                            await tts_q.put(buf.strip())
                            buf = ""
                            need_flush = False
            # print("@@")
            await mem_mgr.save_turn(query, full_reply)
            print(colored(f"\n(Total {(time.perf_counter()-first_token):.2f}s)", "blue"))
            await wait_tts_done()
    except (EOFError, KeyboardInterrupt):
        print("\n[Quit]")
    finally:
        checkpoint_db()
        if _PLAYER_TASK is not None:
            await _PLAY_Q.join()
            _PLAYER_TASK.cancel()
        # os.kill(os.getpid(), signal.SIGKILL)
        # sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except asyncio.CancelledError:
        print("\n[System] Cancelled by user or task")
    except KeyboardInterrupt:
        print("\n[System] Interrupted by keyboard")
