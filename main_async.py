import atexit, asyncio, ctypes, ctypes.util
import os, re, signal, sys, time, uuid, emoji, math
from pathlib import Path
from termcolor import colored

ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(
    None, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p
)


def _alsa_error_silent(*_: object) -> None:
    pass


libasound = ctypes.cdll.LoadLibrary(ctypes.util.find_library("asound"))
_alsa_handler = ERROR_HANDLER_FUNC(_alsa_error_silent)
libasound.snd_lib_error_set_handler(_alsa_handler)


@atexit.register
def _reset_alsa_handler() -> None:
    libasound.snd_lib_error_set_handler(None)


import uvloop  # type: ignore

uvloop.install()

from aioconsole import ainput
from dotenv import load_dotenv, find_dotenv
from langchain_community.cache import SQLiteCache
from langchain_core.caches import InMemoryCache
from langchain_ollama import ChatOllama, OllamaEmbeddings

import riva.client
import riva.client.audio_io as audio_io
from riva.client import ASRService, SpeechSynthesisService
from riva.client.proto.riva_audio_pb2 import AudioEncoding as TTSAudioEncoding
from riva.client import RecognitionConfig, StreamingRecognitionConfig, AudioEncoding

from agent.chains.chat_chain_async import get_chat_chain
from agent.memory.engine import checkpoint_db
from agent.memory.manager import MemoryManager
from agent.memory.manager_async import init_db, load_memory
from agent.rag import RAGManager, rag_manager as global_rag_manager
from audio.vad import VADSource
import opencc

cc = opencc.OpenCC('s2tw.json')  # s2t=通用繁體, s2tw=台灣正體, s2hk=香港繁體…
# print(cc.convert("汉字转换工具how are you"))

load_dotenv(find_dotenv())

import os  # already imported above, safe to import twice in small scripts


def _sigint_handler(sig, frame):
    print("Interrupted!!")
    checkpoint_db()
    os.kill(os.getpid(), signal.SIGKILL)
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

from audio.webrtc_frontend import WebRTCAudioFrontend

# apm = WebRTCAudioFrontend(rate=16000, channels=1)
apm = WebRTCAudioFrontend(rate=16000, channels=1, aec=0, ns=True, agc=0, vad=False)
apm.apm.set_ns_level(1)  # 噪音抑制 0‑3，建議 2
# apm.apm.set_agc_level(2)  # AGC 目標 dBFS，數值大 → 輸出較小聲
# apm.apm.set_agc_target(5)
# apm.apm.set_aec_level(0)  # AEC 0=Low, 1=Moderate, 2=High
# apm.apm.set_vad_level(0)  # VAD 0=敏感，3=嚴格

auth = riva.client.Auth(ssl_cert=None, use_ssl=False, uri=RIVA_URI)
asr_service = ASRService(auth)
tts_service = SpeechSynthesisService(auth)

sound_stream = audio_io.SoundCallBack(
    output_device_index=None, sampwidth=2, nchannels=1, framerate=TTS_SR
)


TTS_QUEUE: asyncio.Queue[str | None] = asyncio.Queue()


async def speak_tts(text: str) -> None:
    def _synth_and_play() -> None:
        for resp in tts_service.synthesize_online(
            text=text,
            voice_name=VOICE_NAME,
            language_code="zh-CN",
            sample_rate_hz=TTS_SR,
            encoding=TTSAudioEncoding.LINEAR_PCM,
        ):
            if resp.audio:
                apm.feed_far_end(resp.audio, src_rate=TTS_SR)  # WEBRTC
                sound_stream(resp.audio)

    await asyncio.get_running_loop().run_in_executor(None, _synth_and_play)


async def tts_worker() -> None:
    while True:
        sentence = await TTS_QUEUE.get()
        if sentence is None:
            TTS_QUEUE.task_done()
            break
        if TTS_MODE:
            await speak_tts(sentence)
        TTS_QUEUE.task_done()


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


REC_CFG = RecognitionConfig(
    encoding=AudioEncoding.LINEAR_PCM,
    language_code="zh-CN",
    # language_code="ja-JP",
    sample_rate_hertz=16000,
    audio_channel_count=1,
    max_alternatives=1,
    enable_automatic_punctuation=True,
)
STR_CFG = StreamingRecognitionConfig(config=REC_CFG, interim_results=True)
# single_utterance=True


async def listen_once() -> str:
    vad = VADSource(
        rate=16000,
        frame_duration_ms=30,
        padding_duration_ms=300,
        vad_mode=1,
        start_ratio=0.6,
        end_ratio=0.4,
        apm=None,
    )

    def _sync() -> str:
        for resp in asr_service.streaming_response_generator(vad, STR_CFG):
            for res in resp.results:
                if res.is_final and res.alternatives:
                    return res.alternatives[0].transcript.strip()
        return ""

    return await asyncio.to_thread(_sync)


async def main() -> None:
    llm, embed, mem_llm = init_models()

    global_rag_manager = RAGManager(embed, store_dir=Path(".cache/rag/"))
    await global_rag_manager.ainit()

    await init_db()

    tts_task = asyncio.create_task(tts_worker())

    user_id = (await ainput("User ID: ")).strip() or str(uuid.uuid4())
    await load_memory(user_id)
    mem_mgr = MemoryManager(mem_llm, session_id=user_id, max_messages=12, token_limit=512)

    chains = get_chat_chain(user_id, llm, mem_mgr, global_rag_manager)
    stream_chain = chains["stream"]

    print("\n[Car‑Agent READY]")
    try:
        while True:
            await TTS_QUEUE.join()
            if VOICE_MODE:
                print(colored("\nVoice input started…", "cyan"))
                query = await listen_once()
                query = query.strip().rstrip("。.")
                if query:
                    query = cc.convert(query)
                    print(colored(f"\n{query}", "white"))
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
                text = emoji.replace_emoji(text, replace="")
                print(colored(text, "green"), end="", flush=True)
                buf += text
                full_reply += text
                if any(p in buf[-1:] for p in "，,。.!！?？") or len(buf) >= 20:
                    await TTS_QUEUE.put(buf)
                    buf = ""

            if buf.strip():
                await TTS_QUEUE.put(buf)

            await mem_mgr.save_turn(query, full_reply)
            print(colored(f"\n(Total {(time.perf_counter()-first_token):.2f}s)", "blue"))

    except (EOFError, KeyboardInterrupt):
        print("\n[Quit]")
    finally:
        await TTS_QUEUE.put(None)
        await tts_task
        sound_stream.close()
        checkpoint_db()
        # os.kill(os.getpid(), signal.SIGKILL)
        # sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
