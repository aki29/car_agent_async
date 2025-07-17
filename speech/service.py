# src/car_agent_async/speech/service.py
from __future__ import annotations
import asyncio, hashlib, wave, aiofiles, pathlib
from cachetools import LRUCache, cached
from diskcache import Cache
import riva.client

AUDIO_TMP = pathlib.Path(".cache/audio_cache")
AUDIO_TMP.mkdir(exist_ok=True, parents=True)
MEM_CACHE = LRUCache(maxsize=128)          # 熱門語句常駐
DISK_CACHE = Cache(AUDIO_TMP)              # 長期快取

def _key(text: str, voice: str) -> str:
    return hashlib.sha1(f"{voice}:{text}".encode()).hexdigest()

class SpeechService:
    def __init__(self, host="localhost:50051"):
        self.channel = riva.client.grpc.aio.insecure_channel(host)
        self.asr = riva.client.ASRServiceStub(self.channel)
        self.tts = riva.client.TTSServiceStub(self.channel)

    # ---------- TTS ----------
    async def synth(self, text: str, voice="fastpitch_hifigan") -> pathlib.Path:
        key = _key(text, voice)
        if key in MEM_CACHE:
            return MEM_CACHE[key]
        if key in DISK_CACHE:
            MEM_CACHE[key] = DISK_CACHE[key]
            return DISK_CACHE[key]

        req = riva.client.SynthesizeSpeechRequest(
            text=text,
            language_code="zh-TW",
            encoding=riva.client.AudioEncoding.LINEAR_PCM,
            sample_rate_hz=48000,
            voice_name=voice,
            streaming=True,
        )
        path = AUDIO_TMP / f"{key}.wav"
        async with aiofiles.open(path, "wb") as f:
            async for resp in self.tts.SynthesizeOnline(req):
                await f.write(resp.audio)
        MEM_CACHE[key] = DISK_CACHE[key] = path
        return path

    # ---------- ASR ----------
    async def transcribe(self, pcm_async_iter, lang="zh"):
        conf = riva.client.StreamingRecognitionConfig(
            language_code=lang,
            sample_rate_hz=16000,
            enable_endpointer=True,
            max_alternatives=1,
        )
        async def req_gen():
            yield riva.client.StreamingRecognizeRequest(config=conf)
            async for chunk in pcm_async_iter:
                yield riva.client.StreamingRecognizeRequest(audio_content=chunk)

        async for resp in self.asr.StreamingRecognize(req_gen()):
            if resp.results and resp.results[0].is_final:
                yield resp.results[0].alternatives[0].transcript

