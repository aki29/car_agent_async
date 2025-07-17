# speech/service.py  ← 全新
from __future__ import annotations
import asyncio, hashlib, pathlib, aiofiles
from cachetools import LRUCache
from diskcache import Cache
import riva.client
from riva.client.proto import riva_audio_pb2

_AUD_DIR = pathlib.Path("data/audio_cache")
_AUD_DIR.mkdir(parents=True, exist_ok=True)
_RAM = LRUCache(maxsize=128)
_DISK = Cache(_AUD_DIR)

def _key(text: str, voice: str) -> str:
    return hashlib.sha1(f"{voice}:{text}".encode()).hexdigest()

class SpeechService:
    def __init__(self, host: str = "localhost:50051"):
        self._chan = riva.client.grpc.aio.insecure_channel(host)
        self._asr = riva.client.ASRServiceStub(self._chan)
        self._tts = riva.client.TTSServiceStub(self._chan)

    # ---------- TTS ----------
    async def synth(self, text: str, *, voice: str, sr: int = 48000) -> pathlib.Path:
        k = _key(text, voice)
        if k in _RAM:
            return _RAM[k]
        if k in _DISK:
            _RAM[k] = _DISK[k]
            return _DISK[k]

        req = riva.client.SynthesizeSpeechRequest(
            text=text,
            language_code="zh-TW",
            voice_name=voice,
            sample_rate_hz=sr,
            encoding=riva_audio_pb2.AudioEncoding.LINEAR_PCM,
            streaming=True,
        )
        out = _AUD_DIR / f"{k}.wav"
        async with aiofiles.open(out, "wb") as f:
            async for rsp in self._tts.SynthesizeOnline(req):
                await f.write(rsp.audio)
        _RAM[k] = _DISK[k] = out
        return out

    # ---------- ASR ----------
    async def transcribe(self, audio_iter, *, lang: str = "zh") -> asyncio.AsyncIterator[str]:
        cfg = riva.client.StreamingRecognitionConfig(
            language_code=lang,
            sample_rate_hz=16000,
            enable_endpointer=True,
            max_alternatives=1,
        )

        async def _req():
            yield riva.client.StreamingRecognizeRequest(config=cfg)
            async for chunk in audio_iter:
                yield riva.client.StreamingRecognizeRequest(audio_content=chunk)

        async for resp in self._asr.StreamingRecognize(_req()):
            if resp.results and resp.results[0].is_final:
                yield resp.results[0].alternatives[0].transcript

