from __future__ import annotations
import asyncio, hashlib, pathlib, aiofiles, wave, grpc
from cachetools import LRUCache
from diskcache import Cache
from riva.client.proto import (
    riva_audio_pb2,
    riva_asr_pb2_grpc,
    riva_tts_pb2_grpc,
    riva_asr_pb2,
    riva_tts_pb2,
)
from audio.vad import VADSource
from termcolor import colored


_AUD_DIR = pathlib.Path(".cache/audio_cache")
_AUD_DIR.mkdir(parents=True, exist_ok=True)
_RAM = LRUCache(maxsize=128)
_DISK = Cache(_AUD_DIR, size_limit=512 * 1024 * 1024)


def _key(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()


class SpeechService:
    def __init__(self, host: str = "localhost:50051"):
        self._chan: grpc.aio.Channel = grpc.aio.insecure_channel(host)
        self._asr = riva_asr_pb2_grpc.RivaSpeechRecognitionStub(self._chan)
        self._tts = riva_tts_pb2_grpc.RivaSpeechSynthesisStub(self._chan)
        self._vad = None

    # ---------- TTS ----------
    async def synth(self, text: str, *, voice: str, sr: int = 48000) -> pathlib.Path:
        k = _key(f"{sr}:{voice}:{text}")

        if (p := _RAM.get(k)) or (p := _DISK.get(k)):
            try:
                with wave.open(str(p), "rb") as _:
                    return p
            except wave.Error:
                _RAM.pop(k, None)
                _DISK.pop(k, None)

        req = riva_tts_pb2.SynthesizeSpeechRequest(
            text=text,
            language_code="zh-CN",
            voice_name=voice,
            sample_rate_hz=sr,
            encoding=riva_audio_pb2.AudioEncoding.LINEAR_PCM,
            # streaming=True,
        )

        out_path = _AUD_DIR / f"{k}.wav"
        pcm = bytearray()
        try:
            async for rsp in self._tts.SynthesizeOnline(req):
                pcm.extend(rsp.audio)
            # rsp = await self._tts.Synthesize(req)
            # pcm.extend(rsp.audio)
        except grpc.aio.AioRpcError as e:
            raise
        with wave.open(str(out_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm)
        _RAM[k] = out_path
        _DISK.set(k, out_path, expire=30 * 24 * 3600)
        return out_path

    async def _stream_recognize_once(
        self, *, lang: str, sr: int, device: int | None, vad_mode: int, padding_ms: int
    ) -> str:

        if self._vad is None:
            self._vad = VADSource(
                rate=sr,
                frame_duration_ms=20,
                padding_duration_ms=padding_ms,
                device=device,
                vad_mode=vad_mode,
                start_ratio=0.5,
                end_ratio=0.8,
            )
        self._vad.reset()

        stream_cfg = riva_asr_pb2.StreamingRecognitionConfig(
            config=riva_asr_pb2.RecognitionConfig(
                language_code=lang,
                sample_rate_hertz=sr,
                encoding=riva_audio_pb2.AudioEncoding.LINEAR_PCM,
                audio_channel_count=1,
                enable_automatic_punctuation=False,
                max_alternatives=1,
            ),
            interim_results=True,
            # single_utterance=False,
        )

        async def req_gen():
            yield riva_asr_pb2.StreamingRecognizeRequest(streaming_config=stream_cfg)
            for frame in self._vad:
                yield riva_asr_pb2.StreamingRecognizeRequest(audio_content=frame)
            yield riva_asr_pb2.StreamingRecognizeRequest(audio_content=b"")

        final_text: list[str] = []
        async for rsp in self._asr.StreamingRecognize(req_gen()):
            for res in rsp.results:
                if res.is_final and res.alternatives:
                    final_text.append(res.alternatives[0].transcript)

        return " ".join(final_text).strip()

    async def listen_and_transcribe(
        self,
        *,
        lang: str = "zh-CN",
        sr: int = 16000,
        device: int | None = None,
        vad_mode: int = 2,
        padding_ms: int = 500,
    ) -> str:
        while True:
            txt = await self._stream_recognize_once(
                lang=lang, sr=sr, device=device, vad_mode=vad_mode, padding_ms=padding_ms
            )
            print("R", txt)
            if txt:
                return txt
