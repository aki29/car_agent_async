from __future__ import annotations
import asyncio, hashlib, pathlib, aiofiles, wave
from cachetools import LRUCache
from diskcache import Cache

# import riva.client
import grpc
from riva.client.proto import (
    riva_audio_pb2,
    riva_asr_pb2_grpc,
    riva_tts_pb2_grpc,
    riva_asr_pb2,
    riva_tts_pb2,
)
from audio.vad import VADSource


async def listen_once(sr: int = 16000) -> bytes:
    vad = VADSource(
        rate=sr,
        frame_duration_ms=30,  # 30 ms 讓判斷更穩定
        padding_duration_ms=300,
        vad_mode=2,  # 0~3 越大越嚴格
        start_ratio=0.7,
        end_ratio=0.5,
        apm=None,
    )
    # VADSource 是同步 iterator，要放進 executor
    loop = asyncio.get_running_loop()
    frames = await loop.run_in_executor(None, lambda: b"".join(vad))
    return frames


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
            # audio_sample_rate_hz=sr,
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

    # async def transcribe(
    #     self,
    #     audio_iter,
    #     *,
    #     lang: str = "zh-CN",
    #     sr: int = 16000,
    # ) -> asyncio.AsyncIterator[str]:
    #     rec_cfg = riva_asr_pb2.RecognitionConfig(
    #         language_code=lang,
    #         sample_rate_hertz=sr,
    #         encoding=riva_audio_pb2.AudioEncoding.LINEAR_PCM,
    #         max_alternatives=1,
    #         audio_channel_count=1,
    #         enable_automatic_punctuation=True,
    #     )

    #     stream_cfg = riva_asr_pb2.StreamingRecognitionConfig(
    #         config=rec_cfg,
    #         interim_results=True,  # 要即時 partial 就改 True
    #     )

    #     async def _req_gen():
    #         # 第一包必須是 streaming_config
    #         yield riva_asr_pb2.StreamingRecognizeRequest(streaming_config=stream_cfg)
    #         async for chunk in audio_iter:
    #             yield riva_asr_pb2.StreamingRecognizeRequest(audio_content=chunk)

    #     async for resp in self._asr.StreamingRecognize(_req_gen()):
    #         if resp.results and resp.results[0].is_final:
    #             yield resp.results[0].alternatives[0].transcript

    async def transcribe_bytes(self, pcm: bytes, *, lang="zh-CN", sr=16000) -> str:
        # 將整段 PCM 封裝成 riva_asr_pb2.RecognizeRequest (非串流)
        req = riva_asr_pb2.RecognizeRequest(
            config=riva_asr_pb2.RecognitionConfig(
                language_code=lang,
                sample_rate_hertz=sr,
                encoding=riva_audio_pb2.AudioEncoding.LINEAR_PCM,
                audio_channel_count=1,
                enable_automatic_punctuation=True,
            ),
            audio_content=pcm,
        )
        resp = await self._asr.Recognize(req)
        if resp.results and resp.results[0].alternatives:
            return resp.results[0].alternatives[0].transcript.strip()
        return ""
