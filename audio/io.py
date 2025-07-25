import asyncio, pathlib, wave, sounddevice as sd, numpy as np
from speech.service import SpeechService

Q_SIZE = 100


async def mic_stream(sr: int = 16_000, block_ms: int = 20):
    q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=Q_SIZE)
    block = int(sr * (block_ms / 1000.0))

    def _cb(indata, frames, t, status):
        try:
            q.put_nowait(bytes(indata))
        except asyncio.QueueFull:
            return

    with sd.RawInputStream(
        samplerate=sr,
        channels=1,
        dtype="int16",
        blocksize=block,
        callback=_cb,
    ):
        while True:
            yield await q.get()


_PLAY_Q: asyncio.Queue[tuple[np.ndarray, int]] | None = None
_PLAYER_TASK: asyncio.Task | None = None


def _ensure_player(loop: asyncio.AbstractEventLoop):
    global _PLAY_Q, _PLAYER_TASK
    if _PLAY_Q is None or _PLAYER_TASK is None or _PLAYER_TASK.done():
        _PLAY_Q = asyncio.Queue()
        _PLAYER_TASK = loop.create_task(_player_worker())


# def stop_playing():
#     setattr(_player_worker, "interrupted", True)
#     sd.stop()
#     setattr(_player_worker, "interrupted", False)


def stop_playing():
    if _PLAYER_TASK and not _PLAYER_TASK.done():
        sd.stop()
        try:
            _PLAY_Q.put_nowait(None)
        except asyncio.QueueFull:
            _PLAY_Q = asyncio.Queue()  


# async def _player_worker():
#     while True:
#         samples, sr = await _PLAY_Q.get()
#         if getattr(_player_worker, "interrupted", False):
#             _PLAY_Q.task_done()
#             continue

#         sd.play(samples, sr)
#         try:
#             sd.wait()
#         except Exception:
#             pass
#         _PLAY_Q.task_done()


async def _player_worker():
    while True:
        item = await _PLAY_Q.get()
        if item is None:  # 偵測結束訊號
            _PLAY_Q.task_done()
            break

        samples, sr = item
        try:
            sd.play(samples, sr)
            sd.wait()
        except Exception:
            pass
        finally:
            _PLAY_Q.task_done()


async def play_wav(path: pathlib.Path):
    loop = asyncio.get_running_loop()
    _ensure_player(loop)

    with wave.open(str(path), "rb") as wf:
        data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        sr = wf.getframerate()
        ch = wf.getnchannels()
        if ch > 1:
            data = data.reshape(-1, ch)

    await _PLAY_Q.put((data, sr))


async def synth_and_enqueue(text: str, speech: SpeechService, voice: str, sr: int = 22050):
    try:
        wav = await speech.synth(text.strip(), voice=voice, sr=sr)
        await play_wav(wav)
    except Exception as e:
        print("[TTS synth failed]:", e)


async def detect_barge_in(speech: SpeechService):
    async for chunk in speech.transcribe(mic_stream()):
        if len(chunk.strip()) > 2:
            print("[BARGE-IN] User started talking.", "yellow")
            stop_playing()  # 停止播放中 TTS 音訊
            return chunk.strip()
