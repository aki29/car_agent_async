import asyncio, pathlib, wave, sounddevice as sd, numpy as np

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
    if _PLAY_Q is None:
        _PLAY_Q = asyncio.Queue()
        _PLAYER_TASK = loop.create_task(_player_worker())


async def _player_worker():
    while True:
        samples, sr = await _PLAY_Q.get()
        sd.play(samples, sr)
        sd.wait()  # 播完才取下一段
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

    await _PLAY_Q.put((data, sr))  # 丟進佇列即可，立即 return
