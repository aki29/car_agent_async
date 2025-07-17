# audio/io.py  ← 全新
import asyncio, pathlib, sounddevice as sd, wave

async def mic_stream(sr: int = 16000, block: int = 1024):
    """非同步麥克風輸入，yield bytes。"""
    q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=20)

    def _cb(indata, frames, t, status):  # pylint: disable=unused-argument
        q.put_nowait(indata.tobytes())

    with sd.RawInputStream(samplerate=sr, channels=1, dtype="int16",
                           blocksize=block, callback=_cb):
        while True:
            yield await q.get()

async def play_wav(path: pathlib.Path):
    data = wave.open(str(path), "rb").readframes(-1)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, sd.play, data)

