# audio/io.py
import asyncio, sounddevice as sd

async def mic_stream(sr=16_000, block=1024):
    q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=20)

    def _callback(indata, frames, time, status):
        q.put_nowait(indata.tobytes())

    with sd.RawInputStream(samplerate=sr, channels=1, dtype="int16",
                           blocksize=block, callback=_callback):
        while True:
            yield await q.get()

async def play_wav_async(path: pathlib.Path):
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, sd.play, wave.open(path, "rb").readframes(-1))

