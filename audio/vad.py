import queue
import threading
import pyaudio
import webrtcvad
from collections import deque


class VADSource:
    _pya = pyaudio.PyAudio()
    _streams: dict[int, pyaudio.Stream] = {}

    def __init__(
        self,
        rate=16000,
        frame_duration_ms=20,
        padding_duration_ms=300,
        device=None,
        vad_mode=3,
        start_ratio=0.6,
        end_ratio=0.5,
        apm=None,
    ):

        self.rate = rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(rate * frame_duration_ms / 1000)  # samples
        self.padding_duration_ms = padding_duration_ms
        self.padding_frames = int(padding_duration_ms / frame_duration_ms)
        self.device = device or VADSource._pya.get_default_input_device_info()["index"]
        self.apm = apm
        self.vad = webrtcvad.Vad(vad_mode)  # mode: 0–3，數字越大越嚴格
        self._buff = queue.Queue()
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio

        self.start_ratio_base = start_ratio
        self.end_ratio_base = end_ratio
        self.min_start_ratio = 0.3
        self.max_end_ratio = 0.95
        self.adapt_decay = 0.01  # 每 frame 衰減幅度
        self.idle_frame_count = 0  # 無語音 frame 計數器

        if self.device in VADSource._streams:
            self._stream = VADSource._streams[self.device]
        else:
            self._stream = VADSource._pya.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.rate,
                input=True,
                # output=False,
                frames_per_buffer=self.frame_size,
                input_device_index=self.device,
                start=False,
            )
            self._stream.start_stream()
            VADSource._streams[self.device] = self._stream
        self.reset()
        t = threading.Thread(target=self._fill_buffer, daemon=True)
        t.start()

    def reset(self):
        self._ring = deque(maxlen=self.padding_frames)
        self._fifo = deque(maxlen=12)
        self._triggered = False

        self.start_ratio = self.start_ratio_base
        self.end_ratio = self.end_ratio_base
        self.idle_frame_count = 0

    def close(self):
        self._stream.stop_stream()
        self._stream.close()

    def _fill_buffer(self):
        while True:
            raw = self._stream.read(self.frame_size, exception_on_overflow=False)
            # rms = audioop.rms(raw, 2)
            # print("RMS_IN", rms)  # DEBUG：確認有音量
            clean = raw
            if self.apm:
                clean = self.apm.process_mic(raw)
                # rms = audioop.rms(clean, 2)
                # print("RMS_OUT", rms)  # DEBUG：確認有音量
            # print('@')
            self._buff.put((clean, self.vad.is_speech(clean, self.rate)))

    def __iter__(self):
        print('S',self.start_ratio * self._ring.maxlen)
        while True:
            frame, is_speech = self._buff.get()
            self._ring.append(is_speech)
            print("#" if is_speech else ".", end="", flush=True)
            if not self._triggered:
                self._fifo.append(frame)
                if self._ring.count(True) >= self.start_ratio * self._ring.maxlen:
                    print('T', self._ring.count(True))
                    self._triggered = True
                    for f in self._fifo:
                        yield f
                    self._fifo.clear()
                continue

            yield frame
            if self._ring.count(False) >= self.end_ratio * self._ring.maxlen:
                print('E', self._ring.count(False))
                break
        self.reset()
