import collections
import queue
import threading
import time

import pyaudio
import webrtcvad
import sys, os
import riva.client
import numpy as np
import audioop


class VADSource:
    def __init__(
        self,
        rate=16000,
        frame_duration_ms=10,
        padding_duration_ms=300,
        device=None,
        vad_mode=2,
        start_ratio=0.8,
        end_ratio=0.65,
        apm=None,
    ):

        self.rate = rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(rate * frame_duration_ms / 1000)  # samples
        self.padding_duration_ms = padding_duration_ms
        self.padding_frames = int(padding_duration_ms / frame_duration_ms)
        self.device = device
        self.apm = apm
        self.vad = webrtcvad.Vad(vad_mode)  # mode: 0–3，數字越大越嚴格
        self._buff = queue.Queue()
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio

        self._audio_interface = pyaudio.PyAudio()
        self._stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=rate,
            input=True,
            frames_per_buffer=self.frame_size,
            input_device_index=device,
        )
        t = threading.Thread(target=self._fill_buffer, daemon=True)
        t.start()

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
            self._buff.put((clean, self.vad.is_speech(clean, self.rate)))

    def __iter__(self):
        ring = collections.deque(maxlen=self.padding_frames)
        triggered = False
        silence_frame = bytes(self.frame_size * 2)  # int16 × frame_size，全 0
        while True:
            frame, is_speech = self._buff.get()
            # print("#" if is_speech else "@", end="", flush=True)
            if not triggered:
                ring.append(is_speech)
                if ring.count(True) > self.start_ratio * ring.maxlen:
                    triggered = True
                    yield silence_frame
                    ring.clear()  #
            else:
                yield frame
                ring.append(is_speech)
                if ring.count(False) > self.end_ratio * ring.maxlen:
                    break
        return
