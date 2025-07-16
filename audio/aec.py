import numpy as np
from collections import deque
from webrtc_audio_processing import AudioProcessingModule as APM     

# pip install "livekit-rtc[audio]"
# from livekit.rtc.apm import AudioProcessingModule as APM  # echo_cancellation 有暴露
# self.apm = APM(echo_cancellation=True, noise_suppression=True, auto_gain_control=False)

class WebRTCAEC:
    """Process <near_in, far_ref> -> echo‑free near_out."""
    def __init__(self, sample_rate=16000, frame_ms=10, channels=1):
        self.frame = sample_rate * frame_ms // 1000
        # self.apm = APM(enable_aec=True, enable_ns=True, enable_agc=False)
        self.apm = APM(enable_ns=True, enable_agc=False)
        self.apm.set_stream_format(sample_rate, channels)

        # 喇叭參考訊號的環形緩衝（2 s）
        self._far_ring = deque(maxlen=int(sample_rate * 2 / self.frame))

    def add_far(self, pcm16: bytes) -> None:
        self._far_ring.append(np.frombuffer(pcm16, dtype=np.int16))

    def process(self, pcm16: bytes) -> bytes:
        far = self._far_ring[-1] if self._far_ring else None
        near = np.frombuffer(pcm16, dtype=np.int16)
        clean = self.apm.process_stream(near, echo_reference=far)
        return clean.astype(np.int16).tobytes()
