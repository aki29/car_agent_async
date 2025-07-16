# audio/webrtc_frontend.py
from webrtc_audio_processing import AudioProcessingModule as APM
import audioop


def slice_pcm(pcm: bytes, rate: int, channels: int = 1):
    frame = int(rate * 0.01) * 2 * channels  # 10 ms × 16‑bit
    for i in range(0, len(pcm), frame):
        yield pcm[i : i + frame]


def resample(pcm: bytes, rate_in: int, rate_out: int, ch: int = 1) -> bytes:
    if rate_in == rate_out:
        return pcm
    return audioop.ratecv(pcm, 2, ch, rate_in, rate_out, None)[0]


class WebRTCAudioFrontend:
    """AEC / NS / AGC / VAD 一站式前處理."""

    def __init__(
        self,
        rate: int = 16000,
        channels: int = 1,
        aec: int = 2,
        ns: bool = True,
        agc: int = 1,
        vad: bool = True,
    ):
        self.rate = rate
        self.channels = channels
        self.apm = APM(aec, ns, agc, vad)
        self.apm.set_stream_format(rate, channels, rate, channels)
        self.apm.set_reverse_stream_format(rate, channels)
        self.frame_bytes = int(rate * 0.01) * 2 * channels

    # ── far‑end ───────────────────────────────────────────────
    def feed_far_end(self, pcm_any_rate: bytes, src_rate: int):
        """把即將播出的喇叭聲（TTS）先送進 AEC."""
        for seg in slice_pcm(resample(pcm_any_rate, src_rate, self.rate), self.rate):
            self.apm.process_reverse_stream(seg)

    # ── near‑end ──────────────────────────────────────────────
    def process_mic(self, pcm10ms: bytes):
        clean = self.apm.process_stream(pcm10ms)
        return clean
