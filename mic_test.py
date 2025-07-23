#!/usr/bin/env python3
"""
mic_test.py  ── 測試 USB/Pulse 麥克風是否能以 16 kHz 正常錄音，
並觀察 WebRTC VAD 判斷結果。

依需求修改：
    DEVICE_INDEX = 26   # PulseAudio 虛擬裝置
    DURATION_SEC  = 5
"""

import pyaudio, wave, webrtcvad, sys, time

DEVICE_INDEX = 26        # 改成 None=預設，或 USB 裝置的 index
#DEVICE_INDEX = None        # 改成 None=預設，或 USB 裝置的 ind
RATE         = 16000
FRAME_MS     = 20        # 10 / 20 / 30 皆可
SAMPLES      = int(RATE * FRAME_MS / 1000)
DURATION_SEC = 5
VAD_MODE     = 3         # 0~3，越大越嚴格

vad    = webrtcvad.Vad(VAD_MODE)
pa     = pyaudio.PyAudio()

try:
    stream = pa.open(format=pyaudio.paInt16,
                     channels=1,
                     rate=RATE,
                     input=True,
                     frames_per_buffer=SAMPLES,
                     input_device_index=DEVICE_INDEX)
except OSError as e:
    sys.exit(f"[Err] 開啟錄音裝置失敗：{e}")

#from audio.webrtc_frontend import WebRTCAudioFrontend
#apm = WebRTCAudioFrontend(rate=16000, channels=1, ns=True, agc=False)

print(f"🟢 開始錄音 {DURATION_SEC}s… 說話看看，silence=·, speech=#")
frames, speech_cnt = [], 0
start = time.time()

while time.time() - start < DURATION_SEC:
    chunk = stream.read(SAMPLES, exception_on_overflow=False)
    #clean = apm.process_mic(chunk)
    is_speech = vad.is_speech(chunk, RATE)
    print("#" if is_speech else ".", end="", flush=True)
    if is_speech:
        speech_cnt += 1
    frames.append(chunk)

stream.stop_stream()
stream.close()
pa.terminate()
print("\n🔚 錄音結束")

# 儲存 WAV 以便後續檢聽
wav_name = "test_mic.wav"
with wave.open(wav_name, "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(RATE)
    wf.writeframes(b"".join(frames))

total_frames = len(frames)
print(f"📜 總 frame：{total_frames}（{total_frames*FRAME_MS} ms）")
print(f"🗣️  語音 frame：{speech_cnt}（{speech_cnt*FRAME_MS} ms）")
print(f"💾 音檔已存：{wav_name}")

