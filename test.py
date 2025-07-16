from webrtc_audio_processing import AudioProcessingModule as AP

ap = AP(enable_ns=True)          # 只先開 Noise Suppression
ap.set_stream_format(16000, 1)   # 16 kHz，單聲道
print("Noise Suppressor initialised ✔")
