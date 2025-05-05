import os
import subprocess

import torch.cuda
import webrtcvad
import numpy as np
import wave
import contextlib
from pydub import AudioSegment
from faster_whisper import WhisperModel

# --- Step 1: Convert video to clean WAV (mono, 16kHz) ---
def convert_video_to_wav(video_path, audio_path):
    subprocess.run([
        "ffmpeg", "-i", video_path,
        "-ac", "1", "-ar", "16000",
        "-f", "wav", audio_path,
        "-y"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# --- Step 2: VAD - Voice Activity Detection ---
def get_voice_segments(wav_path, aggressiveness=2):
    vad = webrtcvad.Vad(aggressiveness)
    sample_rate = 16000
    frame_duration = 30  # ms
    frame_size = int(sample_rate * frame_duration / 1000) * 2  # bytes
    voice_segments = []

    with wave.open(wav_path, 'rb') as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        rate = wf.getframerate()
        num_frames = wf.getnframes()
        audio = wf.readframes(num_frames)

    for i in range(0, len(audio) - frame_size, frame_size):
        frame = audio[i:i + frame_size]
        if vad.is_speech(frame, sample_rate):
            start_sec = i / (sample_rate * 2)
            end_sec = (i + frame_size) / (sample_rate * 2)
            voice_segments.append((start_sec, end_sec))

    return merge_segments(voice_segments)

def merge_segments(segments, threshold=0.5):
    if not segments:
        return []

    merged = [segments[0]]
    for start, end in segments[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= threshold:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged

# --- Step 3: Transcribe with Faster-Whisper ---
def transcribe_segments(model, wav_path, segments):
    audio = AudioSegment.from_wav(wav_path)
    results = []

    for start, end in segments:
        chunk = audio[start * 1000:end * 1000]
        chunk.export("temp_chunk.wav", format="wav")
        print("started transcription...")
        segments, _ = model.transcribe("temp_chunk.wav", beam_size=1,language="en")
        text = "".join([seg.text for seg in segments])
        print(text)
        results.append((start, end, text.strip()))

    os.remove("temp_chunk.wav")
    return results

# --- Step 4: Run all ---
def transcribe_video(video_path):
    audio_path = "temp_audio.wav"
    convert_video_to_wav(video_path, audio_path)

    voice_segments = get_voice_segments(audio_path)
    print(f"[INFO] Found {len(voice_segments)} voice segments.")
    print(voice_segments)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    # Use int8 for speed, or float16 for accuracy/speed balance
    model = WhisperModel("small", device=device, compute_type="float16")
    print("model loaded succesfully...")
    transcript = transcribe_segments(model, audio_path, voice_segments)

    os.remove(audio_path)
    return transcript

# --- Example use ---
if __name__ == "__main__":
    result = transcribe_video("input.mp4")
    for start, end, text in result:
        print(f"[{start:.2f}-{end:.2f}] {text}")

# import torch
# print(torch.backends.cudnn.version())
# print(torch.backends.cudnn.enabled)
