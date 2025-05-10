import os
import subprocess
import wave
import torch
import yt_dlp
from pydub import AudioSegment
from faster_whisper import WhisperModel
import webrtcvad


def download_youtube_audio(youtube_url, audio_filename="audio.mp3", wav_filename="audio.wav"):

    ydl_opts = {
        "format": "bestaudio/best",
        "extractaudio": True,
        "audioformat": "mp3",
        "outtmpl": audio_filename,
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

    # Step 2: Convert to mono 16kHz WAV
    audio = AudioSegment.from_file(audio_filename)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(wav_filename, format="wav")
    os.remove(audio_filename)
    return wav_filename


def convert_video_to_wav(video_path, audio_path="temp_audio.wav"):
    subprocess.run([
        "ffmpeg", "-i", video_path,
        "-ac", "1", "-ar", "16000", "-f", "wav", audio_path, "-y"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return audio_path


def get_voice_segments(wav_path, aggressiveness=2):
    vad = webrtcvad.Vad(aggressiveness)
    sample_rate = 16000
    frame_duration = 30  # ms
    frame_size = int(sample_rate * frame_duration / 1000) * 2  # bytes

    with wave.open(wav_path, 'rb') as wf:
        audio = wf.readframes(wf.getnframes())

    segments = []
    for i in range(0, len(audio) - frame_size, frame_size):
        frame = audio[i:i + frame_size]
        if vad.is_speech(frame, sample_rate):
            start = i / (sample_rate * 2)
            end = (i + frame_size) / (sample_rate * 2)
            segments.append((start, end))

    return merge_segments(segments)


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


def transcribe_segments(model, wav_path, segments):
    audio = AudioSegment.from_wav(wav_path)
    results = []
    for start, end in segments:
        chunk = audio[start * 1000:end * 1000]
        chunk.export("temp_chunk.wav", format="wav")
        print(f"🟡 Transcribing {start:.2f}s - {end:.2f}s")
        segs, _ = model.transcribe("temp_chunk.wav", beam_size=1, language="en")
        text = "".join([s.text for s in segs])
        # print("text: ",text)
        results.append((start, end, text.strip()))
    os.remove("temp_chunk.wav")
    return results


def transcribe_youtube_video(youtube_url):
    wav_path = download_youtube_audio(youtube_url)
    return transcribe_from_wav(wav_path)


def transcribe_from_wav(wav_path):
    print("🔍 Detecting voice segments...")
    voice_segments = get_voice_segments(wav_path)
    print(f"🧠 Found {len(voice_segments)} voice segments.")
    print(voice_segments)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WhisperModel("small", device=device, compute_type="float16" if device == "cuda" else "int8")
    print("✅ Whisper model loaded.")

    transcript = transcribe_segments(model, wav_path, voice_segments)

    os.remove(wav_path)
    return transcript


def transcribe_local_video(video_path):
    wav_path = convert_video_to_wav(video_path)
    return transcribe_from_wav(wav_path)
