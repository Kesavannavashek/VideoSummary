import yt_dlp
from pydub import AudioSegment
import torch
from transformers import pipeline

def transcribe_audio_from_youtube(youtube_url, audio_filename="audio.mp3", wav_filename="processed_audio.wav"):

    ydl_opts = {
        "format": "bestaudio/best",
        "extractaudio": True,
        "audioformat": "mp3",
        "outtmpl": audio_filename,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

    audio = AudioSegment.from_file(audio_filename)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(wav_filename, format="wav")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    whisper_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-base", device=device)
    result = whisper_pipeline(wav_filename, return_timestamps=True)

    return result["text"]
