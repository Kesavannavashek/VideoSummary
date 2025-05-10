# import os
# import subprocess
#
# import torch.cuda
# import webrtcvad
# import numpy as np
# import wave
# import contextlib
# from pydub import AudioSegment
# from faster_whisper import WhisperModel
#
# # --- Step 1: Convert video to clean WAV (mono, 16kHz) ---
# def convert_video_to_wav(video_path, audio_path):
#     subprocess.run([
#         "ffmpeg", "-i", video_path,
#         "-ac", "1", "-ar", "16000",
#         "-f", "wav", audio_path,
#         "-y"
#     ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#
# # --- Step 2: VAD - Voice Activity Detection ---
# def get_voice_segments(wav_path, aggressiveness=2):
#     vad = webrtcvad.Vad(aggressiveness)
#     sample_rate = 16000
#     frame_duration = 30  # ms
#     frame_size = int(sample_rate * frame_duration / 1000) * 2  # bytes
#     voice_segments = []
#
#     with wave.open(wav_path, 'rb') as wf:
#         n_channels = wf.getnchannels()
#         sample_width = wf.getsampwidth()
#         rate = wf.getframerate()
#         num_frames = wf.getnframes()
#         audio = wf.readframes(num_frames)
#
#     for i in range(0, len(audio) - frame_size, frame_size):
#         frame = audio[i:i + frame_size]
#         if vad.is_speech(frame, sample_rate):
#             start_sec = i / (sample_rate * 2)
#             end_sec = (i + frame_size) / (sample_rate * 2)
#             voice_segments.append((start_sec, end_sec))
#
#     return merge_segments(voice_segments)
#
# def merge_segments(segments, threshold=0.5):
#     if not segments:
#         return []
#
#     merged = [segments[0]]
#     for start, end in segments[1:]:
#         prev_start, prev_end = merged[-1]
#         if start - prev_end <= threshold:
#             merged[-1] = (prev_start, end)
#         else:
#             merged.append((start, end))
#     return merged
#
# # --- Step 3: Transcribe with Faster-Whisper ---
# def transcribe_segments(model, wav_path, segments):
#     audio = AudioSegment.from_wav(wav_path)
#     results = []
#
#     for start, end in segments:
#         chunk = audio[start * 1000:end * 1000]
#         chunk.export("temp_chunk.wav", format="wav")
#         print("started transcription...")
#         segments, _ = model.transcribe("temp_chunk.wav", beam_size=1,language="en")
#         text = "".join([seg.text for seg in segments])
#         print(text)
#         results.append((start, end, text.strip()))
#
#     os.remove("temp_chunk.wav")
#     return results
#
# # --- Step 4: Run all ---
# def transcribe_video(video_path):
#     audio_path = "temp_audio.wav"
#     convert_video_to_wav(video_path, audio_path)
#
#     voice_segments = get_voice_segments(audio_path)
#     print(f"[INFO] Found {len(voice_segments)} voice segments.")
#     print(voice_segments)
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(device)
#     # Use int8 for speed, or float16 for accuracy/speed balance
#     model = WhisperModel("small", device=device, compute_type="float16")
#     print("model loaded succesfully...")
#     transcript = transcribe_segments(model, audio_path, voice_segments)
#
#     os.remove(audio_path)
#     return transcript
#
# # --- Example use ---
# if __name__ == "__main__":
#     result = transcribe_video("input.mp4")
#     for start, end, text in result:
#         print(f"[{start:.2f}-{end:.2f}] {text}")
#
# # import torch
# # print(torch.backends.cudnn.version())
# # print(torch.backends.cudnn.enabled)
import time

# import cv2
# import time
#
# print("started...")
# cap = cv2.VideoCapture("input.mp4")
# start = time.time()
# frames = 0
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     frames += 1
#
# end = time.time()
# cap.release()
#
# print(f"Read {frames} frames in {end - start:.2f} seconds")
# print(f"Speed: {frames / (end - start):.2f} FPS")
import cv2
import numpy as np
import time

def get_timestamp_millis(frame_index, fps):
    return int((frame_index / fps) * 1000)

def compute_hist_diff(prev_gray, curr_gray):
    prev_hist = cv2.calcHist([prev_gray], [0], None, [256], [0, 256])
    curr_hist = cv2.calcHist([curr_gray], [0], None, [256], [0, 256])
    return cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_BHATTACHARYYA)

def stream_and_collect_frames(video_path, hist_thresh=0.3, frame_skip=3):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_index = 0
    prev_gray = None
    extracted_frames = []

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % frame_skip != 0:
            frame_index += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        save_this = False

        if prev_gray is None:
            save_this = True
        else:
            hist_diff = compute_hist_diff(prev_gray, gray)
            if hist_diff > hist_thresh:
                save_this = True

        if save_this:
            timestamp_ms = get_timestamp_millis(frame_index, fps)
            extracted_frames.append((timestamp_ms, frame.copy()))
            prev_gray = gray

        frame_index += 1

    cap.release()
    total_time = time.time() - start_time
    return extracted_frames, total_time


from concurrent.futures import ThreadPoolExecutor, as_completed
from paddleocr import PaddleOCR
import logging
logging.getLogger('ppocr').setLevel(logging.ERROR)
from src.VideoProcessing.ocr_text_extraction import extract_text_from_frame
# Initialize the OCR model (English)
# ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True)  # Set use_gpu=False if no GPU
ocr = PaddleOCR(
    det_model_dir='en_PP-OCRv4_det',
    rec_model_dir='en_PP-OCRv4_rec',
    use_angle_cls=False,  # turn off for speed if not needed
    lang='en',
    use_gpu=True,
    precision='fp16'
)

def run_ocr_on_frames(frames_with_timestamps):
    results = []
    for timestamp_ms, frame in frames_with_timestamps:
        # Convert OpenCV BGR image to RGB (as required by PaddleOCR)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ocr_result = extract_text_from_frame(img_rgb)
        results.append((timestamp_ms, ocr_result))
    return results


# Example usage
if __name__ == "__main__":
    video_path = "input.mp4"  # Replace with your video file
    print("started....")
    st = time.time()

    frames, duration = stream_and_collect_frames(video_path)
    print(f"Extracted {len(frames)} frames in {duration:.2f} seconds")
    print("ocr started...")
    ocr_data = run_ocr_on_frames(frames)
    print(ocr_data)
    e = time.time()
    print("total time: ",e-st)
