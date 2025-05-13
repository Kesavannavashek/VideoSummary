import cv2
import string
import re
from textblob import TextBlob
from src.VideoProcessing.ocr_text_extraction import extract_text_from_frame  # Make sure this function returns a string or ""

def clean_ocr_text_with_format(ocr_text, apply_spellcheck=False):
    if not ocr_text:
        return ""

    cleaned_lines = []

    for line in ocr_text.splitlines():
        match = re.match(r"^(\s*)(.*)", line)
        indent, content = match.groups()

        content = ''.join(c for c in content if c in string.printable)

        replacements = {
            '0': 'O',
            '1': 'I',
            'ﬁ': 'fi',
            'ﬂ': 'fl'
        }
        for wrong, right in replacements.items():
            content = content.replace(wrong, right)

        if sum(c.isalpha() for c in content) < 3:
            continue

        if apply_spellcheck:
            content = str(TextBlob(content).correct())

        cleaned_lines.append(f"{indent}{content}")

    return '\n'.join(cleaned_lines)

def get_timestamp_millis(frame_index, fps):
    return int((frame_index / fps) * 1000)

def compute_hist_diff(prev_gray, curr_gray):
    prev_hist = cv2.calcHist([prev_gray], [0], None, [256], [0, 256])
    curr_hist = cv2.calcHist([curr_gray], [0], None, [256], [0, 256])
    return cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_BHATTACHARYYA)

def stream_and_collect_frames(video_path, hist_thresh=0.3, frame_skip=3):
    print("stream started...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_index = 0
    prev_gray = None
    extracted_frames = []

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
    print("Frames Extracted...")
    return run_ocr_on_frames(extracted_frames)

def run_ocr_on_frames(frames_with_timestamps):
    results = []
    for timestamp_ms, frame in frames_with_timestamps:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            ocr_result = extract_text_from_frame(img_rgb)
        except Exception as e:
            print(f"OCR failed at {timestamp_ms} ms: {e}")
            ocr_result = ""

        cleaned_text = clean_ocr_text_with_format(ocr_result)
        results.append((timestamp_ms, cleaned_text))
    return results


