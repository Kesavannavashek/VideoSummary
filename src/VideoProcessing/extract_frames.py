import ffmpeg
import numpy as np
import cv2
from src.VideoProcessing.ocr_text_extraction import extract_text_from_frame


def histogram_diff(img1, img2):
    """Calculate histogram difference between two frames."""
    hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)

def extract_frames_from_video(video_url, frame_rate=1):


    # Probe the video to get the resolution and FPS
    probe = ffmpeg.probe(video_url)
    video_info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])

    # Calculate FPS from avg_frame_rate
    fps = float(video_info['avg_frame_rate'].split('/')[0]) / float(video_info['avg_frame_rate'].split('/')[1])

    # Start FFmpeg process to read the video in raw RGB format with high quality pixel format and bitrate
    process = (
        ffmpeg
        .input(video_url)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24', video_bitrate=8000)  # RGB format and high bitrate
        .run_async(pipe_stdout=True)
    )

    frame_data = []
    frame_number = 0
    prev_frame = None

    while True:
        # Read the raw bytes from FFmpeg output
        in_bytes = process.stdout.read(width * height * 3)  # RGB24 format (3 bytes per pixel)
        if not in_bytes:
            break

        # Convert the bytes to a numpy array representing the frame
        frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])

        if frame_number % frame_rate == 0:
            timestamp = (frame_number / fps) * 1000

            # Check if frame is significantly different from the previous one
            if prev_frame is not None:
                diff = histogram_diff(prev_frame, frame)
                if diff < 0.1:
                    frame_number += 1
                    continue

            # Extract text using OCR
            text = extract_text_from_frame(frame)
            if text:
                print(text)
                frame_data.append((timestamp, text))

            # Wait a bit to see the frame (adjust delay if needed)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            prev_frame = frame.copy()

        frame_number += 1

    process.wait()
    cv2.destroyAllWindows()
    return frame_data

