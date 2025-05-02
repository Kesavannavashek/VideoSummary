import ffmpeg
import numpy as np
import cv2
from src.VideoProcessing.ocr_text_extraction import extract_text_from_frame


def histogram_diff(img1, img2):

    hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)

def extract_frames_from_video(video_url, frame_rate=1):

    probe = ffmpeg.probe(video_url)
    video_info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])

    fps = float(video_info['avg_frame_rate'].split('/')[0]) / float(video_info['avg_frame_rate'].split('/')[1])

    process = (
        ffmpeg
        .input(video_url)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24', video_bitrate=8000)
        .run_async(pipe_stdout=True)
    )

    frame_data = []
    frame_number = 0
    prev_frame = None

    while True:

        in_bytes = process.stdout.read(width * height * 3)
        if not in_bytes:
            break


        frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])

        if frame_number % frame_rate == 0:
            timestamp = (frame_number / fps) * 1000

            if prev_frame is not None:
                diff = histogram_diff(prev_frame, frame)
                if diff < 0.1:
                    frame_number += 1
                    continue


            text = extract_text_from_frame(frame)
            if text:
                # print(text)
                frame_data.append((timestamp, text))
            # cv2.imshow("hello",frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            prev_frame = frame.copy()

        frame_number += 1

    process.wait()
    cv2.destroyAllWindows()
    return frame_data

