from src.AudioSubtitleProcessing.extract_speech import transcribe_local_video
from src.VideoProcessing.extract_frames_local_videos import extract_significant_ocr_frames

def local_video_pipeline(video_path):
    print("started...")
    subtitle = transcribe_local_video(video_path)
    print("subtitle extracted...")
    print(subtitle)
    ocr_data = extract_significant_ocr_frames(video_path)
    print(ocr_data)

if __name__ == "__main__":
    video_path = "input.mp4"
    local_video_pipeline(video_path)
