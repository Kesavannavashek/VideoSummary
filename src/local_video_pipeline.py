from concurrent.futures import ThreadPoolExecutor
from src.AudioSubtitleProcessing.extract_speech import transcribe_local_video
from src.AudioSubtitleProcessing.chunk_text import split_text_spacy, batch_chunks_duration_range
from src.VideoProcessing.extract_frames_local_videos import stream_and_collect_frames
from src.VideoProcessing.ocr_text_extraction import match_subs_with_ocr
from src.SummaryGeneration.generate_summary import summarize_matched_data
import time

def parallel_transcription_ocr(video_path):
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            subtitle_future = executor.submit(transcribe_local_video, video_path)
            ocr_future = executor.submit(stream_and_collect_frames, video_path)
            subtitle = subtitle_future.result()
            ocr_data = ocr_future.result()
        return subtitle, ocr_data
    except Exception as e:
        print(f"Error in parallel processing: {e}")
        raise

def local_video_pipeline(video_path):
    s = time.time()
    print("Started...")

    try:

        subtitle, ocr_data = parallel_transcription_ocr(video_path)
        print("Total time for ASR and OCR:", time.time() - s)

        print("Subtitle extracted...")
        print("OCR extracted...")

        # Process subtitle text
        chunked_text = split_text_spacy(subtitle)
        print("Subtitle chunked:", chunked_text)

        timed_chunk = batch_chunks_duration_range(chunked_text)
        print("Subtitle timestamped...")

        print("OCR data:", ocr_data)

        # Match subtitles with OCR data
        matched_subtitles = match_subs_with_ocr(timed_chunk, ocr_data)
        print("Matched subtitles:", matched_subtitles)

        # Generate summary
        summary = summarize_matched_data(matched_subtitles, context="local")
        print("Summary generated:", summary)
        print("Total time:", time.time() - s)


        return summary

    except Exception as e:
        print(f"Pipeline error: {e}")
        return None

if __name__ == "__main__":
    video_path = "input.mp4"
    local_video_pipeline(video_path)