# from src.AudioSubtitleProcessing.chunk_text import chunk_text
# from src.AudioSubtitleProcessing.extact_subtitles import get_subtitle_text, get_timestamped_chunks
# from src.AudioSubtitleProcessing.extract_speech import transcribe_audio_from_youtube
# from src.GetVideoInfo.get_video_info import extract_video_info,get_video_direct_url
# from src.SummaryGeneration.generate_summary import summarize_matched_data
# from src.VideoProcessing.extract_frames import extract_frames_from_video
# from src.VideoProcessing.ocr_text_extraction import match_subs_with_ocr
#
#
# def pipeline(video_url):
#     # Step 1: Extract video information
#     info = extract_video_info(video_url)
#     # print("INFO: ",info)
#     direct_url = get_video_direct_url(info)
#     print("subtitle info are retrieved...")
#     if(direct_url == None):
#         print("No URL found....")
#         return
#     # Step 2: Try to get subtitles
#     text = get_subtitle_text(info)
#     if not text:
#         print("No subtitles found. Using Whisper to transcribe...")
#         text = transcribe_audio_from_youtube(video_url)
#
#     chunks = chunk_text(text)
#     timestamped_chunks = get_timestamped_chunks(info,chunks)
#     print("time stamped chunks are retrieved...")
#     print("starting ocr extraction...")
#     # Step 4: Extract frames and OCR data
#     ocr_data = extract_frames_from_video(direct_url)
#     print("ocr data extracted...")
#     print("syncing subtitles with ocr data...")
#     # Step 5: Match subtitles with OCR
#     matched_data = match_subs_with_ocr(timestamped_chunks, ocr_data)
#     print("successfully synced subtitles with ocr data...")
#     print("summarizing.....")
#     return summarize_matched_data(matched_data)
#
#
# if __name__ == "__main__":
#     # url = "https://youtu.be/4EP8YzcN0hQ?si=-lfdX62fpjkgvq8O"
#     # url = "https://youtu.be/ldYLYRNaucM?si=ximNENy042FR06sR"
#     url = "https://youtu.be/LFGBTFxHJII?si=O7wULtF5Dfz5lHzh"
#     matchedData = pipeline(url)
#     print("summary: ",matchedData)


import concurrent.futures
from src.AudioSubtitleProcessing.chunk_text import chunk_text
from src.AudioSubtitleProcessing.extact_subtitles import get_subtitle_text, get_timestamped_chunks
from src.AudioSubtitleProcessing.extract_speech import transcribe_audio_from_youtube
from src.GetVideoInfo.get_video_info import extract_video_info, get_video_direct_url
from src.SummaryGeneration.generate_summary import summarize_matched_data
from src.VideoProcessing.extract_frames import extract_frames_from_video
from src.VideoProcessing.ocr_text_extraction import match_subs_with_ocr


def get_subtitles(info, video_url):
    text = get_subtitle_text(info)
    if not text:
        print("No subtitles found. Using Whisper to transcribe...")
        text = transcribe_audio_from_youtube(video_url)

    chunks = chunk_text(text)
    timestamped_chunks = get_timestamped_chunks(info, chunks)
    print("Time stamped chunks are retrieved...")
    return timestamped_chunks


def process_ocr(direct_url):
    print("Starting OCR extraction...")
    ocr_data = extract_frames_from_video(direct_url)
    print("OCR data extracted...")
    return ocr_data


def pipeline(video_url):
    # Step 1: Extract video information
    info = extract_video_info(video_url)
    direct_url = get_video_direct_url(info)
    print("Subtitle info are retrieved...")

    if direct_url is None:
        print("No URL found....")
        return

    # Create a thread pool
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        # Submit tasks to be executed concurrently
        subtitle_future = executor.submit(get_subtitles, info, video_url)
        ocr_future = executor.submit(process_ocr, direct_url)

        # Wait for both to complete and get results
        timestamped_chunks = subtitle_future.result()
        ocr_data = ocr_future.result()

    # Sequential operations that depend on both subtitle and OCR data
    print("Syncing subtitles with OCR data...")
    matched_data = match_subs_with_ocr(timestamped_chunks, ocr_data)
    print("Successfully synced subtitles with OCR data...")

    print("Summarizing.....")
    return summarize_matched_data(matched_data)


if __name__ == "__main__":
    url = "https://youtu.be/LFGBTFxHJII?si=O7wULtF5Dfz5lHzh"
    summary = pipeline(url)
    print("Summary: ", summary)