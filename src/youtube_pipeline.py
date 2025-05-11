import asyncio
from starlette.websockets import WebSocket, WebSocketState
import time
from src.AudioSubtitleProcessing.chunk_text import chunk_text, split_text_spacy
from src.AudioSubtitleProcessing.extact_subtitles import (
    get_subtitle_text,
    get_timestamped_chunks,
    extract_text_for_spacy,
)
from src.AudioSubtitleProcessing.extract_speech import transcribe_youtube_video
from src.GetVideoInfo.get_video_info import extract_video_info, get_video_direct_url
from src.SummaryGeneration.generate_summary import summarize_matched_data
from src.VideoProcessing.extract_frames_local_videos import stream_and_collect_frames
from src.VideoProcessing.ocr_text_extraction import match_subs_with_ocr

# --- Safe WebSocket messaging ---
async def send_status(websocket, message: str):
    if websocket and websocket.application_state == WebSocketState.CONNECTED:
        try:
            await websocket.send_text(message)
        except Exception as e:
            print(f"[WebSocket Send Error] {e}")
    else:
        print(f"[WebSocket Closed] {message}")

# --- Subtitle/Audio Text Extraction ---
async def extract_text_and_timestamps(info, video_url, websocket):
    isAudio = False
    text = get_subtitle_text(info)

    if not text:
        isAudio = True
        await send_status(websocket, "[STATUS]No subtitles found. Using Whisper to transcribe...")
        text = transcribe_youtube_video(video_url)
        text = split_text_spacy(text)
        return text, [], isAudio
    else:
        text = extract_text_for_spacy(text)
        chunks = chunk_text(text)
        timestamped_chunks = get_timestamped_chunks(info, chunks)
        await send_status(websocket, "[STATUS]Chunks are timestamped...")
        return text, timestamped_chunks, isAudio

# --- Main Pipeline ---
async def pipeline(video_url: str, websocket: WebSocket = None):
    try:
        info = extract_video_info(video_url)
        video_title = info["title"]
        direct_url = get_video_direct_url(info)

        if not direct_url:
            await send_status(websocket, "[STATUS]❌ Not a YouTube video URL. Quitting.")
            return

        await send_status(websocket, "[STATUS]🚀 Launching tasks...")

        # Run extraction and OCR in parallel
        text_proc_task = extract_text_and_timestamps(info, video_url, websocket)
        ocr_task = asyncio.to_thread(stream_and_collect_frames, direct_url)

        text, timestamped_chunks, isAudio = await text_proc_task
        ocr_data = await ocr_task

        await send_status(websocket, "[STATUS]🖼️ OCR data extracted...")
        matched_data = match_subs_with_ocr(timestamped_chunks, ocr_data)
        await send_status(websocket, "[STATUS]🔗 Matching OCR with subtitle data...")
        print("here1")
        await send_status(websocket, "[STATUS]🧠 Generating summary...")
        print("here2")

        await summarize_matched_data(matched_data,websocket=websocket ,title=video_title)

        await send_status(websocket, f"[STATUS]✅ Summary generated successfully.")


    except Exception as e:
        await send_status(websocket, f"[STATUS]❌ Error occurred: {str(e)}")



if __name__ == "__main__":
    # url = "https://youtu.be/4EP8YzcN0hQ?si=-lfdX62fpjkgvq8O"
    url = "https://youtu.be/ldYLYRNaucM?si=ximNENy042FR06sR"
    s = time.time()
    # url = "https://youtu.be/CPk8pffKV64?si=oPERlnocqOwfuCxW"
    matchedData = pipeline(url)
    print("Total Time:", time.time() - s)
    print("summary: ",matchedData)








































































# import concurrent.futures
# from src.AudioSubtitleProcessing.chunk_text import chunk_text
# from src.AudioSubtitleProcessing.extact_subtitles import get_subtitle_text, get_timestamped_chunks
# from src.AudioSubtitleProcessing.extract_speech import transcribe_audio_from_youtube
# from src.GetVideoInfo.get_video_info import extract_video_info, get_video_direct_url
# from src.SummaryGeneration.generate_summary import summarize_matched_data
# from src.VideoProcessing.extract_frames import extract_frames_from_video
# from src.VideoProcessing.ocr_text_extraction import match_subs_with_ocr
#
#
# def get_subtitles(info, video_url):
#     text = get_subtitle_text(info)
#     if not text:
#         print("No subtitles found. Using Whisper to transcribe...")
#         text = transcribe_audio_from_youtube(video_url)
#
#     chunks = chunk_text(text)
#     timestamped_chunks = get_timestamped_chunks(info, chunks)
#     print("Time stamped chunks are retrieved...")
#     return timestamped_chunks
#
#
# def process_ocr(direct_url):
#     print("Starting OCR extraction...")
#     ocr_data = extract_frames_from_video(direct_url)
#     print("OCR data extracted...")
#     return ocr_data
#
#
# def pipeline(video_url):
#     # Step 1: Extract video information
#     info = extract_video_info(video_url)
#     print("info: ",info)
    # direct_url = get_video_direct_url(info)
    # print("Subtitle info are retrieved...")
    #
    # if direct_url is None:
    #     print("No URL found....")
    #     return
    #
    # # Create a thread pool
    # with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    #     # Submit tasks to be executed concurrently
    #     subtitle_future = executor.submit(get_subtitles, info, video_url)
    #     ocr_future = executor.submit(process_ocr, direct_url)
    #
    #     # Wait for both to complete and get results
    #     timestamped_chunks = subtitle_future.result()
    #     ocr_data = ocr_future.result()
    #
    # # Sequential operations that depend on both subtitle and OCR data
    # print("Syncing subtitles with OCR data...")
    # matched_data = match_subs_with_ocr(timestamped_chunks, ocr_data)
    # print("Successfully synced subtitles with OCR data...")
    #
    # print("Summarizing.....")
    # return summarize_matched_data(matched_data)


# if __name__ == "__main__":
#     url = "https://youtu.be/CPk8pffKV64?si=oPERlnocqOwfuCxW"
#     summary = pipeline(url)
#     print("Summary: ", summary)