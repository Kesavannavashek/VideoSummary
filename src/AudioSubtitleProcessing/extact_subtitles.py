import requests
import json


def get_subtitle_text(info):
    """Fetches and parses English automatic captions from video info."""
    subtitles = info.get("automatic_captions", {})
    if 'en' in subtitles:
        subtitle_url = subtitles['en'][0]['url']
        response = requests.get(subtitle_url)
        subtitle_content = response.text
        subtitle_data = json.loads(subtitle_content)
        return extract_text_for_spacy(subtitle_data)
    else:
        print("Subtitles not available in English.")
        return ""

def get_subtitle_text_from_info(info):
    """Fetches and parses English automatic captions from video info."""
    subtitles = info.get("automatic_captions", {})
    if 'en' in subtitles:
        subtitle_url = subtitles['en'][0]['url']
        response = requests.get(subtitle_url)
        subtitle_content = response.text
        return json.loads(subtitle_content)
    else:
        print("Subtitles not available in English.")
        return ""


def extract_text_for_spacy(subtitle_data):
    """Extracts plain text from subtitle JSON."""
    parsed_text = ""
    for event in subtitle_data.get("events", []):
        for seg in event.get("segs", []):
            text = seg.get("utf8", "")
            if text.strip():
                parsed_text += text + " "
    return parsed_text.strip()


def get_timestamped_chunks(info, chunked_subs):
    subtitle_data = get_subtitle_text_from_info(info)
    # print("sub_data: ",subtitle_data)
    results = []
    current_text = ""
    current_length = 0
    chunk_index = 0

    # Loop through all events as per the provided code
    for event in subtitle_data.get('events', []):
        # Skip events without 'segs' or 'tStartMs'
        if not event.get('segs') or not event.get('tStartMs'):
            continue

        # Process segments in the event
        for seg in event['segs']:
            text = seg['utf8']
            # Skip newline-only segments
            if text == '\n':
                continue

            current_text += text
            current_length += len(text.replace(" ", ""))

            # Check if there are chunks to process
            if chunk_index < len(chunked_subs):
                chunk = chunked_subs[chunk_index]
                chunk_length = len(chunk.replace(" ", ""))  # Calculate chunk length (ignoring spaces)

                # If accumulated length matches or exceeds chunk length
                if current_length >= chunk_length:
                    start_time = event['tStartMs']
                    duration = event['dDurationMs']
                    results.append({
                        'text': chunk,
                        'start': start_time,
                        'duration_ms': duration,
                        'end':start_time+duration
                    })
                    # Reset for next chunk
                    current_text = ""
                    current_length = 0
                    chunk_index += 1

    return results

