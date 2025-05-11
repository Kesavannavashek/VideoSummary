import requests
import json


def get_subtitle_text(info):
    # print(info)
    for source in [info.get("subtitles", {}), info.get("automatic_captions", {})]:
        english = source.get("en")
        if english:
            try:
                url = english[0].get('url')
                response = requests.get(url)
                response.raise_for_status()
                return json.loads(response.text)
            except Exception as e:
                print(f"Error fetching subtitles: {e}")
                return {}
    print("No English subtitles found.")
    return {}

def get_subtitle_text_from_info(info):
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

    for event in subtitle_data.get('events', []):

        if not event.get('segs') or not event.get('tStartMs'):
            continue

        for seg in event['segs']:
            text = seg['utf8']

            if text == '\n':
                continue

            current_text += text
            current_length += len(text.replace(" ", ""))


            if chunk_index < len(chunked_subs):
                chunk = chunked_subs[chunk_index]
                chunk_length = len(chunk.replace(" ", ""))


                if current_length >= chunk_length:
                    start_time = event['tStartMs']
                    duration = event['dDurationMs']
                    results.append({
                        'text': chunk,
                        'start': start_time,
                        'duration_ms': duration,
                        'end':start_time+duration
                    })

                    current_text = ""
                    current_length = 0
                    chunk_index += 1

    return results

