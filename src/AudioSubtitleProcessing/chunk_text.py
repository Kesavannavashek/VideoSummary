import spacy

# Load the spaCy model (You can use 'en_core_web_sm' for a lighter model)
nlp = spacy.load("en_core_web_sm")


def chunk_text(input_text):
    # Process the text using spaCy
    doc = nlp(input_text)

    # Break the text into sentences (chunks)
    chunks = list(doc.sents)

    # Return the list of chunks
    return [chunk.text for chunk in chunks]




# import spacy
# import requests
#
# # Load the spaCy model
# nlp = spacy.load("en_core_web_sm")
#
# def process_subtitles_grouped_with_spacy(subtitle_url, max_gap_ms=1000, max_duration_ms=10000):
#     try:
#         response = requests.get(subtitle_url)
#         response.raise_for_status()
#         subtitle_data = response.json()
#     except Exception as e:
#         print(f"Subtitle error: {e}")
#         return []
#
#     results = []
#     group = []
#     start_time = None
#     end_time = None
#
#     for event in subtitle_data.get('events', []):
#         if not event.get('segs') or not event.get('tStartMs'):
#             continue
#
#         t_start = event['tStartMs']
#         duration = event.get('dDurationMs', 1000)
#         t_end = t_start + duration
#
#         text = ' '.join(seg.get('utf8', '').strip() for seg in event['segs'] if seg.get('utf8')).strip()
#         if not text:
#             continue
#
#         # Process the text with spaCy (for example, sentence segmentation or tokenization)
#         doc = nlp(text)
#         sentences = [sent.text for sent in doc.sents]  # Split text into sentences
#
#         if start_time is None:
#             start_time = t_start
#             end_time = t_end
#             group.extend(sentences)
#         elif (t_start - end_time <= max_gap_ms) and ((t_end - start_time) <= max_duration_ms):
#             group.extend(sentences)
#             end_time = t_end
#         else:
#             full_text = ' '.join(group)
#             results.append({
#                 'start_time': start_time,
#                 'end_time': end_time,
#                 'text': full_text
#             })
#
#             # Start a new chunk for the next group of text
#             start_time = t_start
#             end_time = t_end
#             group = sentences
#
#     # Add the last chunk if any
#     if group:
#         full_text = ' '.join(group)
#         results.append({
#             'start_time': start_time,
#             'end_time': end_time,
#             'text': full_text
#         })
#
#     return results
#
