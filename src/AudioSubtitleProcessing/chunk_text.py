import spacy

nlp = spacy.load("en_core_web_sm")

def chunk_text(input_text):
    doc = nlp(input_text)

    chunks = list(doc.sents)

    return [chunk.text for chunk in chunks]

def split_text_for_local(chunks):
    result = []

    for start, end, text in chunks:
        doc = nlp(text)
        total_duration = end - start
        total_chars = len(text)

        char_offset = 0
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if not sent_text or len(sent_text) < 3:
                continue

            sent_chars = len(sent_text)
            sent_start = start + total_duration * (char_offset / total_chars)
            sent_end = sent_start + total_duration * (sent_chars / total_chars)
            char_offset += sent_chars

            result.append((round(sent_start, 2), round(sent_end, 2), sent_text))
    return result


def batch_chunks_duration_range(chunks, min_duration_ms=30000, max_duration_ms=50000):
    batches = []
    current_batch = []
    batch_start = None

    for start, end, sentence in chunks:
        start_ms = int(start * 1000)
        end_ms = int(end * 1000)

        if not current_batch:
            batch_start = start_ms

        current_batch.append((start_ms, end_ms, sentence))
        batch_end = end_ms
        batch_duration = batch_end - batch_start

        if batch_duration >= min_duration_ms:
            if batch_duration <= max_duration_ms:
                merged_text = " ".join([s for _, _, s in current_batch])
                batches.append({
                    'start': batch_start,
                    'end': batch_end,
                    'text': merged_text
                })
                current_batch = []
            else:
                if len(current_batch) > 1:
                    last = current_batch.pop()
                    merged_text = " ".join([s for _, _, s in current_batch])
                    batch_end = current_batch[-1][1]
                    batches.append({
                        'start': batch_start,
                        'end': batch_end,
                        'text': merged_text
                    })
                    current_batch = [last]
                    batch_start = last[0]
                else:
                    merged_text = current_batch[0][2]
                    batches.append({
                        'start': start_ms,
                        'end': end_ms,
                        'text': merged_text
                    })
                    current_batch = []

    if current_batch:
        merged_text = " ".join([s for _, _, s in current_batch])
        batch_end = current_batch[-1][1]
        batches.append({
            'start': batch_start,
            'end': batch_end,
            'text': merged_text
        })

    return batches



