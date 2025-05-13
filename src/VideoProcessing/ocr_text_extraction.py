from paddleocr import PaddleOCR
import logging
logging.getLogger('ppocr').setLevel(logging.ERROR)

ocr = PaddleOCR(
    use_angle_cls=False,
    use_gpu=True,
    det_db_box_thresh=0.2,
    rec_algorithm='CRNN',
    det_db_unclip_ratio=2.0,
    max_text_length=100,
    use_space_char=True
)

# ocr = PaddleOCR(
#     det_model_dir='en_PP-OCRv4_det',
#     rec_model_dir='en_PP-OCRv4_rec',
#     use_angle_cls=False,  # turn off for speed if not needed
#     lang='en',
#     use_gpu=True,
#     precision='fp16'
# )
def extract_text_from_frame(frame):
    result = ocr.ocr(frame, cls=True)
    # print("result: ",result)
    if not result or not result[0]:
        return None

    # Sort lines top to bottom
    lines = sorted(result[0], key=lambda x: min(pt[1] for pt in x[0]))
    formatted_lines = []

    for line in lines:
        text = line[1][0]
        # print("TEXT: ",text)
        x_positions = [pt[0] for pt in line[0]]
        min_x = min(x_positions)

        indent_level = int(min_x // 20)
        indentation = " " * (indent_level * 2)

        formatted_lines.append(f"{indentation}{text}")

    formatted_text = "\n".join(formatted_lines)
    return formatted_text

def match_subs_with_ocr(subs, ocr_data):
    subs = [(d['start'], d['end'], d['text']) for d in subs if d]
    matched = []
    for start, end, text in subs:
        collected_texts = [ocr_text for ts, ocr_text in ocr_data if start <= ts <= end]

        matched.append({
            "start": start,
            "end": end,
            "text": text,
            "ocr_texts": collected_texts
        })

    print("Matched:", matched)
    return matched


def combine_matched_chunks(matched_data, max_words=1000):
    combined = []
    buffer_text = ""
    buffer_ocr = []
    word_count = 0

    for text, ocrs in matched_data:
        total_new_words = len(text.split()) + sum(len(ocr.split()) for ocr in ocrs)
        if word_count + total_new_words <= max_words:
            buffer_text += "\n" + text
            buffer_ocr.extend(ocrs)
            word_count += total_new_words
        else:
            combined.append((buffer_text.strip(), buffer_ocr))
            buffer_text = text
            buffer_ocr = ocrs
            word_count = total_new_words

    if buffer_text:
        combined.append((buffer_text.strip(), buffer_ocr))

    return combined


