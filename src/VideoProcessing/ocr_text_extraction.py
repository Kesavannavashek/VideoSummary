from paddleocr import PaddleOCR
import logging
# logging.getLogger('ppocr').setLevel(logging.ERROR)

ocr = PaddleOCR(
    use_angle_cls=True,
    use_gpu=True,
    det_db_box_thresh=0.2,
    rec_algorithm='CRNN',
    det_db_unclip_ratio=2.0,
    max_text_length=100,
    use_space_char=True
)


def extract_text_from_frame(frame):
    result = ocr.ocr(frame, cls=True)
    if not result or not result[0]:
        return None

    # Sort lines top to bottom
    lines = sorted(result[0], key=lambda x: min(pt[1] for pt in x[0]))
    formatted_lines = []

    for line in lines:
        text = line[1][0]
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

        if collected_texts:
            matched.append((text, tuple(collected_texts)))
        else:
            matched.append((text, ()))

    return matched

