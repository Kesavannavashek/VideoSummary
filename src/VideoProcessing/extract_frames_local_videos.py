import cv2
from src.VideoProcessing.ocr_text_extraction import extract_text_from_frame


def calculate_histogram_diff(frame1, frame2):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
    return diff


def extract_significant_ocr_frames(video_path, hist_threshold=0.1, target_fps=2):
    cap = cv2.VideoCapture(video_path)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = max(int(actual_fps // target_fps), 1)

    results = []
    prev_frame = None
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        timestamp = frame_idx / actual_fps
        cv2.imshow("Current Frame", frame)
        cv2.waitKey(1)

        if prev_frame is not None:
            diff = calculate_histogram_diff(prev_frame, frame)
            if diff >= hist_threshold:
                print(f"[INFO] Significant frame @ {timestamp:.2f}s (diff={diff:.3f})")
                try:
                    ocr_result = extract_text_from_frame(frame)
                    texts = []
                    if ocr_result and isinstance(ocr_result[0], list):
                        texts = [line[1][0] for line in ocr_result[0] if line and len(line) > 1]
                    print("text: ",texts)
                    if texts:
                        results.append({
                            "timestamp": round(timestamp, 2),
                            "text": " ".join(texts)
                        })
                except Exception as e:
                    print(f"[WARN] OCR failed at frame {frame_idx}: {e}")

        prev_frame = frame
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    return results


if __name__ == "__main__":
    video_path = "../input.mp4"
    results = extract_significant_ocr_frames(video_path, hist_threshold=0.35, target_fps=2)
    print("res: ",results)
    for r in results:
        print(f"[{r['timestamp']}s] {r['text']}")
