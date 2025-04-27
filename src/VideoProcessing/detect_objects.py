from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
print("Model loaded successfully..")

def object_detect(frame):
    results = model(frame)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = model.names[int(box.cls[0].item())]


            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{cls} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


    cv2.imshow("YOLOv8 Detection", frame)
