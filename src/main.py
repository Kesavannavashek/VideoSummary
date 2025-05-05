import asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi import File, UploadFile
import shutil
import os
import time
from pathlib import Path

app = FastAPI()

# Global variable to store WebSocket connection
connection = None

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- CONFIG ----------
UPLOAD_DIR = Path("uploaded_videos")
UPLOAD_DIR.mkdir(exist_ok=True)


# ---------- WebSocket Endpoint ----------
@app.websocket("/ws/status")
async def websocket_endpoint(websocket: WebSocket):
    global connection
    await websocket.accept()

    # Store the connection
    connection = websocket
    print("📥 WebSocket connected for video processing.")

    try:
        # Simulate video processing
        await process_video()
    except WebSocketDisconnect:
        connection = None
        print("❌ WebSocket disconnected.")


# ---------- Helper Functions for Processing Stages ----------
async def process_video():
    stages = [
        "Video Downloading",
        "Speech Extraction",
        "OCR Extraction",
        "Summary Generation",
        "Processing Complete"
    ]

    # Simulate processing each stage with delays
    for stage in stages:
        await send_status_update(stage)
        await asyncio.sleep(2)  # Simulate time delay for each stage

    print("✅ Processing complete.")


# Function to send real-time status updates via WebSocket
async def send_status_update(status: str):
    if connection:
        await connection.send_text(f"Video processing status: {status}")
    else:
        print(f"❌ No active WebSocket connection.")


# ---------- File Upload Endpoint ----------
@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    video_path = UPLOAD_DIR / file.filename

    # Save the uploaded file
    with video_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print(f"📥 Uploaded file saved to: {video_path}")
    return JSONResponse(content={"status": "upload_successful", "file_path": str(video_path)})


# ---------- Running the FastAPI server ----------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
