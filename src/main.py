import asyncio
from src.youtube_pipeline import pipeline
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from fastapi import File, UploadFile
import shutil
import os
import time
from pathlib import Path
from src.youtube_pipeline import send_status
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


@app.websocket("/ws/status")
async def websocket_endpoint(websocket: WebSocket):
    global connection

    try:
        await websocket.accept()
        print("📥 WebSocket connected.")
        connection = websocket

        while True:
            # Keep the connection alive
            await websocket.receive_text()

    except WebSocketDisconnect:
        print("❌ WebSocket disconnected.")
        connection = None

# ---------- File Upload Endpoint ----------
@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    video_path = UPLOAD_DIR / file.filename

    with video_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print(f"📥 Uploaded file saved to: {video_path}")
    return JSONResponse(content={"status": "upload_successful", "file_path": str(video_path)})

# ---------- YouTube URL Submission Endpoint ----------
@app.post("/submit_youtube_url")
async def submit_youtube_url(request: Request):
    global connection
    data = await request.json()
    youtube_url = data.get("url")

    if not youtube_url:
        return JSONResponse(content={"error": "Missing 'url' field"}, status_code=400)

    await send_status(connection,f"[STATUS]📥 Received YouTube URL: {youtube_url}")

    if connection:
        await pipeline(youtube_url, connection)
    else:
        print("❌ No active WebSocket connection.")
        return JSONResponse(content={"error": "No active WebSocket connection"}, status_code=400)

    return JSONResponse(content={"status": "url_received", "url": youtube_url})



# ---------- Running the FastAPI server ----------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
