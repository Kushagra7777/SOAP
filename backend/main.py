from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import uuid
import torch
import whisper
import os

from backend.script import generate_soap_summary  # Adjust if needed

app = FastAPI()

# ✅ Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 📁 Set up upload directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# ✅ Select best available device: CUDA (NVIDIA), MPS (Apple), or CPU
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# ✅ Load Whisper model on selected device
model = whisper.load_model("base", device=device)

@app.get("/")
def read_root():
    return {"message": "Welcome to the transcription and summarization API!"}

@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    try:
        file_id = str(uuid.uuid4())
        file_path = UPLOAD_DIR / f"{file_id}_{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())
        return {"file_path": str(file_path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class FilePathInput(BaseModel):
    file_path: str

@app.post("/transcribe")
def transcribe(data: FilePathInput):
    file_path = data.file_path
    result = model.transcribe(file_path, language="ja")
    transcript = result["text"]
    with open("original_transcript.txt", "w", encoding="utf-8") as f:
        f.write(transcript)
    return {"transcript": transcript}

@app.post("/summarize")
def summarize_route():
    try:
        soap_summary = generate_soap_summary("original_transcript.txt")
        return {"summary": soap_summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# uvicorn backend.main:app --reload