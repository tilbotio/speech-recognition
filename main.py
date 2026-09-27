import multiprocessing
import os
import subprocess

import numpy as np
import whisper

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from uvicorn import run

load_dotenv()

PORT = os.getenv("PORT", 8081)
MODELNAME = os.getenv("MODELNAME", "tiny")
LANGUAGE = os.getenv("LANGUAGE", "auto")

model = whisper.load_model(MODELNAME)

app = FastAPI()

@app.get("/healthcheck")
async def healthcheck():
    return {"status": "ok"}

def decode_audio(audio_bytes: bytes, sample_rate: int = 16000) -> np.ndarray:
    command = [
        "ffmpeg",
        "-nostdin",
        "-i",
        "pipe:0",
        "-f",
        "f32le",
        "-ac",
        "1",
        "-acodec",
        "pcm_f32le",
        "-ar",
        str(sample_rate),
        "pipe:1",
    ]

    try:
        result = subprocess.run(
            command,
            input=audio_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required to decode uploaded audio.") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode(errors="ignore").strip()
        raise ValueError(f"ffmpeg could not decode uploaded audio: {stderr}") from exc

    decoded = np.frombuffer(result.stdout, dtype=np.float32)
    if decoded.size == 0:
        raise ValueError("Decoded audio was empty.")

    return decoded

@app.get("/healthcheck")
async def healthcheck():
    return {"status": "ok"}

@app.post("/")
async def get_intent(audio: UploadFile = File(...)):
    theaudio = await audio.read()
    try:
        data = decode_audio(theaudio)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    transcribe_kwargs = {}
    if LANGUAGE != 'auto':
        transcribe_kwargs['language'] = LANGUAGE

    result = model.transcribe(data, **transcribe_kwargs)

    return {"intent": result["text"]}

if __name__ == "__main__":
    multiprocessing.freeze_support()  # For Windows support
    run(app, host="0.0.0.0", port=PORT, reload=False, workers=1)
