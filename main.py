import os
import subprocess

import numpy as np
import whisper

from fastapi import FastAPI, File, HTTPException, UploadFile

modelname = 'tiny'
language = 'auto'

if 'MODELNAME' in os.environ:
    modelname = os.environ['MODELNAME']

if 'LANGUAGE' in os.environ:
    language = os.environ['LANGUAGE']

model = whisper.load_model(modelname)

app = FastAPI()


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
    if language != 'auto':
        transcribe_kwargs['language'] = language

    result = model.transcribe(data, **transcribe_kwargs)

    return {"intent": result["text"]}