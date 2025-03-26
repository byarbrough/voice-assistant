"""This application is intended to run inside a PyTorch container that support CUDA on the
NVIDIA Jetson Orin Super 8GB Development Board.

The FastAPI app waits for a GET request to /weather upon which it:

1. Starts recording with the webcam for 5 seconds
2. Transcribes the recording using Whisper model via HuggingFace transformer pipeline
3. Makes a request to a cloud-based LLM to process the transcription
4. Uses the processed request to fetch the weather from wttr.in
5. Returns the weather as the response to the original GET request.
"""

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
import sounddevice as sd
import numpy as np
import requests
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, Pipeline, pipeline


# Load the Whisper model
def build_pipeline(model_id: str = "distil-whisper/distil-medium.en") -> Pipeline:
    """Create a HuggingFace transformer pipeline to run Whisper"""
    # Use CUDA if available
    if torch.cuda.is_available():
        print("CUDA available")
        device = "cuda"
        torch_dtype = torch.float16
    else:
        print("CUDA not available")
        device = "cpu"
        torch_dtype = torch.float32

    # Load model
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    # Send to selected device
    model.to(device)
    # Create pipeline
    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        torch_dtype=torch_dtype,
        device=device,
    )
    return pipe


# Method to record audio
def record_audio(duration_seconds: int = 5) -> np.ndarray:
    """Record duration_seconds of audio from default microphone.
    Return a single channel numpy array."""
    sample_rate = 16000  # Hz
    samples = int(duration_seconds * sample_rate)
    # Will use default microphone; on Jetson this is likely a USB WebCam
    audio = sd.rec(samples, samplerate=sample_rate, channels=1, dtype=np.float32)
    # Blocks until recording complete
    sd.wait()
    # Model expects single axis
    return np.squeeze(audio, axis=1)


# TODO: Method to process transcription via the LLM
def llm_process(raw_text: str) -> str:
    return "New+York"


# Method to make request to wttr.in
def wttr_request(location: str) -> str:
    headers = {
        "User-Agent": "curl/7.81.0",
        "Accept": "text/plain",  # explicitly tell wttr.in to return terminal-style text
    }
    url = f"https://wttr.in/{location}?format="
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Error: {response}")
        raise requests.exceptions.RequestException

    # Return just the text of the request
    return response.text


pipe: Pipeline = build_pipeline()
print("HuggingFace pipeline created")

app: FastAPI = FastAPI()
print("FastAPI app starting")


@app.get("/weather", response_class=PlainTextResponse)
def get_weather() -> str:
    # Start recording
    # print("Recording...")
    # audio = record_audio()
    # print("Done")
    # print("Transcribing...")
    # speech = pipe(audio)
    # print(f"Speech: {speech}")
    # print("Processing with LLM...")
    # location = llm_process(speech)
    location = llm_process("speech")
    print(f"LLM returned: {location}")
    print("Making request to wttr.in")
    weather = wttr_request(location)
    print("Done")

    return weather
