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

# TODO: Load the Whisper model

# TODO: Method to process transcription via the LLM

# TODO: Method to make request to wttr.in

app: FastAPI = FastAPI()


@app.get("/weather")
def get_weather() -> str:
    return "Sunny and 75"
