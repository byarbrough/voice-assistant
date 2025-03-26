# This Dockerfile runs a FastAPI application that serves the Whisper model and process requests to get the weather.

# For JetPack 6
FROM nvcr.io/nvidia/pytorch:24.11-py3-igpu

WORKDIR /app
# HuggingFace Model Cache
ENV HF_HOME="/huggingface/"

# C library that python-sounddevice binds to
RUN apt-get update && apt-get install -y --no-install-recommends \
	libportaudio2 \
	&& rm -rf /var/lib/apt/lists/

# We want to upgrade pip but not other dependencies
# because don't want to accidentally overwrite special
# builds container came with
RUN pip install --upgrade --no-cache-dir pip && \
	pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENTRYPOINT ["fastapi", "dev", "app.py"]
