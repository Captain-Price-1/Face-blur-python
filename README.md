# Face Blur App

Local web app to blur selected faces in uploaded videos.

## Setup

```bash
brew install ffmpeg cmake     # cmake is needed by dlib
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
uvicorn app.main:app --reload
# open http://localhost:8000
```

## Test

```bash
pytest
```
