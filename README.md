# MindGuard Emotion Detection Backend

A FastAPI-based multimodal emotion detection service using DeepFace for facial emotion analysis and librosa/SpeechBrain for voice emotion analysis.

## Features

- **Face Emotion Detection**: Uses DeepFace with multiple backend fallbacks (OpenCV, RetinaFace, MTCNN)
- **Voice Emotion Detection**: Uses librosa for feature extraction with optional SpeechBrain classifier
- **Multimodal Fusion**: Combines face and voice analysis for higher accuracy
- **Docker Ready**: Containerized for easy deployment

## Setup

### Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Docker

```bash
# Build
docker build -t mindguard-emotion .

# Run
docker run -p 8000:8000 mindguard-emotion
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/analyze/face` | POST | Analyze face emotion from image |
| `/analyze/voice` | POST | Analyze voice emotion from audio |
| `/analyze/multimodal` | POST | Combined face + voice analysis |

## Example Usage

```python
import requests

# Face emotion
with open("face.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/analyze/face",
        files={"file": f}
    )
    print(response.json())

# Voice emotion
with open("voice.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/analyze/voice",
        files={"file": f}
    )
    print(response.json())
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 8000 | Server port |
| `LOG_LEVEL` | INFO | Logging level |

## Models

- **Face**: DeepFace with VGG-Face, Facenet, OpenFace backends
- **Voice**: Librosa feature extraction + SpeechBrain IEMOCAP (optional)
