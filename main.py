"""
MindGuard Emotion Detection Service
A FastAPI-based multimodal emotion detection backend using DeepFace and SpeechBrain.
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import base64
import io
from typing import Optional

from face_analyzer import FaceAnalyzer
from voice_analyzer import VoiceAnalyzer

app = FastAPI(
    title="MindGuard Emotion Detection API",
    description="Advanced multimodal emotion detection using DeepFace and SpeechBrain",
    version="1.0.0"
)

# CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize analyzers
face_analyzer = FaceAnalyzer()
voice_analyzer = VoiceAnalyzer()


@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers."""
    return {
        "status": "healthy",
        "service": "emotion-detection",
        "face_analyzer": face_analyzer.is_ready(),
        "voice_analyzer": voice_analyzer.is_ready()
    }


@app.post("/analyze/face")
async def analyze_face(file: Optional[UploadFile] = File(None), image_base64: Optional[str] = None):
    """
    Analyze facial emotion from an uploaded image or base64-encoded frame.
    
    Args:
        file: Image file upload
        image_base64: Base64-encoded image string
    
    Returns:
        Emotion analysis result with dominant emotion and confidence scores
    """
    try:
        if file:
            image_data = await file.read()
        elif image_base64:
            # Remove data URL prefix if present
            if "," in image_base64:
                image_base64 = image_base64.split(",")[1]
            image_data = base64.b64decode(image_base64)
        else:
            raise HTTPException(status_code=400, detail="No image provided")
        
        result = face_analyzer.analyze(image_data)
        return JSONResponse(content=result)
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "emotion": "neutral", "confidence": 0.0}
        )


@app.post("/analyze/voice")
async def analyze_voice(file: UploadFile = File(...)):
    """
    Analyze voice emotion from an uploaded audio file.
    
    Args:
        file: Audio file (WAV, MP3, WebM, etc.)
    
    Returns:
        Emotion analysis result with dominant emotion and confidence scores
    """
    try:
        audio_data = await file.read()
        result = voice_analyzer.analyze(audio_data, file.filename)
        return JSONResponse(content=result)
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "emotion": "neutral", "confidence": 0.0}
        )


@app.post("/analyze/multimodal")
async def analyze_multimodal(
    face_image: Optional[UploadFile] = File(None),
    voice_audio: Optional[UploadFile] = File(None),
    face_base64: Optional[str] = None
):
    """
    Analyze both face and voice for fused emotion detection.
    
    Returns weighted combination of face and voice emotions.
    """
    results = {"face": None, "voice": None, "fused": None}
    
    try:
        # Analyze face if provided
        if face_image or face_base64:
            if face_image:
                image_data = await face_image.read()
            else:
                if "," in face_base64:
                    face_base64 = face_base64.split(",")[1]
                image_data = base64.b64decode(face_base64)
            results["face"] = face_analyzer.analyze(image_data)
        
        # Analyze voice if provided
        if voice_audio:
            audio_data = await voice_audio.read()
            results["voice"] = voice_analyzer.analyze(audio_data, voice_audio.filename)
        
        # Fuse results if both available
        if results["face"] and results["voice"]:
            face_conf = results["face"].get("confidence", 0)
            voice_conf = results["voice"].get("confidence", 0)
            
            # Weighted average (face weighted higher for visual context)
            if face_conf > voice_conf:
                results["fused"] = {
                    "emotion": results["face"]["emotion"],
                    "confidence": (face_conf * 0.6) + (voice_conf * 0.4),
                    "source": "face_primary"
                }
            else:
                results["fused"] = {
                    "emotion": results["voice"]["emotion"],
                    "confidence": (voice_conf * 0.6) + (face_conf * 0.4),
                    "source": "voice_primary"
                }
        elif results["face"]:
            results["fused"] = results["face"]
        elif results["voice"]:
            results["fused"] = results["voice"]
        else:
            results["fused"] = {"emotion": "neutral", "confidence": 0.0}
        
        return JSONResponse(content=results)
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "fused": {"emotion": "neutral", "confidence": 0.0}}
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
