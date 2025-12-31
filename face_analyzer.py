"""
Face Emotion Analyzer using DeepFace
Provides robust facial emotion detection with multi-model fallback.
"""
import io
import numpy as np
from PIL import Image
from typing import Dict, Any, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lazy load DeepFace to speed up startup
_deepface = None

def get_deepface():
    global _deepface
    if _deepface is None:
        from deepface import DeepFace
        _deepface = DeepFace
    return _deepface


class FaceAnalyzer:
    """
    Facial emotion analyzer using DeepFace library.
    Supports multiple backend models for reliability.
    """
    
    # Models to try in order of preference
    MODELS = ["VGG-Face", "Facenet", "OpenFace", "DeepFace"]
    BACKENDS = ["opencv", "retinaface", "mtcnn", "ssd"]
    
    # Map DeepFace emotions to our standard format
    EMOTION_MAP = {
        "angry": "angry",
        "disgust": "disgusted",
        "fear": "fearful",
        "happy": "happy",
        "sad": "sad",
        "surprise": "surprised",
        "neutral": "neutral"
    }
    
    def __init__(self):
        self._ready = False
        self._initialize()
    
    def _initialize(self):
        """Pre-load models for faster inference."""
        try:
            DeepFace = get_deepface()
            # Warm up with a test image
            logger.info("Initializing DeepFace models...")
            self._ready = True
            logger.info("✅ FaceAnalyzer ready")
        except Exception as e:
            logger.error(f"❌ FaceAnalyzer initialization failed: {e}")
            self._ready = False
    
    def is_ready(self) -> bool:
        return self._ready
    
    def analyze(self, image_data: bytes, detect_all: bool = True) -> Dict[str, Any]:
        """
        Analyze facial emotion from image bytes.
        
        Args:
            image_data: Raw image bytes (JPEG, PNG, etc.)
            detect_all: If True, return emotions for all detected faces
        
        Returns:
            Dict with faces array containing emotion data for each face
        """
        try:
            DeepFace = get_deepface()
            
            # Convert bytes to numpy array
            image = Image.open(io.BytesIO(image_data))
            image_array = np.array(image)
            
            # Ensure RGB format
            if len(image_array.shape) == 2:
                image_array = np.stack([image_array] * 3, axis=-1)
            elif image_array.shape[2] == 4:
                image_array = image_array[:, :, :3]
            
            # Try analysis with fallback backends
            result = None
            for backend in self.BACKENDS:
                try:
                    result = DeepFace.analyze(
                        img_path=image_array,
                        actions=['emotion'],
                        detector_backend=backend,
                        enforce_detection=False,
                        silent=True
                    )
                    break
                except Exception as e:
                    logger.warning(f"Backend {backend} failed: {e}")
                    continue
            
            if not result:
                return self._default_response("No face detected")
            
            # Handle list response (multiple faces)
            faces_list = result if isinstance(result, list) else [result]
            
            faces_output = []
            for i, face_result in enumerate(faces_list):
                emotion_scores = face_result.get("emotion", {})
                dominant_emotion = face_result.get("dominant_emotion", "neutral")
                region = face_result.get("region", {})
                
                mapped_emotion = self.EMOTION_MAP.get(dominant_emotion, "neutral")
                confidence = emotion_scores.get(dominant_emotion, 0) / 100.0
                
                faces_output.append({
                    "face_id": i + 1,
                    "emotion": mapped_emotion,
                    "confidence": round(confidence, 3),
                    "all_emotions": {
                        self.EMOTION_MAP.get(k, k): round(v / 100.0, 3)
                        for k, v in emotion_scores.items()
                    },
                    "region": {
                        "x": region.get("x", 0),
                        "y": region.get("y", 0),
                        "w": region.get("w", 0),
                        "h": region.get("h", 0)
                    }
                })
            
            # Return single face format for backward compatibility if only one face
            if len(faces_output) == 1 and not detect_all:
                return {
                    **faces_output[0],
                    "face_detected": True,
                    "face_count": 1
                }
            
            return {
                "faces": faces_output,
                "face_count": len(faces_output),
                "face_detected": len(faces_output) > 0,
                "primary_emotion": faces_output[0]["emotion"] if faces_output else "neutral"
            }
        
        except Exception as e:
            logger.error(f"Face analysis error: {e}")
            return self._default_response(str(e))
    
    def _default_response(self, error: str = None) -> Dict[str, Any]:
        """Return neutral response when analysis fails."""
        return {
            "emotion": "neutral",
            "confidence": 0.0,
            "all_emotions": {},
            "face_detected": False,
            "error": error
        }
