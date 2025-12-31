"""
Voice Emotion Analyzer using librosa and SpeechBrain
Provides audio-based emotion detection from voice recordings.
"""
import io
import tempfile
import os
import numpy as np
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lazy imports for heavy libraries
_librosa = None
_speechbrain_model = None


def get_librosa():
    global _librosa
    if _librosa is None:
        import librosa
        _librosa = librosa
    return _librosa


class VoiceAnalyzer:
    """
    Voice emotion analyzer using audio feature extraction.
    Uses librosa for feature extraction and optional SpeechBrain for classification.
    """
    
    # Map emotion predictions to standard format
    EMOTION_MAP = {
        "angry": "angry",
        "happy": "happy",
        "sad": "sad",
        "neutral": "neutral",
        "fear": "fearful",
        "disgust": "disgusted",
        "surprise": "surprised",
        "calm": "calm",
        "excited": "happy"
    }
    
    # Audio feature thresholds for simple emotion detection
    FEATURE_PROFILES = {
        "angry": {"energy_high": True, "pitch_high": True, "tempo_fast": True},
        "sad": {"energy_low": True, "pitch_low": True, "tempo_slow": True},
        "happy": {"energy_high": True, "pitch_high": True, "tempo_fast": True},
        "fearful": {"energy_mid": True, "pitch_high": True, "tempo_varied": True},
        "neutral": {"energy_mid": True, "pitch_mid": True, "tempo_normal": True}
    }
    
    def __init__(self):
        self._ready = False
        self._speechbrain_classifier = None
        self._initialize()
    
    def _initialize(self):
        """Initialize audio processing libraries."""
        try:
            librosa = get_librosa()
            logger.info("✅ VoiceAnalyzer ready (librosa mode)")
            self._ready = True
            
            # Try to load SpeechBrain model (optional enhancement)
            try:
                from speechbrain.inference.interfaces import foreign_class
                self._speechbrain_classifier = foreign_class(
                    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
                    pymodule_file="custom_interface.py",
                    classname="CustomEncoderWav2vec2Classifier",
                    run_opts={"device": "cpu"}
                )
                logger.info("✅ SpeechBrain classifier loaded")
            except Exception as e:
                logger.warning(f"⚠️ SpeechBrain not available, using librosa features: {e}")
                self._speechbrain_classifier = None
                
        except Exception as e:
            logger.error(f"❌ VoiceAnalyzer initialization failed: {e}")
            self._ready = False
    
    def is_ready(self) -> bool:
        return self._ready
    
    def analyze(self, audio_data: bytes, filename: str = "audio.wav") -> Dict[str, Any]:
        """
        Analyze voice emotion from audio bytes.
        
        Args:
            audio_data: Raw audio bytes
            filename: Original filename for format detection
        
        Returns:
            Dict with emotion, confidence, and audio features
        """
        try:
            librosa = get_librosa()
            
            # Write to temp file for processing
            ext = os.path.splitext(filename)[1] or ".wav"
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(audio_data)
                tmp_path = tmp.name
            
            try:
                # Load audio
                y, sr = librosa.load(tmp_path, sr=16000, mono=True)
                
                # Try SpeechBrain first if available
                if self._speechbrain_classifier:
                    return self._analyze_with_speechbrain(tmp_path, y, sr)
                
                # Fallback to feature-based analysis
                return self._analyze_with_features(y, sr)
                
            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        
        except Exception as e:
            logger.error(f"Voice analysis error: {e}")
            return self._default_response(str(e))
    
    def _analyze_with_speechbrain(self, audio_path: str, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Use SpeechBrain model for emotion classification."""
        try:
            out_prob, score, index, text_lab = self._speechbrain_classifier.classify_file(audio_path)
            
            emotion = text_lab[0].lower() if text_lab else "neutral"
            confidence = float(score.squeeze().max())
            
            return {
                "emotion": self.EMOTION_MAP.get(emotion, emotion),
                "confidence": round(confidence, 3),
                "method": "speechbrain",
                "audio_features": self._extract_features(y, sr)
            }
        except Exception as e:
            logger.warning(f"SpeechBrain failed, falling back to features: {e}")
            return self._analyze_with_features(y, sr)
    
    def _analyze_with_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analyze emotion using audio feature extraction."""
        librosa = get_librosa()
        
        features = self._extract_features(y, sr)
        
        # Simple heuristic-based emotion detection
        emotion = "neutral"
        confidence = 0.5
        
        energy = features.get("rms_energy", 0)
        pitch_mean = features.get("pitch_mean", 0)
        tempo = features.get("tempo", 0)
        
        # Classify based on features
        if energy > 0.15 and pitch_mean > 200 and tempo > 120:
            emotion = "angry"
            confidence = 0.7
        elif energy < 0.05 and pitch_mean < 150 and tempo < 80:
            emotion = "sad"
            confidence = 0.65
        elif energy > 0.1 and pitch_mean > 180 and tempo > 110:
            emotion = "happy"
            confidence = 0.6
        elif energy > 0.08 and pitch_mean > 220:
            emotion = "fearful"
            confidence = 0.55
        else:
            emotion = "neutral"
            confidence = 0.5
        
        return {
            "emotion": emotion,
            "confidence": round(confidence, 3),
            "method": "librosa_features",
            "audio_features": features
        }
    
    def _extract_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract audio features for analysis."""
        librosa = get_librosa()
        
        try:
            # RMS Energy
            rms = librosa.feature.rms(y=y)[0]
            rms_energy = float(np.mean(rms))
            
            # Pitch (fundamental frequency)
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_values = pitches[pitches > 0]
            pitch_mean = float(np.mean(pitch_values)) if len(pitch_values) > 0 else 0
            
            # Tempo
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            tempo = float(tempo) if isinstance(tempo, (int, float, np.number)) else float(tempo[0]) if hasattr(tempo, '__len__') else 0
            
            # Spectral centroid (brightness)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            brightness = float(np.mean(spectral_centroid))
            
            # Zero crossing rate (noisiness)
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            zcr_mean = float(np.mean(zcr))
            
            return {
                "rms_energy": round(rms_energy, 4),
                "pitch_mean": round(pitch_mean, 2),
                "tempo": round(tempo, 2),
                "brightness": round(brightness, 2),
                "zero_crossing_rate": round(zcr_mean, 4),
                "duration_seconds": round(len(y) / sr, 2)
            }
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return {}
    
    def _default_response(self, error: str = None) -> Dict[str, Any]:
        """Return neutral response when analysis fails."""
        return {
            "emotion": "neutral",
            "confidence": 0.0,
            "method": "fallback",
            "audio_features": {},
            "error": error
        }
