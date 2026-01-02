"""
TTS engine module for PJ-Break experiment reproduction.

Implements controllable TTS with prosody variations as specified in the paper
(Appendix B.1): Azure Neural TTS with SSML parameters for different prosody conditions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np
import os
import time
import warnings

# Try to import Azure Speech SDK
try:
    import azure.cognitiveservices.speech as speechsdk
    HAS_AZURE = True
except ImportError:
    HAS_AZURE = False
    warnings.warn("Azure Speech SDK not installed. Azure TTS unavailable.")


@dataclass
class TTSResult:
    """Result of TTS synthesis."""
    audio_data: np.ndarray
    sample_rate: int
    duration: float
    prosody: str
    text: str
    success: bool
    error: Optional[str] = None


class TTSEngine(ABC):
    """Abstract base class for TTS engines."""
    
    # Prosody configurations from paper Table 11
    PROSODY_CONFIGS = {
        "neutral": {"style": "default"},
        "panic": {"style": "terrified", "degree": 2.0},
        "anger": {"style": "angry", "degree": 1.5},
        "commanding": {"style": "newscast-formal", "pitch": "-15%"},
        "fast": {"rate": "+50%"},
        "whisper": {"style": "whispering"},
    }
    
    @abstractmethod
    def synthesize(self, text: str, prosody: str = "neutral") -> TTSResult:
        """
        Synthesize speech from text with specified prosody.
        
        Args:
            text: Text to synthesize
            prosody: Prosody condition (neutral, panic, anger, commanding, fast, whisper)
        
        Returns:
            TTSResult with audio data and metadata
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the TTS engine is available."""
        pass
    
    def get_prosody_config(self, prosody: str) -> Dict[str, Any]:
        """Get prosody configuration for a condition."""
        return self.PROSODY_CONFIGS.get(prosody, self.PROSODY_CONFIGS["neutral"])


class AzureTTSEngine(TTSEngine):
    """
    Azure Neural TTS engine with prosody control.
    
    Uses Microsoft Azure Cognitive Services Speech SDK with SSML
    for fine-grained prosody control.
    """
    
    def __init__(
        self,
        subscription_key: Optional[str] = None,
        region: str = "eastus",
        voice_id: str = "en-US-JennyNeural",
        male_voice_id: str = "en-US-GuyNeural",
        sample_rate: int = 16000
    ):
        """
        Initialize Azure TTS engine.
        
        Args:
            subscription_key: Azure Speech subscription key (or from env AZURE_SPEECH_KEY)
            region: Azure region (default: eastus)
            voice_id: Default voice ID for female voice
            male_voice_id: Voice ID for male voice (used for commanding)
            sample_rate: Output sample rate
        """
        self.subscription_key = subscription_key or os.environ.get("AZURE_SPEECH_KEY")
        self.region = region
        self.voice_id = voice_id
        self.male_voice_id = male_voice_id
        self.sample_rate = sample_rate
        
        self._speech_config = None
        self._init_speech_config()
    
    def _init_speech_config(self):
        """Initialize Azure Speech configuration."""
        if not HAS_AZURE:
            return
        
        if not self.subscription_key:
            warnings.warn("Azure Speech key not provided. Set AZURE_SPEECH_KEY env var.")
            return
        
        try:
            self._speech_config = speechsdk.SpeechConfig(
                subscription=self.subscription_key,
                region=self.region
            )
            self._speech_config.set_speech_synthesis_output_format(
                speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm
            )
        except Exception as e:
            warnings.warn(f"Failed to initialize Azure Speech: {e}")
    
    def is_available(self) -> bool:
        """Check if Azure TTS is available."""
        return HAS_AZURE and self._speech_config is not None
    
    def synthesize(self, text: str, prosody: str = "neutral") -> TTSResult:
        """
        Synthesize speech using Azure Neural TTS.
        
        Args:
            text: Text to synthesize
            prosody: Prosody condition
        
        Returns:
            TTSResult with audio data
        """
        if not self.is_available():
            return self._fallback_synthesize(text, prosody)
        
        try:
            ssml = self._build_ssml(text, prosody)
            
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=self._speech_config,
                audio_config=None  # Output to memory
            )
            
            result = synthesizer.speak_ssml_async(ssml).get()
            
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                # Convert audio data to numpy array
                audio_data = np.frombuffer(result.audio_data, dtype=np.int16)
                audio_data = audio_data.astype(np.float32) / 32768.0  # Normalize to [-1, 1]
                
                duration = len(audio_data) / self.sample_rate
                
                return TTSResult(
                    audio_data=audio_data,
                    sample_rate=self.sample_rate,
                    duration=duration,
                    prosody=prosody,
                    text=text,
                    success=True
                )
            else:
                error_msg = f"Synthesis failed: {result.reason}"
                if result.cancellation_details:
                    error_msg += f" - {result.cancellation_details.error_details}"
                
                return TTSResult(
                    audio_data=np.array([], dtype=np.float32),
                    sample_rate=self.sample_rate,
                    duration=0.0,
                    prosody=prosody,
                    text=text,
                    success=False,
                    error=error_msg
                )
        
        except Exception as e:
            return TTSResult(
                audio_data=np.array([], dtype=np.float32),
                sample_rate=self.sample_rate,
                duration=0.0,
                prosody=prosody,
                text=text,
                success=False,
                error=str(e)
            )
    
    def _build_ssml(self, text: str, prosody: str) -> str:
        """
        Build SSML markup for prosody control.
        
        Args:
            text: Text to synthesize
            prosody: Prosody condition
        
        Returns:
            SSML string
        """
        config = self.get_prosody_config(prosody)
        
        # Select voice based on prosody
        voice = self.male_voice_id if prosody == "commanding" else self.voice_id
        
        # Build SSML
        ssml_parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" ',
            'xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="en-US">',
            f'<voice name="{voice}">'
        ]
        
        # Add style if specified
        style = config.get("style", "default")
        degree = config.get("degree", 1.0)
        
        if style != "default":
            ssml_parts.append(f'<mstts:express-as style="{style}" styledegree="{degree}">')
        
        # Add prosody modifications
        prosody_attrs = []
        if "rate" in config:
            prosody_attrs.append(f'rate="{config["rate"]}"')
        if "pitch" in config:
            prosody_attrs.append(f'pitch="{config["pitch"]}"')
        
        if prosody_attrs:
            ssml_parts.append(f'<prosody {" ".join(prosody_attrs)}>')
            ssml_parts.append(text)
            ssml_parts.append('</prosody>')
        else:
            ssml_parts.append(text)
        
        # Close tags
        if style != "default":
            ssml_parts.append('</mstts:express-as>')
        
        ssml_parts.append('</voice>')
        ssml_parts.append('</speak>')
        
        return ''.join(ssml_parts)
    
    def _fallback_synthesize(self, text: str, prosody: str) -> TTSResult:
        """Fallback synthesis when Azure is unavailable."""
        # Generate simple placeholder audio (silence with noise)
        duration = len(text.split()) * 0.3  # Rough estimate
        num_samples = int(self.sample_rate * duration)
        
        # Generate very quiet noise as placeholder
        audio_data = np.random.randn(num_samples).astype(np.float32) * 0.001
        
        return TTSResult(
            audio_data=audio_data,
            sample_rate=self.sample_rate,
            duration=duration,
            prosody=prosody,
            text=text,
            success=True,
            error="Using fallback (Azure unavailable)"
        )


class MockTTSEngine(TTSEngine):
    """
    Mock TTS engine for testing.
    
    Generates synthetic audio with prosody-like characteristics
    without requiring external APIs.
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
    
    def is_available(self) -> bool:
        return True
    
    def synthesize(self, text: str, prosody: str = "neutral") -> TTSResult:
        """
        Generate mock audio with prosody characteristics.
        
        Args:
            text: Text to synthesize
            prosody: Prosody condition
        
        Returns:
            TTSResult with synthetic audio
        """
        # Estimate duration based on text length
        words = text.split()
        base_wpm = 150  # Base words per minute
        
        # Adjust rate based on prosody
        rate_multipliers = {
            "neutral": 1.0,
            "panic": 1.2,
            "anger": 1.15,
            "commanding": 0.95,
            "fast": 1.5,
            "whisper": 0.85,
        }
        rate_mult = rate_multipliers.get(prosody, 1.0)
        
        duration = len(words) / (base_wpm * rate_mult) * 60
        duration = max(duration, 0.5)  # Minimum 0.5 seconds
        
        num_samples = int(self.sample_rate * duration)
        
        # Generate audio with prosody characteristics
        audio_data = self._generate_prosodic_audio(num_samples, prosody)
        
        return TTSResult(
            audio_data=audio_data,
            sample_rate=self.sample_rate,
            duration=duration,
            prosody=prosody,
            text=text,
            success=True
        )
    
    def _generate_prosodic_audio(self, num_samples: int, prosody: str) -> np.ndarray:
        """Generate synthetic audio with prosody characteristics."""
        t = np.linspace(0, num_samples / self.sample_rate, num_samples)
        
        # Base F0 and characteristics by prosody
        prosody_params = {
            "neutral": {"f0": 150, "f0_var": 20, "amplitude": 0.3},
            "panic": {"f0": 180, "f0_var": 60, "amplitude": 0.5},
            "anger": {"f0": 170, "f0_var": 45, "amplitude": 0.45},
            "commanding": {"f0": 120, "f0_var": 15, "amplitude": 0.4},
            "fast": {"f0": 155, "f0_var": 25, "amplitude": 0.35},
            "whisper": {"f0": 140, "f0_var": 8, "amplitude": 0.15},
        }
        
        params = prosody_params.get(prosody, prosody_params["neutral"])
        
        # Generate F0 contour with variation
        f0_contour = params["f0"] + np.random.randn(num_samples) * params["f0_var"] * 0.1
        f0_contour = np.clip(f0_contour, 80, 400)
        
        # Generate audio with harmonics
        audio = np.zeros(num_samples, dtype=np.float32)
        phase = 0
        
        for i in range(num_samples):
            # Fundamental
            audio[i] = np.sin(phase) * params["amplitude"]
            # Add harmonics
            audio[i] += np.sin(2 * phase) * params["amplitude"] * 0.5
            audio[i] += np.sin(3 * phase) * params["amplitude"] * 0.25
            
            phase += 2 * np.pi * f0_contour[i] / self.sample_rate
        
        # Add some noise
        noise_level = 0.02 if prosody != "whisper" else 0.05
        audio += np.random.randn(num_samples).astype(np.float32) * noise_level
        
        # Apply amplitude envelope (fade in/out)
        envelope = np.ones(num_samples)
        fade_samples = int(0.05 * self.sample_rate)
        envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
        envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
        audio *= envelope
        
        return audio.astype(np.float32)


def get_tts_engine(engine_type: str = "auto", **kwargs) -> TTSEngine:
    """
    Factory function to get appropriate TTS engine.
    
    Args:
        engine_type: "azure", "mock", or "auto"
        **kwargs: Engine-specific arguments
    
    Returns:
        TTSEngine instance
    """
    if engine_type == "azure":
        return AzureTTSEngine(**kwargs)
    elif engine_type == "mock":
        return MockTTSEngine(**kwargs)
    elif engine_type == "auto":
        # Try Azure first, fall back to mock
        azure = AzureTTSEngine(**kwargs)
        if azure.is_available():
            return azure
        return MockTTSEngine(**kwargs)
    else:
        raise ValueError(f"Unknown engine type: {engine_type}")
