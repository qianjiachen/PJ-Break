"""
Time-Domain Noise Flooding (TDNF) defense.

Adds noise to audio to disrupt adversarial patterns.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import time

from ..models_data import DefenseResult, Audio


@dataclass
class TDNFConfig:
    """Configuration for TDNF defense."""
    noise_level_db: float = 20.0  # Noise level in dB
    seed: Optional[int] = None


class TDNFDefense:
    """
    Time-Domain Noise Flooding defense.
    
    Adds Gaussian noise to audio input to disrupt
    adversarial prosody patterns.
    """
    
    def __init__(self, config: Optional[TDNFConfig] = None):
        """
        Initialize TDNF defense.
        
        Args:
            config: TDNF configuration
        """
        self.config = config or TDNFConfig()
        self.rng = np.random.RandomState(self.config.seed)
    
    def apply(self, audio: Audio) -> Audio:
        """
        Apply noise flooding to audio.
        
        Args:
            audio: Input audio
        
        Returns:
            Audio with added noise
        """
        # Convert dB to linear scale
        noise_amplitude = 10 ** (self.config.noise_level_db / 20) * 0.01
        
        # Generate noise
        noise = self.rng.randn(len(audio.data)).astype(np.float32) * noise_amplitude
        
        # Add noise
        noisy_data = audio.data + noise
        
        # Clip to valid range
        noisy_data = np.clip(noisy_data, -1.0, 1.0)
        
        return Audio(
            id=f"{audio.id}_tdnf",
            data=noisy_data,
            sample_rate=audio.sample_rate,
            instruction_id=audio.instruction_id,
            prosody=audio.prosody,
            duration=audio.duration,
            transcript=audio.transcript
        )
    
    def detect(self, audio: Audio, model=None) -> DefenseResult:
        """
        TDNF doesn't detect - it transforms.
        
        Returns a result indicating the audio was processed.
        """
        start_time = time.time()
        
        # Apply noise
        _ = self.apply(audio)
        
        latency = (time.time() - start_time) * 1000
        
        return DefenseResult(
            audio_id=audio.id,
            is_blocked=False,  # TDNF doesn't block, it transforms
            aggregate_score=1.0,
            text_score=1.0,
            prosody_score=1.0,
            refusal_score=0.0,
            latency_ms=latency,
            defense_name="tdnf"
        )
