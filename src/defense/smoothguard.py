"""
SmoothGuard defense.

Randomized smoothing for audio adversarial detection.
"""

from dataclasses import dataclass
from typing import Optional, List
import numpy as np
import time

from ..models_data import DefenseResult, Audio
from ..models.audio_llm import AudioLLM


@dataclass
class SmoothGuardConfig:
    """Configuration for SmoothGuard defense."""
    num_samples: int = 10
    noise_std: float = 0.01
    agreement_threshold: float = 0.7
    seed: Optional[int] = None


class SmoothGuardDefense:
    """
    SmoothGuard: Randomized smoothing defense.
    
    Generates multiple noisy versions of input and checks
    if model responses are consistent. Adversarial inputs
    tend to produce inconsistent responses under noise.
    """
    
    def __init__(self, config: Optional[SmoothGuardConfig] = None):
        """
        Initialize SmoothGuard defense.
        
        Args:
            config: SmoothGuard configuration
        """
        self.config = config or SmoothGuardConfig()
        self.rng = np.random.RandomState(self.config.seed)
    
    def detect(
        self,
        audio: Audio,
        model: Optional[AudioLLM] = None
    ) -> DefenseResult:
        """
        Detect adversarial audio using randomized smoothing.
        
        Args:
            audio: Audio to check
            model: Target model
        
        Returns:
            DefenseResult
        """
        start_time = time.time()
        
        if model is None:
            return DefenseResult(
                audio_id=audio.id,
                is_blocked=False,
                aggregate_score=0.5,
                text_score=0.5,
                prosody_score=0.5,
                refusal_score=0.0,
                latency_ms=0,
                defense_name="smoothguard"
            )
        
        # Generate noisy samples and get responses
        responses = []
        
        for _ in range(self.config.num_samples):
            noisy_audio = self._add_noise(audio)
            response = model.generate(noisy_audio.data, noisy_audio.sample_rate)
            responses.append(response.text)
        
        # Check response consistency
        agreement_score = self._check_consistency(responses)
        
        # Low agreement = likely adversarial
        is_blocked = agreement_score < self.config.agreement_threshold
        
        latency = (time.time() - start_time) * 1000
        
        return DefenseResult(
            audio_id=audio.id,
            is_blocked=is_blocked,
            aggregate_score=agreement_score,
            text_score=agreement_score,
            prosody_score=1.0,
            refusal_score=0.0,
            latency_ms=latency,
            defense_name="smoothguard"
        )
    
    def _add_noise(self, audio: Audio) -> Audio:
        """Add Gaussian noise to audio."""
        noise = self.rng.randn(len(audio.data)).astype(np.float32)
        noise *= self.config.noise_std
        
        noisy_data = audio.data + noise
        noisy_data = np.clip(noisy_data, -1.0, 1.0)
        
        return Audio(
            id=f"{audio.id}_noisy",
            data=noisy_data,
            sample_rate=audio.sample_rate,
            instruction_id=audio.instruction_id,
            prosody=audio.prosody,
            duration=audio.duration
        )
    
    def _check_consistency(self, responses: List[str]) -> float:
        """
        Check consistency of responses.
        
        Returns agreement score in [0, 1].
        """
        if len(responses) < 2:
            return 1.0
        
        # Classify each response as refusal or compliance
        classifications = []
        
        refusal_phrases = [
            "cannot", "can't", "sorry", "apologize",
            "unable", "won't", "refuse"
        ]
        
        for response in responses:
            response_lower = response.lower()
            is_refusal = any(phrase in response_lower for phrase in refusal_phrases)
            classifications.append(is_refusal)
        
        # Compute agreement (majority class proportion)
        refusal_count = sum(classifications)
        compliance_count = len(classifications) - refusal_count
        
        majority_count = max(refusal_count, compliance_count)
        agreement = majority_count / len(classifications)
        
        return agreement
