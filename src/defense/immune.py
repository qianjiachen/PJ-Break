"""
Immune defense (adapted for audio).

Inference-time alignment checking defense.
"""

from dataclasses import dataclass
from typing import Optional, List
import numpy as np
import time

from ..models_data import DefenseResult, Audio
from ..models.audio_llm import AudioLLM


@dataclass
class ImmuneConfig:
    """Configuration for Immune defense."""
    num_perturbations: int = 5
    perturbation_std: float = 0.01
    consistency_threshold: float = 0.7


class ImmuneDefense:
    """
    Immune defense adapted for audio.
    
    Checks response consistency under small perturbations.
    Adversarial inputs tend to produce inconsistent responses.
    """
    
    def __init__(self, config: Optional[ImmuneConfig] = None):
        """
        Initialize Immune defense.
        
        Args:
            config: Immune configuration
        """
        self.config = config or ImmuneConfig()
    
    def detect(
        self,
        audio: Audio,
        model: Optional[AudioLLM] = None
    ) -> DefenseResult:
        """
        Detect adversarial audio using consistency checking.
        
        Args:
            audio: Audio to check
            model: Target model
        
        Returns:
            DefenseResult
        """
        start_time = time.time()
        
        if model is None:
            # Can't check without model
            return DefenseResult(
                audio_id=audio.id,
                is_blocked=False,
                aggregate_score=0.5,
                text_score=0.5,
                prosody_score=0.5,
                refusal_score=0.0,
                latency_ms=0,
                defense_name="immune"
            )
        
        # Generate perturbed versions
        responses = []
        
        # Original response
        orig_response = model.generate(audio.data, audio.sample_rate)
        responses.append(orig_response.text)
        
        # Perturbed responses
        for _ in range(self.config.num_perturbations):
            perturbed = self._perturb_audio(audio)
            response = model.generate(perturbed.data, perturbed.sample_rate)
            responses.append(response.text)
        
        # Check consistency
        consistency_score = self._compute_consistency(responses)
        
        # Low consistency = likely adversarial
        is_blocked = consistency_score < self.config.consistency_threshold
        
        latency = (time.time() - start_time) * 1000
        
        return DefenseResult(
            audio_id=audio.id,
            is_blocked=is_blocked,
            aggregate_score=consistency_score,
            text_score=consistency_score,
            prosody_score=1.0,
            refusal_score=0.0,
            latency_ms=latency,
            defense_name="immune"
        )
    
    def _perturb_audio(self, audio: Audio) -> Audio:
        """Add small perturbation to audio."""
        noise = np.random.randn(len(audio.data)).astype(np.float32)
        noise *= self.config.perturbation_std
        
        perturbed_data = audio.data + noise
        perturbed_data = np.clip(perturbed_data, -1.0, 1.0)
        
        return Audio(
            id=f"{audio.id}_perturbed",
            data=perturbed_data,
            sample_rate=audio.sample_rate,
            instruction_id=audio.instruction_id,
            prosody=audio.prosody,
            duration=audio.duration
        )
    
    def _compute_consistency(self, responses: List[str]) -> float:
        """
        Compute response consistency score.
        
        Uses simple word overlap as proxy for semantic similarity.
        """
        if len(responses) < 2:
            return 1.0
        
        # Compute pairwise similarities
        similarities = []
        
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                sim = self._word_overlap(responses[i], responses[j])
                similarities.append(sim)
        
        return np.mean(similarities) if similarities else 1.0
    
    def _word_overlap(self, text1: str, text2: str) -> float:
        """Compute word overlap between two texts."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
