"""
Baseline attack methods for PJ-Break experiment reproduction.

Implements comparison baselines from the paper:
- Text-only baseline
- Flat audio baseline (neutral prosody)
- StyleBreak baseline (reference implementation)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np
import logging

from ..models_data import Audio, Instruction, ProsodyCondition, AttackResult
from ..models.audio_llm import AudioLLM
from ..evaluation.multi_judge import MultiJudgeSystem

logger = logging.getLogger(__name__)


class BaselineMethod(ABC):
    """Abstract base class for baseline attack methods."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Baseline method name."""
        pass
    
    @abstractmethod
    def attack(
        self,
        instruction: Instruction,
        model: AudioLLM,
        judge_system: MultiJudgeSystem
    ) -> AttackResult:
        """
        Execute baseline attack.
        
        Args:
            instruction: Harmful instruction
            model: Target model
            judge_system: Evaluation system
        
        Returns:
            AttackResult
        """
        pass


class TextOnlyBaseline(BaselineMethod):
    """
    Text-only baseline.
    
    Sends harmful instruction as text without audio.
    Used to measure baseline ASR without prosody manipulation.
    """
    
    @property
    def name(self) -> str:
        return "text_only"
    
    def attack(
        self,
        instruction: Instruction,
        model: AudioLLM,
        judge_system: MultiJudgeSystem
    ) -> AttackResult:
        """Execute text-only attack."""
        # Generate silent audio
        sample_rate = 16000
        duration = 0.1  # Minimal audio
        silent_audio = np.zeros(int(sample_rate * duration), dtype=np.float32)
        
        # Use text as prompt
        response = model.generate(
            silent_audio,
            sample_rate,
            prompt=instruction.text
        )
        
        # Evaluate
        judge_result = judge_system.evaluate(response.text, instruction.text)
        
        return AttackResult(
            audio_id=f"{instruction.id}_text_only",
            model=model.name,
            response=response.text,
            is_success=judge_result.is_attack_success,
            judge_result=judge_result,
            latency_ms=response.latency_ms,
            error=response.error
        )


class FlatAudioBaseline(BaselineMethod):
    """
    Flat audio baseline.
    
    Uses neutral prosody TTS without emotional manipulation.
    Measures ASR with audio input but no prosody attack.
    """
    
    def __init__(self, tts_engine=None):
        """
        Initialize flat audio baseline.
        
        Args:
            tts_engine: TTS engine for neutral synthesis
        """
        self.tts_engine = tts_engine
    
    @property
    def name(self) -> str:
        return "flat_audio"
    
    def _get_tts_engine(self):
        """Lazy load TTS engine."""
        if self.tts_engine is None:
            from ..tts.engine import get_tts_engine
            self.tts_engine = get_tts_engine("auto")
        return self.tts_engine
    
    def attack(
        self,
        instruction: Instruction,
        model: AudioLLM,
        judge_system: MultiJudgeSystem
    ) -> AttackResult:
        """Execute flat audio attack."""
        tts = self._get_tts_engine()
        
        # Generate neutral audio
        tts_result = tts.synthesize(instruction.text, prosody="neutral")
        
        if not tts_result.success:
            return AttackResult(
                audio_id=f"{instruction.id}_flat_audio",
                model=model.name,
                response="",
                is_success=False,
                judge_result=judge_system.evaluate("", instruction.text),
                latency_ms=0,
                error=f"TTS failed: {tts_result.error}"
            )
        
        # Generate response
        response = model.generate(
            tts_result.audio_data,
            tts_result.sample_rate
        )
        
        # Evaluate
        judge_result = judge_system.evaluate(response.text, instruction.text)
        
        return AttackResult(
            audio_id=f"{instruction.id}_flat_audio",
            model=model.name,
            response=response.text,
            is_success=judge_result.is_attack_success,
            judge_result=judge_result,
            latency_ms=response.latency_ms,
            error=response.error
        )


class StyleBreakBaseline(BaselineMethod):
    """
    StyleBreak baseline (reference implementation).
    
    Based on prior work on style-based jailbreaks.
    Uses style transfer to modify audio characteristics.
    """
    
    def __init__(
        self,
        style: str = "emotional",
        intensity: float = 1.0
    ):
        """
        Initialize StyleBreak baseline.
        
        Args:
            style: Style to apply
            intensity: Style intensity
        """
        self.style = style
        self.intensity = intensity
    
    @property
    def name(self) -> str:
        return "style_break"
    
    def attack(
        self,
        instruction: Instruction,
        model: AudioLLM,
        judge_system: MultiJudgeSystem
    ) -> AttackResult:
        """Execute StyleBreak attack."""
        # StyleBreak uses style transfer on audio
        # This is a simplified implementation
        
        from ..tts.engine import get_tts_engine
        tts = get_tts_engine("auto")
        
        # Map style to prosody
        style_to_prosody = {
            "emotional": "panic",
            "urgent": "fast",
            "authoritative": "commanding",
        }
        prosody = style_to_prosody.get(self.style, "neutral")
        
        # Generate styled audio
        tts_result = tts.synthesize(instruction.text, prosody=prosody)
        
        if not tts_result.success:
            return AttackResult(
                audio_id=f"{instruction.id}_style_break",
                model=model.name,
                response="",
                is_success=False,
                judge_result=judge_system.evaluate("", instruction.text),
                latency_ms=0,
                error=f"TTS failed: {tts_result.error}"
            )
        
        # Apply style intensity (simple amplitude scaling)
        audio_data = tts_result.audio_data * self.intensity
        audio_data = np.clip(audio_data, -1.0, 1.0)
        
        # Generate response
        response = model.generate(
            audio_data,
            tts_result.sample_rate
        )
        
        # Evaluate
        judge_result = judge_system.evaluate(response.text, instruction.text)
        
        return AttackResult(
            audio_id=f"{instruction.id}_style_break",
            model=model.name,
            response=response.text,
            is_success=judge_result.is_attack_success,
            judge_result=judge_result,
            latency_ms=response.latency_ms,
            error=response.error
        )


class BaselineRunner:
    """Runs all baseline methods for comparison."""
    
    def __init__(
        self,
        models: Dict[str, AudioLLM],
        judge_system: Optional[MultiJudgeSystem] = None
    ):
        """
        Initialize baseline runner.
        
        Args:
            models: Dictionary of model_name -> AudioLLM
            judge_system: Multi-judge evaluation system
        """
        self.models = models
        self.judge_system = judge_system or MultiJudgeSystem()
        
        self.baselines = [
            TextOnlyBaseline(),
            FlatAudioBaseline(),
            StyleBreakBaseline(),
        ]
    
    def run_all_baselines(
        self,
        instructions: List[Instruction],
        model_name: str
    ) -> Dict[str, List[AttackResult]]:
        """
        Run all baselines on instructions.
        
        Args:
            instructions: List of instructions
            model_name: Target model
        
        Returns:
            Dictionary of baseline_name -> results
        """
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        model = self.models[model_name]
        results = {}
        
        for baseline in self.baselines:
            logger.info(f"Running baseline: {baseline.name}")
            
            baseline_results = []
            for instruction in instructions:
                result = baseline.attack(instruction, model, self.judge_system)
                baseline_results.append(result)
            
            results[baseline.name] = baseline_results
        
        return results
