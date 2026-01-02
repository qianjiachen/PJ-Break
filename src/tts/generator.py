"""
Prosody variation generator for PJ-Break experiment reproduction.

Generates all prosody variations for seed instructions using TTS.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Callable
import numpy as np
import time
import logging
from pathlib import Path

from .engine import TTSEngine, TTSResult, get_tts_engine
from ..audio.processor import AudioProcessor

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result of generating prosody variations for one instruction."""
    instruction_id: str
    text: str
    variations: Dict[str, TTSResult]
    all_success: bool
    errors: List[str]


class ProsodyGenerator:
    """
    Generates prosody variations for instructions.
    
    For each instruction, generates 6 prosody variations:
    - Neutral
    - Panic
    - Anger
    - Commanding
    - Fast
    - Whisper
    """
    
    PROSODY_CONDITIONS = ["neutral", "panic", "anger", "commanding", "fast", "whisper"]
    
    def __init__(
        self,
        tts_engine: Optional[TTSEngine] = None,
        audio_processor: Optional[AudioProcessor] = None,
        retry_attempts: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize prosody generator.
        
        Args:
            tts_engine: TTS engine to use (auto-detected if None)
            audio_processor: Audio processor for normalization
            retry_attempts: Number of retry attempts on failure
            retry_delay: Delay between retries in seconds
        """
        self.tts_engine = tts_engine or get_tts_engine("auto")
        self.audio_processor = audio_processor or AudioProcessor()
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
    
    def generate_prosody_variations(
        self,
        text: str,
        instruction_id: str = "",
        conditions: Optional[List[str]] = None,
        normalize: bool = True
    ) -> GenerationResult:
        """
        Generate all prosody variations for a text.
        
        Args:
            text: Text to synthesize
            instruction_id: Identifier for the instruction
            conditions: List of prosody conditions (default: all 6)
            normalize: Whether to normalize audio
        
        Returns:
            GenerationResult with all variations
        """
        if conditions is None:
            conditions = self.PROSODY_CONDITIONS
        
        variations = {}
        errors = []
        
        for prosody in conditions:
            result = self._generate_with_retry(text, prosody)
            
            if result.success:
                # Normalize audio if requested
                if normalize and len(result.audio_data) > 0:
                    processed = self.audio_processor.process(
                        result.audio_data,
                        result.sample_rate
                    )
                    result = TTSResult(
                        audio_data=processed.data,
                        sample_rate=processed.sample_rate,
                        duration=processed.duration,
                        prosody=prosody,
                        text=text,
                        success=True
                    )
                
                variations[prosody] = result
            else:
                errors.append(f"{prosody}: {result.error}")
                variations[prosody] = result
        
        all_success = all(v.success for v in variations.values())
        
        return GenerationResult(
            instruction_id=instruction_id,
            text=text,
            variations=variations,
            all_success=all_success,
            errors=errors
        )
    
    def _generate_with_retry(self, text: str, prosody: str) -> TTSResult:
        """Generate with retry logic."""
        last_error = None
        
        for attempt in range(self.retry_attempts):
            result = self.tts_engine.synthesize(text, prosody)
            
            if result.success:
                return result
            
            last_error = result.error
            logger.warning(f"TTS attempt {attempt + 1} failed for {prosody}: {result.error}")
            
            if attempt < self.retry_attempts - 1:
                time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
        
        return TTSResult(
            audio_data=np.array([], dtype=np.float32),
            sample_rate=16000,
            duration=0.0,
            prosody=prosody,
            text=text,
            success=False,
            error=f"Failed after {self.retry_attempts} attempts: {last_error}"
        )
    
    def generate_batch(
        self,
        instructions: List[Dict[str, str]],
        output_dir: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[GenerationResult]:
        """
        Generate prosody variations for a batch of instructions.
        
        Args:
            instructions: List of dicts with 'id' and 'text' keys
            output_dir: Optional directory to save audio files
            progress_callback: Optional callback(current, total) for progress
        
        Returns:
            List of GenerationResult for each instruction
        """
        results = []
        total = len(instructions)
        
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for i, inst in enumerate(instructions):
            inst_id = inst.get("id", f"inst_{i}")
            text = inst.get("text", "")
            
            if progress_callback:
                progress_callback(i + 1, total)
            
            result = self.generate_prosody_variations(text, inst_id)
            results.append(result)
            
            # Save audio files if output_dir specified
            if output_dir and result.all_success:
                self._save_variations(result, output_dir)
        
        return results
    
    def _save_variations(self, result: GenerationResult, output_dir: str) -> None:
        """Save audio variations to files."""
        import soundfile as sf
        
        output_path = Path(output_dir)
        
        for prosody, tts_result in result.variations.items():
            if tts_result.success and len(tts_result.audio_data) > 0:
                filename = f"{result.instruction_id}_{prosody}.wav"
                filepath = output_path / filename
                
                try:
                    sf.write(
                        str(filepath),
                        tts_result.audio_data,
                        tts_result.sample_rate
                    )
                except Exception as e:
                    logger.error(f"Failed to save {filepath}: {e}")
