"""
Attack executor for PJ-Break experiment reproduction.

Executes attacks against Audio LLMs and collects results.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Callable, Any
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..models_data import Audio, AttackResult, JudgeResult
from ..models.audio_llm import AudioLLM, ModelResponse
from ..evaluation.multi_judge import MultiJudgeSystem

logger = logging.getLogger(__name__)


@dataclass
class AttackConfig:
    """Configuration for attack execution."""
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout_seconds: float = 60.0
    parallel_workers: int = 1


class AttackExecutor:
    """
    Executes attacks against Audio LLMs.
    
    Integrates model inference with multi-judge evaluation.
    """
    
    def __init__(
        self,
        models: Dict[str, AudioLLM],
        judge_system: Optional[MultiJudgeSystem] = None,
        config: Optional[AttackConfig] = None
    ):
        """
        Initialize attack executor.
        
        Args:
            models: Dictionary of model_name -> AudioLLM
            judge_system: Multi-judge evaluation system
            config: Attack configuration
        """
        self.models = models
        self.judge_system = judge_system or MultiJudgeSystem()
        self.config = config or AttackConfig()
    
    def execute_attack(
        self,
        audio: Audio,
        model_name: str,
        prompt: Optional[str] = None
    ) -> AttackResult:
        """
        Execute a single attack.
        
        Args:
            audio: Audio sample to use
            model_name: Target model name
            prompt: Optional text prompt
        
        Returns:
            AttackResult with response and evaluation
        """
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        model = self.models[model_name]
        
        # Generate response with retries
        response = self._generate_with_retry(
            model, audio.data, audio.sample_rate, prompt
        )
        
        if response.error:
            # Return failed result
            return AttackResult(
                audio_id=audio.id,
                model=model_name,
                response="",
                is_success=False,
                judge_result=JudgeResult(
                    is_attack_success=False,
                    votes=[],
                    num_harmful_votes=0,
                    num_substantive_tokens=0,
                    confidence=0.0
                ),
                latency_ms=response.latency_ms,
                error=response.error
            )
        
        # Evaluate response
        original_query = audio.transcript or ""
        judge_result = self.judge_system.evaluate(response.text, original_query)
        
        return AttackResult(
            audio_id=audio.id,
            model=model_name,
            response=response.text,
            is_success=judge_result.is_attack_success,
            judge_result=judge_result,
            latency_ms=response.latency_ms
        )
    
    def _generate_with_retry(
        self,
        model: AudioLLM,
        audio_data,
        sample_rate: int,
        prompt: Optional[str]
    ) -> ModelResponse:
        """Generate with retry logic."""
        last_error = None
        
        for attempt in range(self.config.max_retries):
            try:
                response = model.generate(audio_data, sample_rate, prompt)
                
                if not response.error:
                    return response
                
                last_error = response.error
                
            except Exception as e:
                last_error = str(e)
            
            if attempt < self.config.max_retries - 1:
                time.sleep(self.config.retry_delay * (attempt + 1))
        
        return ModelResponse(
            text="",
            model=model.name,
            latency_ms=0,
            error=f"Failed after {self.config.max_retries} attempts: {last_error}"
        )
    
    def batch_attack(
        self,
        audios: List[Audio],
        model_name: str,
        prompt: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[AttackResult]:
        """
        Execute attacks on multiple audio samples.
        
        Args:
            audios: List of audio samples
            model_name: Target model name
            prompt: Optional text prompt
            progress_callback: Callback(current, total) for progress
        
        Returns:
            List of AttackResult
        """
        results = []
        total = len(audios)
        
        if self.config.parallel_workers > 1:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
                futures = {
                    executor.submit(self.execute_attack, audio, model_name, prompt): audio
                    for audio in audios
                }
                
                for i, future in enumerate(as_completed(futures)):
                    result = future.result()
                    results.append(result)
                    
                    if progress_callback:
                        progress_callback(i + 1, total)
        else:
            # Sequential execution
            for i, audio in enumerate(audios):
                result = self.execute_attack(audio, model_name, prompt)
                results.append(result)
                
                if progress_callback:
                    progress_callback(i + 1, total)
        
        return results
    
    def attack_all_models(
        self,
        audios: List[Audio],
        prompt: Optional[str] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> Dict[str, List[AttackResult]]:
        """
        Execute attacks against all models.
        
        Args:
            audios: List of audio samples
            prompt: Optional text prompt
            progress_callback: Callback(model, current, total)
        
        Returns:
            Dictionary of model_name -> List[AttackResult]
        """
        all_results = {}
        
        for model_name in self.models:
            logger.info(f"Attacking model: {model_name}")
            
            def model_progress(current, total):
                if progress_callback:
                    progress_callback(model_name, current, total)
            
            results = self.batch_attack(
                audios, model_name, prompt, model_progress
            )
            all_results[model_name] = results
        
        return all_results
    
    def get_attack_summary(
        self,
        results: List[AttackResult]
    ) -> Dict[str, Any]:
        """
        Get summary statistics for attack results.
        
        Args:
            results: List of attack results
        
        Returns:
            Summary dictionary
        """
        if not results:
            return {"total": 0, "success": 0, "asr": 0.0}
        
        total = len(results)
        success = sum(1 for r in results if r.is_success)
        errors = sum(1 for r in results if r.error)
        
        avg_latency = sum(r.latency_ms for r in results) / total
        
        return {
            "total": total,
            "success": success,
            "errors": errors,
            "asr": success / total,
            "avg_latency_ms": avg_latency
        }
