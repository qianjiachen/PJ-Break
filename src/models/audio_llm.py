"""
Audio LLM interfaces for PJ-Break experiment reproduction.

Implements adapters for the 4 target models from the paper:
- Qwen2-Audio (transformers)
- GPT-4o (OpenAI API)
- Gemini 2.0 Flash (Google API)
- SALMONN (custom)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import numpy as np
import os
import time
import logging
import base64
import io

logger = logging.getLogger(__name__)


@dataclass
class ModelResponse:
    """Response from an Audio LLM."""
    text: str
    model: str
    latency_ms: float
    tokens_generated: int = 0
    logits: Optional[np.ndarray] = None
    error: Optional[str] = None


class AudioLLM(ABC):
    """Abstract base class for Audio LLMs."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Model name identifier."""
        pass
    
    @abstractmethod
    def generate(
        self,
        audio: np.ndarray,
        sample_rate: int,
        prompt: Optional[str] = None
    ) -> ModelResponse:
        """
        Generate response from audio input.
        
        Args:
            audio: Audio waveform as numpy array
            sample_rate: Audio sample rate
            prompt: Optional text prompt
        
        Returns:
            ModelResponse with generated text
        """
        pass
    
    def get_logits(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> Optional[np.ndarray]:
        """
        Get output logits (if supported).
        
        Args:
            audio: Audio waveform
            sample_rate: Sample rate
        
        Returns:
            Logits array or None if not supported
        """
        return None
    
    def _audio_to_base64(self, audio: np.ndarray, sample_rate: int) -> str:
        """Convert audio to base64 encoded WAV."""
        try:
            import soundfile as sf
            
            buffer = io.BytesIO()
            sf.write(buffer, audio, sample_rate, format='WAV')
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode('utf-8')
        except ImportError:
            # Fallback: simple WAV encoding
            return self._simple_wav_base64(audio, sample_rate)
    
    def _simple_wav_base64(self, audio: np.ndarray, sample_rate: int) -> str:
        """Simple WAV encoding without soundfile."""
        import struct
        
        # Normalize to int16
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # WAV header
        num_samples = len(audio_int16)
        bytes_per_sample = 2
        num_channels = 1
        
        header = struct.pack(
            '<4sI4s4sIHHIIHH4sI',
            b'RIFF',
            36 + num_samples * bytes_per_sample,
            b'WAVE',
            b'fmt ',
            16,  # Subchunk1Size
            1,   # AudioFormat (PCM)
            num_channels,
            sample_rate,
            sample_rate * num_channels * bytes_per_sample,
            num_channels * bytes_per_sample,
            bytes_per_sample * 8,
            b'data',
            num_samples * bytes_per_sample
        )
        
        wav_data = header + audio_int16.tobytes()
        return base64.b64encode(wav_data).decode('utf-8')


class Qwen2AudioModel(AudioLLM):
    """
    Qwen2-Audio model adapter.
    
    Uses Hugging Face transformers for local inference.
    """
    
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2-Audio-7B-Instruct",
        device: str = "auto",
        torch_dtype: str = "auto"
    ):
        """
        Initialize Qwen2-Audio model.
        
        Args:
            model_id: Hugging Face model ID
            device: Device to use ("auto", "cuda", "cpu")
            torch_dtype: Torch dtype ("auto", "float16", "bfloat16")
        """
        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype
        self._model = None
        self._processor = None
    
    @property
    def name(self) -> str:
        return "qwen2-audio"
    
    def _load_model(self):
        """Lazy load model."""
        if self._model is not None:
            return
        
        try:
            import torch
            from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
            
            dtype_map = {
                "auto": "auto",
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }
            
            self._processor = AutoProcessor.from_pretrained(self.model_id)
            self._model = Qwen2AudioForConditionalGeneration.from_pretrained(
                self.model_id,
                device_map=self.device,
                torch_dtype=dtype_map.get(self.torch_dtype, "auto")
            )
            
            logger.info(f"Loaded Qwen2-Audio model: {self.model_id}")
        
        except ImportError as e:
            logger.error(f"Failed to import transformers: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load Qwen2-Audio: {e}")
            raise
    
    def generate(
        self,
        audio: np.ndarray,
        sample_rate: int,
        prompt: Optional[str] = None
    ) -> ModelResponse:
        """Generate response using Qwen2-Audio."""
        start_time = time.time()
        
        try:
            self._load_model()
            
            import torch
            
            # Prepare conversation
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": audio, "sample_rate": sample_rate},
                    ]
                }
            ]
            
            if prompt:
                conversation[0]["content"].append({"type": "text", "text": prompt})
            
            # Process inputs
            text = self._processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False
            )
            
            inputs = self._processor(
                text=text,
                audios=[audio],
                sampling_rate=sample_rate,
                return_tensors="pt"
            )
            inputs = inputs.to(self._model.device)
            
            # Generate
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False
                )
            
            # Decode
            response_text = self._processor.batch_decode(
                outputs[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )[0]
            
            latency = (time.time() - start_time) * 1000
            
            return ModelResponse(
                text=response_text,
                model=self.name,
                latency_ms=latency,
                tokens_generated=outputs.shape[1] - inputs.input_ids.shape[1]
            )
        
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            logger.error(f"Qwen2-Audio generation failed: {e}")
            return ModelResponse(
                text="",
                model=self.name,
                latency_ms=latency,
                error=str(e)
            )
    
    def get_logits(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> Optional[np.ndarray]:
        """Get output logits for Pro-Guard."""
        try:
            self._load_model()
            
            import torch
            
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": audio, "sample_rate": sample_rate},
                    ]
                }
            ]
            
            text = self._processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False
            )
            
            inputs = self._processor(
                text=text,
                audios=[audio],
                sampling_rate=sample_rate,
                return_tensors="pt"
            )
            inputs = inputs.to(self._model.device)
            
            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits[:, -1, :].cpu().numpy()
            
            return logits[0]
        
        except Exception as e:
            logger.error(f"Failed to get logits: {e}")
            return None


class GPT4oModel(AudioLLM):
    """
    GPT-4o model adapter.
    
    Uses OpenAI API with audio input support.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-audio-preview"
    ):
        """
        Initialize GPT-4o model.
        
        Args:
            api_key: OpenAI API key (or from env OPENAI_API_KEY)
            model: Model identifier
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self._client = None
    
    @property
    def name(self) -> str:
        return "gpt-4o"
    
    def _get_client(self):
        """Lazy initialize OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                logger.error("openai package not installed")
                raise
        return self._client
    
    def generate(
        self,
        audio: np.ndarray,
        sample_rate: int,
        prompt: Optional[str] = None
    ) -> ModelResponse:
        """Generate response using GPT-4o."""
        start_time = time.time()
        
        try:
            client = self._get_client()
            
            # Convert audio to base64
            audio_b64 = self._audio_to_base64(audio, sample_rate)
            
            # Build message
            content = [
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": audio_b64,
                        "format": "wav"
                    }
                }
            ]
            
            if prompt:
                content.append({"type": "text", "text": prompt})
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                max_tokens=512
            )
            
            latency = (time.time() - start_time) * 1000
            response_text = response.choices[0].message.content or ""
            
            return ModelResponse(
                text=response_text,
                model=self.name,
                latency_ms=latency,
                tokens_generated=response.usage.completion_tokens if response.usage else 0
            )
        
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            logger.error(f"GPT-4o generation failed: {e}")
            return ModelResponse(
                text="",
                model=self.name,
                latency_ms=latency,
                error=str(e)
            )


class GeminiModel(AudioLLM):
    """
    Gemini 2.0 Flash model adapter.
    
    Uses Google Generative AI API.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.0-flash-exp"
    ):
        """
        Initialize Gemini model.
        
        Args:
            api_key: Google API key (or from env GOOGLE_API_KEY)
            model: Model identifier
        """
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.model = model
        self._client = None
    
    @property
    def name(self) -> str:
        return "gemini"
    
    def _get_client(self):
        """Lazy initialize Gemini client."""
        if self._client is None:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._client = genai.GenerativeModel(self.model)
            except ImportError:
                logger.error("google-generativeai package not installed")
                raise
        return self._client
    
    def generate(
        self,
        audio: np.ndarray,
        sample_rate: int,
        prompt: Optional[str] = None
    ) -> ModelResponse:
        """Generate response using Gemini."""
        start_time = time.time()
        
        try:
            client = self._get_client()
            
            import google.generativeai as genai
            
            # Convert audio to bytes
            audio_b64 = self._audio_to_base64(audio, sample_rate)
            audio_bytes = base64.b64decode(audio_b64)
            
            # Create audio part
            audio_part = {
                "mime_type": "audio/wav",
                "data": audio_bytes
            }
            
            content = [audio_part]
            if prompt:
                content.append(prompt)
            
            response = client.generate_content(content)
            
            latency = (time.time() - start_time) * 1000
            response_text = response.text if response.text else ""
            
            return ModelResponse(
                text=response_text,
                model=self.name,
                latency_ms=latency
            )
        
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            logger.error(f"Gemini generation failed: {e}")
            return ModelResponse(
                text="",
                model=self.name,
                latency_ms=latency,
                error=str(e)
            )


class SALMONNModel(AudioLLM):
    """
    SALMONN model adapter.
    
    Speech Audio Language Music Open Neural Network.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda"
    ):
        """
        Initialize SALMONN model.
        
        Args:
            model_path: Path to model weights
            device: Device to use
        """
        self.model_path = model_path
        self.device = device
        self._model = None
    
    @property
    def name(self) -> str:
        return "salmonn"
    
    def _load_model(self):
        """Lazy load SALMONN model."""
        if self._model is not None:
            return
        
        # SALMONN requires custom loading
        # This is a placeholder - actual implementation depends on SALMONN repo
        logger.warning("SALMONN model loading not fully implemented")
        self._model = "placeholder"
    
    def generate(
        self,
        audio: np.ndarray,
        sample_rate: int,
        prompt: Optional[str] = None
    ) -> ModelResponse:
        """Generate response using SALMONN."""
        start_time = time.time()
        
        try:
            self._load_model()
            
            # Placeholder implementation
            # Actual SALMONN inference would go here
            
            latency = (time.time() - start_time) * 1000
            
            return ModelResponse(
                text="[SALMONN response placeholder]",
                model=self.name,
                latency_ms=latency,
                error="SALMONN not fully implemented"
            )
        
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            logger.error(f"SALMONN generation failed: {e}")
            return ModelResponse(
                text="",
                model=self.name,
                latency_ms=latency,
                error=str(e)
            )


class MockAudioLLM(AudioLLM):
    """Mock Audio LLM for testing."""
    
    def __init__(
        self,
        name: str = "mock",
        default_response: str = "This is a mock response.",
        latency_ms: float = 100.0
    ):
        self._name = name
        self.default_response = default_response
        self.latency_ms = latency_ms
    
    @property
    def name(self) -> str:
        return self._name
    
    def generate(
        self,
        audio: np.ndarray,
        sample_rate: int,
        prompt: Optional[str] = None
    ) -> ModelResponse:
        """Return mock response."""
        time.sleep(self.latency_ms / 1000)
        
        return ModelResponse(
            text=self.default_response,
            model=self.name,
            latency_ms=self.latency_ms,
            tokens_generated=len(self.default_response.split())
        )


def get_model(model_name: str, **kwargs) -> AudioLLM:
    """
    Factory function to get model by name.
    
    Args:
        model_name: Model identifier
        **kwargs: Model-specific arguments
    
    Returns:
        AudioLLM instance
    """
    models = {
        "qwen2-audio": Qwen2AudioModel,
        "gpt-4o": GPT4oModel,
        "gemini": GeminiModel,
        "salmonn": SALMONNModel,
        "mock": MockAudioLLM,
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")
    
    return models[model_name](**kwargs)
