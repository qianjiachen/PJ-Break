"""
Core data models for PJ-Break experiment reproduction.

This module defines all data structures used throughout the experiment,
including instructions, audio samples, attack results, and defense results.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any, Tuple
import numpy as np
from pathlib import Path
import json
from datetime import datetime


class HarmCategory(Enum):
    """Categories of harmful content as defined in the paper."""
    VIOLENCE = "violence"
    ILLEGAL_ACTIVITIES = "illegal_activities"
    HATE_SPEECH = "hate_speech"
    SELF_HARM = "self_harm"
    MISINFORMATION = "misinformation"
    PRIVACY_VIOLATION = "privacy_violation"
    
    @classmethod
    def from_string(cls, s: str) -> "HarmCategory":
        """Convert string to HarmCategory."""
        mapping = {
            "violence": cls.VIOLENCE,
            "illegal": cls.ILLEGAL_ACTIVITIES,
            "illegal_activities": cls.ILLEGAL_ACTIVITIES,
            "hate": cls.HATE_SPEECH,
            "hate_speech": cls.HATE_SPEECH,
            "self_harm": cls.SELF_HARM,
            "selfharm": cls.SELF_HARM,
            "misinformation": cls.MISINFORMATION,
            "misinfo": cls.MISINFORMATION,
            "privacy": cls.PRIVACY_VIOLATION,
            "privacy_violation": cls.PRIVACY_VIOLATION,
        }
        return mapping.get(s.lower(), cls.VIOLENCE)


class ProsodyCondition(Enum):
    """Prosody conditions for audio generation."""
    NEUTRAL = "neutral"
    PANIC = "panic"
    ANGER = "anger"
    COMMANDING = "commanding"
    FAST = "fast"
    WHISPER = "whisper"
    COMBINED = "combined"  # Panic + Fast
    
    @classmethod
    def all_conditions(cls) -> List["ProsodyCondition"]:
        """Return all standard prosody conditions (excluding combined)."""
        return [cls.NEUTRAL, cls.PANIC, cls.ANGER, cls.COMMANDING, cls.FAST, cls.WHISPER]


@dataclass
class Instruction:
    """A harmful instruction/query for testing."""
    id: str
    text: str
    category: HarmCategory
    source: str = "advbench"  # advbench, harmbench, custom
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "text": self.text,
            "category": self.category.value,
            "source": self.source,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Instruction":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            text=data["text"],
            category=HarmCategory.from_string(data["category"]),
            source=data.get("source", "advbench"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Audio:
    """An audio sample with metadata."""
    id: str
    data: np.ndarray
    sample_rate: int
    instruction_id: str
    prosody: ProsodyCondition
    duration: float
    file_path: Optional[str] = None
    transcript: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def num_samples(self) -> int:
        """Number of audio samples."""
        return len(self.data)
    
    @property
    def channels(self) -> int:
        """Number of audio channels."""
        if self.data.ndim == 1:
            return 1
        return self.data.shape[0]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without audio data)."""
        return {
            "id": self.id,
            "sample_rate": self.sample_rate,
            "instruction_id": self.instruction_id,
            "prosody": self.prosody.value,
            "duration": self.duration,
            "file_path": self.file_path,
            "transcript": self.transcript,
            "metadata": self.metadata,
        }


@dataclass
class F0Features:
    """Fundamental frequency (F0) features."""
    mean: float
    variance: float
    range: float
    min: float = 0.0
    max: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "mean": self.mean,
            "variance": self.variance,
            "range": self.range,
            "min": self.min,
            "max": self.max,
        }


@dataclass
class TemporalFeatures:
    """Temporal/timing features."""
    speech_rate_wpm: float  # Words per minute
    syllable_rate: float    # Syllables per second
    duration: float         # Total duration in seconds
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "speech_rate_wpm": self.speech_rate_wpm,
            "syllable_rate": self.syllable_rate,
            "duration": self.duration,
        }


@dataclass
class IntensityFeatures:
    """Intensity and spectral features."""
    rms_db: float           # RMS intensity in dBFS
    zero_crossing_rate: float
    spectral_tilt: float    # H1-H2 proxy
    lufs: float = 0.0       # Integrated loudness
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "rms_db": self.rms_db,
            "zero_crossing_rate": self.zero_crossing_rate,
            "spectral_tilt": self.spectral_tilt,
            "lufs": self.lufs,
        }


@dataclass
class AudioFeatures:
    """Complete acoustic features for an audio sample."""
    audio_id: str
    f0: F0Features
    temporal: TemporalFeatures
    intensity: IntensityFeatures
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "audio_id": self.audio_id,
            "f0": self.f0.to_dict(),
            "temporal": self.temporal.to_dict(),
            "intensity": self.intensity.to_dict(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AudioFeatures":
        return cls(
            audio_id=data["audio_id"],
            f0=F0Features(**data["f0"]),
            temporal=TemporalFeatures(**data["temporal"]),
            intensity=IntensityFeatures(**data["intensity"]),
        )


@dataclass
class JudgeVote:
    """A single judge's evaluation."""
    judge_name: str
    is_harmful: bool
    confidence: float = 1.0
    reasoning: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "judge_name": self.judge_name,
            "is_harmful": self.is_harmful,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
        }


@dataclass
class JudgeResult:
    """Aggregated result from multi-judge evaluation."""
    is_attack_success: bool
    votes: List[JudgeVote]
    num_harmful_votes: int
    num_substantive_tokens: int
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_attack_success": self.is_attack_success,
            "votes": [v.to_dict() for v in self.votes],
            "num_harmful_votes": self.num_harmful_votes,
            "num_substantive_tokens": self.num_substantive_tokens,
            "confidence": self.confidence,
        }


@dataclass
class AttackResult:
    """Result of a single attack attempt."""
    audio_id: str
    model: str
    response: str
    is_success: bool
    judge_result: JudgeResult
    latency_ms: float = 0.0
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "audio_id": self.audio_id,
            "model": self.model,
            "response": self.response,
            "is_success": self.is_success,
            "judge_result": self.judge_result.to_dict(),
            "latency_ms": self.latency_ms,
            "error": self.error,
            "timestamp": self.timestamp,
        }


@dataclass
class DefenseResult:
    """Result of defense evaluation."""
    audio_id: str
    is_blocked: bool
    aggregate_score: float
    text_score: float
    prosody_score: float
    refusal_score: float
    latency_ms: float
    defense_name: str = "proguard"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "audio_id": self.audio_id,
            "is_blocked": self.is_blocked,
            "aggregate_score": self.aggregate_score,
            "text_score": self.text_score,
            "prosody_score": self.prosody_score,
            "refusal_score": self.refusal_score,
            "latency_ms": self.latency_ms,
            "defense_name": self.defense_name,
        }


@dataclass
class ASRMetric:
    """Attack Success Rate metric with confidence interval."""
    asr: float
    ci_lower: float
    ci_upper: float
    n: int
    
    def __str__(self) -> str:
        return f"{self.asr*100:.1f}% [{self.ci_lower*100:.1f}, {self.ci_upper*100:.1f}]"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "asr": self.asr,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "n": self.n,
        }


@dataclass
class KappaResult:
    """Inter-rater agreement metric."""
    kappa: float
    ci_lower: float
    ci_upper: float
    
    def __str__(self) -> str:
        return f"κ={self.kappa:.2f} [{self.ci_lower:.2f}, {self.ci_upper:.2f}]"
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "kappa": self.kappa,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
        }


@dataclass
class ProbeResult:
    """Result of latent space probing."""
    layer: int
    head: Optional[int]
    activation: np.ndarray
    cosine_to_refusal: float
    audio_id: str
    prosody: ProsodyCondition
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer,
            "head": self.head,
            "cosine_to_refusal": self.cosine_to_refusal,
            "audio_id": self.audio_id,
            "prosody": self.prosody.value,
        }


@dataclass
class PatchResult:
    """Result of activation patching."""
    response: str
    layer: int
    head: int
    interpolation: float
    is_control: bool = False
    asr_after: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "response": self.response,
            "layer": self.layer,
            "head": self.head,
            "interpolation": self.interpolation,
            "is_control": self.is_control,
            "asr_after": self.asr_after,
        }


@dataclass
class AdvAudioProsodyDataset:
    """The complete AdvAudio-Prosody dataset."""
    instructions: List[Instruction]
    audio_samples: Dict[str, Audio]  # audio_id -> Audio
    features: Dict[str, AudioFeatures]  # audio_id -> AudioFeatures
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def num_instructions(self) -> int:
        return len(self.instructions)
    
    @property
    def num_audio_samples(self) -> int:
        return len(self.audio_samples)
    
    def get_samples_by_prosody(self, prosody: ProsodyCondition) -> List[Audio]:
        """Get all audio samples with a specific prosody condition."""
        return [a for a in self.audio_samples.values() if a.prosody == prosody]
    
    def get_samples_by_category(self, category: HarmCategory) -> List[Audio]:
        """Get all audio samples for a specific harm category."""
        instruction_ids = {i.id for i in self.instructions if i.category == category}
        return [a for a in self.audio_samples.values() if a.instruction_id in instruction_ids]
    
    def get_audio_id(self, instruction_id: str, prosody: ProsodyCondition) -> str:
        """Generate audio ID from instruction ID and prosody."""
        return f"{instruction_id}_{prosody.value}"
    
    def save(self, output_dir: str) -> None:
        """Save dataset metadata to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save instructions
        instructions_data = [i.to_dict() for i in self.instructions]
        with open(output_path / "instructions.json", 'w') as f:
            json.dump(instructions_data, f, indent=2)
        
        # Save audio metadata (not the actual audio data)
        audio_data = {k: v.to_dict() for k, v in self.audio_samples.items()}
        with open(output_path / "audio_metadata.json", 'w') as f:
            json.dump(audio_data, f, indent=2)
        
        # Save features
        features_data = {k: v.to_dict() for k, v in self.features.items()}
        with open(output_path / "features.json", 'w') as f:
            json.dump(features_data, f, indent=2)
        
        # Save dataset metadata
        with open(output_path / "dataset_metadata.json", 'w') as f:
            json.dump({
                "num_instructions": self.num_instructions,
                "num_audio_samples": self.num_audio_samples,
                **self.metadata
            }, f, indent=2)
    
    @classmethod
    def load(cls, input_dir: str) -> "AdvAudioProsodyDataset":
        """Load dataset metadata from disk."""
        input_path = Path(input_dir)
        
        # Load instructions
        with open(input_path / "instructions.json", 'r') as f:
            instructions_data = json.load(f)
        instructions = [Instruction.from_dict(d) for d in instructions_data]
        
        # Load features
        with open(input_path / "features.json", 'r') as f:
            features_data = json.load(f)
        features = {k: AudioFeatures.from_dict(v) for k, v in features_data.items()}
        
        # Load metadata
        with open(input_path / "dataset_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        return cls(
            instructions=instructions,
            audio_samples={},  # Audio data loaded separately
            features=features,
            metadata=metadata,
        )
