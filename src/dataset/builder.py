"""
Dataset builder for PJ-Break experiment reproduction.

Builds the AdvAudio-Prosody dataset by combining seed instructions
with TTS-generated prosody variations.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any
from pathlib import Path
import json
import logging
import time
from datetime import datetime
import numpy as np

from ..models_data import (
    Instruction, Audio, ProsodyCondition, HarmCategory,
    F0Features, TemporalFeatures, IntensityFeatures
)
from ..tts.engine import TTSEngine, get_tts_engine
from ..tts.generator import ProsodyGenerator, GenerationResult
from ..audio.processor import AudioProcessor
from ..audio.features import FeatureExtractor
from .loader import InstructionLoader, load_seed_instructions

logger = logging.getLogger(__name__)


@dataclass
class AudioSample:
    """A single audio sample in the dataset."""
    audio: Audio
    instruction: Instruction
    f0_features: Optional[F0Features] = None
    temporal_features: Optional[TemporalFeatures] = None
    intensity_features: Optional[IntensityFeatures] = None


@dataclass
class AdvAudioProsodyDataset:
    """
    The AdvAudio-Prosody dataset.
    
    Contains 600 audio samples (100 seeds × 6 prosody conditions).
    """
    samples: List[AudioSample] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __iter__(self):
        return iter(self.samples)
    
    def __getitem__(self, idx: int) -> AudioSample:
        return self.samples[idx]
    
    def get_by_prosody(self, prosody: ProsodyCondition) -> List[AudioSample]:
        """Get all samples with a specific prosody condition."""
        return [s for s in self.samples if s.audio.prosody == prosody]
    
    def get_by_category(self, category: HarmCategory) -> List[AudioSample]:
        """Get all samples with a specific harm category."""
        return [s for s in self.samples if s.instruction.category == category]
    
    def get_by_instruction(self, instruction_id: str) -> List[AudioSample]:
        """Get all prosody variations for an instruction."""
        return [s for s in self.samples if s.audio.instruction_id == instruction_id]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        stats = {
            "total_samples": len(self.samples),
            "by_prosody": {},
            "by_category": {},
            "total_duration_seconds": 0.0,
        }
        
        for prosody in ProsodyCondition:
            samples = self.get_by_prosody(prosody)
            stats["by_prosody"][prosody.value] = len(samples)
        
        for category in HarmCategory:
            samples = self.get_by_category(category)
            stats["by_category"][category.value] = len(samples)
        
        stats["total_duration_seconds"] = sum(s.audio.duration for s in self.samples)
        
        return stats


class DatasetBuilder:
    """
    Builds the AdvAudio-Prosody dataset.
    
    Integrates:
    - Seed instruction loading
    - TTS prosody generation
    - Audio processing
    - Feature extraction
    """
    
    # Use the standard 6 prosody conditions (excluding COMBINED)
    PROSODY_CONDITIONS = ProsodyCondition.all_conditions()
    
    def __init__(
        self,
        tts_engine: Optional[TTSEngine] = None,
        audio_processor: Optional[AudioProcessor] = None,
        feature_extractor: Optional[FeatureExtractor] = None,
        output_dir: Optional[str] = None
    ):
        """
        Initialize dataset builder.
        
        Args:
            tts_engine: TTS engine for synthesis
            audio_processor: Audio processor for normalization
            feature_extractor: Feature extractor for acoustic features
            output_dir: Directory to save dataset
        """
        self.tts_engine = tts_engine or get_tts_engine("auto")
        self.audio_processor = audio_processor or AudioProcessor()
        self.feature_extractor = feature_extractor or FeatureExtractor()
        self.output_dir = Path(output_dir) if output_dir else None
        
        self.prosody_generator = ProsodyGenerator(
            tts_engine=self.tts_engine,
            audio_processor=self.audio_processor
        )
    
    def build_dataset(
        self,
        instructions: Optional[List[Instruction]] = None,
        source: str = "synthetic",
        filepath: Optional[str] = None,
        num_seeds: int = 100,
        extract_features: bool = True,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> AdvAudioProsodyDataset:
        """
        Build the complete dataset.
        
        Args:
            instructions: Pre-loaded instructions (optional)
            source: Instruction source if not provided
            filepath: Path to instruction file
            num_seeds: Number of seed instructions
            extract_features: Whether to extract acoustic features
            progress_callback: Callback(current, total, message)
        
        Returns:
            AdvAudioProsodyDataset with all samples
        """
        # Load instructions if not provided
        if instructions is None:
            instructions = load_seed_instructions(
                source=source,
                filepath=filepath,
                count=num_seeds
            )
        
        logger.info(f"Building dataset with {len(instructions)} instructions")
        
        samples = []
        total = len(instructions)
        errors = []
        
        for i, instruction in enumerate(instructions):
            if progress_callback:
                progress_callback(i + 1, total, f"Processing: {instruction.id}")
            
            # Generate prosody variations
            result = self.prosody_generator.generate_prosody_variations(
                text=instruction.text,
                instruction_id=instruction.id,
                conditions=[p.value for p in self.PROSODY_CONDITIONS]
            )
            
            # Create audio samples for each variation
            for prosody_str, tts_result in result.variations.items():
                if not tts_result.success or len(tts_result.audio_data) == 0:
                    errors.append(f"{instruction.id}/{prosody_str}: {tts_result.error}")
                    continue
                
                prosody = ProsodyCondition(prosody_str)
                
                # Create Audio object
                audio = Audio(
                    id=f"{instruction.id}_{prosody_str}",
                    data=tts_result.audio_data,
                    sample_rate=tts_result.sample_rate,
                    instruction_id=instruction.id,
                    prosody=prosody,
                    duration=tts_result.duration
                )
                
                # Extract features if requested
                f0_feat = None
                temp_feat = None
                int_feat = None
                
                if extract_features:
                    try:
                        f0_feat = self.feature_extractor.extract_f0_features(
                            audio.data, audio.sample_rate
                        )
                        temp_feat = self.feature_extractor.extract_temporal_features(
                            audio.data, audio.sample_rate, instruction.text
                        )
                        int_feat = self.feature_extractor.extract_intensity_features(
                            audio.data, audio.sample_rate
                        )
                    except Exception as e:
                        logger.warning(f"Feature extraction failed for {audio.id}: {e}")
                
                sample = AudioSample(
                    audio=audio,
                    instruction=instruction,
                    f0_features=f0_feat,
                    temporal_features=temp_feat,
                    intensity_features=int_feat
                )
                samples.append(sample)
        
        # Create dataset
        dataset = AdvAudioProsodyDataset(
            samples=samples,
            metadata={
                "created_at": datetime.now().isoformat(),
                "num_seeds": len(instructions),
                "num_prosody_conditions": len(self.PROSODY_CONDITIONS),
                "expected_samples": len(instructions) * len(self.PROSODY_CONDITIONS),
                "actual_samples": len(samples),
                "errors": errors,
            }
        )
        
        logger.info(f"Dataset built: {len(samples)} samples")
        if errors:
            logger.warning(f"Encountered {len(errors)} errors during generation")
        
        # Save if output directory specified
        if self.output_dir:
            self.save_dataset(dataset)
        
        return dataset
    
    def save_dataset(
        self,
        dataset: AdvAudioProsodyDataset,
        save_audio: bool = True
    ) -> None:
        """
        Save dataset to disk.
        
        Args:
            dataset: Dataset to save
            save_audio: Whether to save audio files
        """
        if self.output_dir is None:
            raise ValueError("output_dir not specified")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(dataset.metadata, f, indent=2)
        
        # Save sample index
        index = []
        for sample in dataset.samples:
            entry = {
                "audio_id": sample.audio.id,
                "instruction_id": sample.instruction.id,
                "instruction_text": sample.instruction.text,
                "category": sample.instruction.category.value,
                "prosody": sample.audio.prosody.value,
                "duration": sample.audio.duration,
            }
            
            if sample.f0_features:
                entry["f0_mean"] = sample.f0_features.mean
                entry["f0_variance"] = sample.f0_features.variance
                entry["f0_range"] = sample.f0_features.range
            
            if sample.temporal_features:
                entry["speech_rate_wpm"] = sample.temporal_features.speech_rate_wpm
                entry["syllable_rate"] = sample.temporal_features.syllable_rate
            
            if sample.intensity_features:
                entry["rms_db"] = sample.intensity_features.rms_db
                entry["zero_crossing_rate"] = sample.intensity_features.zero_crossing_rate
                entry["spectral_tilt"] = sample.intensity_features.spectral_tilt
            
            index.append(entry)
        
        index_path = self.output_dir / "index.json"
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2)
        
        # Save audio files
        if save_audio:
            audio_dir = self.output_dir / "audio"
            audio_dir.mkdir(exist_ok=True)
            
            try:
                import soundfile as sf
                
                for sample in dataset.samples:
                    audio_path = audio_dir / f"{sample.audio.id}.wav"
                    sf.write(
                        str(audio_path),
                        sample.audio.data,
                        sample.audio.sample_rate
                    )
            except ImportError:
                logger.warning("soundfile not installed, saving as numpy arrays")
                for sample in dataset.samples:
                    audio_path = audio_dir / f"{sample.audio.id}.npy"
                    np.save(str(audio_path), sample.audio.data)
        
        logger.info(f"Dataset saved to {self.output_dir}")
    
    def load_dataset(self, dataset_dir: str) -> AdvAudioProsodyDataset:
        """
        Load dataset from disk.
        
        Args:
            dataset_dir: Directory containing saved dataset
        
        Returns:
            Loaded AdvAudioProsodyDataset
        """
        dataset_path = Path(dataset_dir)
        
        # Load metadata
        metadata_path = dataset_path / "metadata.json"
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Load index
        index_path = dataset_path / "index.json"
        with open(index_path, 'r', encoding='utf-8') as f:
            index = json.load(f)
        
        # Load audio and reconstruct samples
        audio_dir = dataset_path / "audio"
        samples = []
        
        for entry in index:
            # Load audio
            wav_path = audio_dir / f"{entry['audio_id']}.wav"
            npy_path = audio_dir / f"{entry['audio_id']}.npy"
            
            if wav_path.exists():
                try:
                    import soundfile as sf
                    audio_data, sample_rate = sf.read(str(wav_path))
                    audio_data = audio_data.astype(np.float32)
                except ImportError:
                    logger.warning("soundfile not installed, trying numpy")
                    audio_data = np.load(str(npy_path))
                    sample_rate = 16000
            elif npy_path.exists():
                audio_data = np.load(str(npy_path))
                sample_rate = 16000
            else:
                logger.warning(f"Audio file not found: {entry['audio_id']}")
                continue
            
            # Reconstruct objects
            instruction = Instruction(
                id=entry['instruction_id'],
                text=entry['instruction_text'],
                category=HarmCategory(entry['category'])
            )
            
            audio = Audio(
                id=entry['audio_id'],
                data=audio_data,
                sample_rate=sample_rate,
                instruction_id=entry['instruction_id'],
                prosody=ProsodyCondition(entry['prosody']),
                duration=entry['duration']
            )
            
            # Reconstruct features if present
            f0_feat = None
            if 'f0_mean' in entry:
                f0_feat = F0Features(
                    mean=entry['f0_mean'],
                    variance=entry['f0_variance'],
                    range=entry['f0_range']
                )
            
            temp_feat = None
            if 'speech_rate_wpm' in entry:
                temp_feat = TemporalFeatures(
                    speech_rate_wpm=entry['speech_rate_wpm'],
                    syllable_rate=entry['syllable_rate'],
                    duration=entry['duration']
                )
            
            int_feat = None
            if 'rms_db' in entry:
                int_feat = IntensityFeatures(
                    rms_db=entry['rms_db'],
                    zero_crossing_rate=entry['zero_crossing_rate'],
                    spectral_tilt=entry['spectral_tilt']
                )
            
            sample = AudioSample(
                audio=audio,
                instruction=instruction,
                f0_features=f0_feat,
                temporal_features=temp_feat,
                intensity_features=int_feat
            )
            samples.append(sample)
        
        return AdvAudioProsodyDataset(samples=samples, metadata=metadata)
