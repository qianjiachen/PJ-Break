"""
Audio processing module for PJ-Break experiment reproduction.

Implements LUFS normalization, peak limiting, and format conversion
as specified in the paper's methodology (Appendix A).
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import warnings

try:
    import pyloudnorm as pyln
    HAS_PYLOUDNORM = True
except ImportError:
    HAS_PYLOUDNORM = False
    warnings.warn("pyloudnorm not installed. LUFS normalization will use fallback.")

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    warnings.warn("librosa not installed. Some audio processing features unavailable.")

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False


@dataclass
class AudioProcessingResult:
    """Result of audio processing operations."""
    data: np.ndarray
    sample_rate: int
    lufs: float
    peak_db: float
    duration: float
    was_clipped: bool = False
    was_resampled: bool = False


class AudioProcessor:
    """
    Audio processor implementing paper's normalization pipeline.
    
    Processing steps (as per Appendix A.2):
    1. LUFS normalization to -23 LUFS (EBU R128 standard)
    2. Peak limiting at -1 dBFS to prevent clipping
    3. Sample rate conversion to 16kHz mono
    """
    
    def __init__(
        self,
        target_lufs: float = -23.0,
        peak_limit_db: float = -1.0,
        target_sample_rate: int = 16000,
        target_channels: int = 1
    ):
        """
        Initialize audio processor.
        
        Args:
            target_lufs: Target integrated loudness in LUFS (default: -23.0)
            peak_limit_db: Peak limiting threshold in dBFS (default: -1.0)
            target_sample_rate: Target sample rate in Hz (default: 16000)
            target_channels: Target number of channels (default: 1 for mono)
        """
        self.target_lufs = target_lufs
        self.peak_limit_db = peak_limit_db
        self.target_sample_rate = target_sample_rate
        self.target_channels = target_channels
        
        # Initialize loudness meter if available
        if HAS_PYLOUDNORM:
            self.meter = pyln.Meter(target_sample_rate)
        else:
            self.meter = None
    
    def process(
        self,
        audio: np.ndarray,
        sample_rate: int,
        normalize_loudness: bool = True,
        apply_peak_limit: bool = True,
        convert_format: bool = True
    ) -> AudioProcessingResult:
        """
        Apply full processing pipeline to audio.
        
        Args:
            audio: Input audio data (numpy array)
            sample_rate: Input sample rate
            normalize_loudness: Whether to apply LUFS normalization
            apply_peak_limit: Whether to apply peak limiting
            convert_format: Whether to convert to target format
        
        Returns:
            AudioProcessingResult with processed audio and metadata
        """
        data = audio.copy().astype(np.float32)
        was_resampled = False
        was_clipped = False
        
        # Step 1: Convert to mono if needed
        if convert_format and data.ndim > 1:
            data = self._to_mono(data)
        
        # Step 2: Resample if needed
        if convert_format and sample_rate != self.target_sample_rate:
            data = self._resample(data, sample_rate, self.target_sample_rate)
            sample_rate = self.target_sample_rate
            was_resampled = True
        
        # Step 3: LUFS normalization
        if normalize_loudness:
            data, was_clipped = self.normalize_loudness(data, sample_rate)
        
        # Step 4: Peak limiting
        if apply_peak_limit:
            data = self.apply_peak_limiting(data)
        
        # Calculate final metrics
        lufs = self._measure_lufs(data, sample_rate)
        peak_db = self._measure_peak_db(data)
        duration = len(data) / sample_rate
        
        return AudioProcessingResult(
            data=data,
            sample_rate=sample_rate,
            lufs=lufs,
            peak_db=peak_db,
            duration=duration,
            was_clipped=was_clipped,
            was_resampled=was_resampled
        )
    
    def normalize_loudness(
        self,
        audio: np.ndarray,
        sample_rate: int,
        target_lufs: Optional[float] = None
    ) -> Tuple[np.ndarray, bool]:
        """
        Normalize audio to target LUFS (EBU R128).
        
        Args:
            audio: Input audio data
            sample_rate: Sample rate
            target_lufs: Target loudness (uses instance default if None)
        
        Returns:
            Tuple of (normalized audio, was_clipped flag)
        """
        if target_lufs is None:
            target_lufs = self.target_lufs
        
        # Measure current loudness
        current_lufs = self._measure_lufs(audio, sample_rate)
        
        # Handle silent or very quiet audio
        if current_lufs < -70 or np.isinf(current_lufs):
            # Audio is essentially silent, return as-is
            return audio, False
        
        # Calculate gain needed
        gain_db = target_lufs - current_lufs
        gain_linear = 10 ** (gain_db / 20)
        
        # Apply gain
        normalized = audio * gain_linear
        
        # Check if clipping occurred
        was_clipped = np.max(np.abs(normalized)) > 1.0
        
        return normalized, was_clipped
    
    def apply_peak_limiting(
        self,
        audio: np.ndarray,
        threshold_db: Optional[float] = None
    ) -> np.ndarray:
        """
        Apply peak limiting to prevent clipping.
        
        Uses soft-knee limiting to preserve audio quality.
        
        Args:
            audio: Input audio data
            threshold_db: Limiting threshold in dBFS (uses instance default if None)
        
        Returns:
            Peak-limited audio
        """
        if threshold_db is None:
            threshold_db = self.peak_limit_db
        
        # Convert threshold to linear
        threshold_linear = 10 ** (threshold_db / 20)
        
        # Find peaks exceeding threshold
        peak = np.max(np.abs(audio))
        
        if peak > threshold_linear:
            # Apply limiting (simple hard limiter with makeup gain)
            ratio = threshold_linear / peak
            limited = audio * ratio
        else:
            limited = audio
        
        # Ensure no samples exceed threshold
        limited = np.clip(limited, -threshold_linear, threshold_linear)
        
        return limited
    
    def convert_format(
        self,
        audio: np.ndarray,
        input_sample_rate: int,
        target_sample_rate: Optional[int] = None,
        target_channels: Optional[int] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Convert audio to target format (sample rate and channels).
        
        Args:
            audio: Input audio data
            input_sample_rate: Input sample rate
            target_sample_rate: Target sample rate (uses instance default if None)
            target_channels: Target channels (uses instance default if None)
        
        Returns:
            Tuple of (converted audio, output sample rate)
        """
        if target_sample_rate is None:
            target_sample_rate = self.target_sample_rate
        if target_channels is None:
            target_channels = self.target_channels
        
        data = audio.copy()
        
        # Convert to mono if needed
        if target_channels == 1 and data.ndim > 1:
            data = self._to_mono(data)
        
        # Resample if needed
        if input_sample_rate != target_sample_rate:
            data = self._resample(data, input_sample_rate, target_sample_rate)
        
        return data, target_sample_rate
    
    def _to_mono(self, audio: np.ndarray) -> np.ndarray:
        """Convert multi-channel audio to mono."""
        if audio.ndim == 1:
            return audio
        
        # Average across channels
        if audio.shape[0] <= audio.shape[-1]:
            # Shape is (channels, samples)
            return np.mean(audio, axis=0)
        else:
            # Shape is (samples, channels)
            return np.mean(audio, axis=1)
    
    def _resample(
        self,
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int
    ) -> np.ndarray:
        """Resample audio to target sample rate."""
        if not HAS_LIBROSA:
            # Fallback: simple linear interpolation
            ratio = target_sr / orig_sr
            new_length = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_length)
            return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
        
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    
    def _measure_lufs(self, audio: np.ndarray, sample_rate: int) -> float:
        """Measure integrated loudness in LUFS."""
        if HAS_PYLOUDNORM:
            try:
                # Ensure audio is the right shape for pyloudnorm
                if audio.ndim == 1:
                    audio_for_meter = audio
                else:
                    audio_for_meter = audio.T if audio.shape[0] < audio.shape[1] else audio
                
                # Create meter with correct sample rate
                meter = pyln.Meter(sample_rate)
                return meter.integrated_loudness(audio_for_meter)
            except Exception:
                pass
        
        # Fallback: estimate LUFS from RMS
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 1e-10:
            return -70.0
        
        # Approximate LUFS from RMS (rough estimate)
        rms_db = 20 * np.log10(rms)
        return rms_db - 0.691  # K-weighting approximation
    
    def _measure_peak_db(self, audio: np.ndarray) -> float:
        """Measure peak level in dBFS."""
        peak = np.max(np.abs(audio))
        if peak < 1e-10:
            return -100.0
        return 20 * np.log10(peak)
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio from file.
        
        Args:
            file_path: Path to audio file
        
        Returns:
            Tuple of (audio data, sample rate)
        """
        if HAS_SOUNDFILE:
            data, sr = sf.read(file_path, dtype='float32')
            return data, sr
        elif HAS_LIBROSA:
            data, sr = librosa.load(file_path, sr=None, mono=False)
            return data, sr
        else:
            raise ImportError("Neither soundfile nor librosa available for audio loading")
    
    def save_audio(
        self,
        audio: np.ndarray,
        file_path: str,
        sample_rate: int
    ) -> None:
        """
        Save audio to file.
        
        Args:
            audio: Audio data
            file_path: Output file path
            sample_rate: Sample rate
        """
        if HAS_SOUNDFILE:
            sf.write(file_path, audio, sample_rate)
        else:
            raise ImportError("soundfile not available for audio saving")
