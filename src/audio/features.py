"""
Acoustic feature extraction module for PJ-Break experiment reproduction.

Implements prosodic feature extraction as specified in the paper's methodology
(Appendix A.1): F0 statistics, speech rate, RMS intensity, spectral features.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
import warnings

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

# Try to import parselmouth for more accurate F0 extraction
try:
    import parselmouth
    from parselmouth.praat import call
    HAS_PARSELMOUTH = True
except ImportError:
    HAS_PARSELMOUTH = False


@dataclass
class F0Features:
    """Fundamental frequency (F0) features."""
    mean: float
    variance: float
    range: float
    min: float = 0.0
    max: float = 0.0
    
    def to_dict(self) -> dict:
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
    
    def to_dict(self) -> dict:
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
    
    def to_dict(self) -> dict:
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
    
    def to_dict(self) -> dict:
        return {
            "audio_id": self.audio_id,
            "f0": self.f0.to_dict(),
            "temporal": self.temporal.to_dict(),
            "intensity": self.intensity.to_dict(),
        }


class FeatureExtractor:
    """
    Acoustic feature extractor implementing paper's prosodic measures.
    
    Features extracted (as per Appendix A.1):
    - F0 mean/variance (Hz)
    - Speech rate (words/min) and syllable rate (syllables/sec)
    - RMS intensity (dBFS)
    - Zero-crossing rate
    - Spectral tilt (H1-H2 proxy)
    """
    
    def __init__(
        self,
        f0_min: float = 75.0,
        f0_max: float = 500.0,
        frame_length: int = 2048,
        hop_length: int = 512
    ):
        """
        Initialize feature extractor.
        
        Args:
            f0_min: Minimum F0 for pitch detection (Hz)
            f0_max: Maximum F0 for pitch detection (Hz)
            frame_length: Frame length for spectral analysis
            hop_length: Hop length for spectral analysis
        """
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.frame_length = frame_length
        self.hop_length = hop_length
    
    def extract_all(
        self,
        audio: np.ndarray,
        sample_rate: int,
        audio_id: str = "",
        transcript: Optional[str] = None
    ) -> AudioFeatures:
        """
        Extract all acoustic features from audio.
        
        Args:
            audio: Audio data (1D numpy array)
            sample_rate: Sample rate in Hz
            audio_id: Identifier for the audio sample
            transcript: Optional transcript for speech rate calculation
        
        Returns:
            AudioFeatures containing all extracted features
        """
        f0_features = self.extract_f0_features(audio, sample_rate)
        temporal_features = self.extract_temporal_features(audio, sample_rate, transcript)
        intensity_features = self.extract_intensity_features(audio, sample_rate)
        
        return AudioFeatures(
            audio_id=audio_id,
            f0=f0_features,
            temporal=temporal_features,
            intensity=intensity_features
        )
    
    def extract_f0_features(self, audio: np.ndarray, sample_rate: int) -> F0Features:
        """
        Extract fundamental frequency (F0) features.
        
        Uses Parselmouth (Praat) if available for more accurate extraction,
        otherwise falls back to librosa's pyin algorithm.
        
        Args:
            audio: Audio data
            sample_rate: Sample rate
        
        Returns:
            F0Features with mean, variance, range, min, max
        """
        f0_values = self._extract_f0(audio, sample_rate)
        
        # Filter out unvoiced frames (NaN or 0)
        voiced_f0 = f0_values[~np.isnan(f0_values) & (f0_values > 0)]
        
        if len(voiced_f0) == 0:
            # No voiced frames detected
            return F0Features(
                mean=0.0,
                variance=0.0,
                range=0.0,
                min=0.0,
                max=0.0
            )
        
        return F0Features(
            mean=float(np.mean(voiced_f0)),
            variance=float(np.var(voiced_f0)),
            range=float(np.max(voiced_f0) - np.min(voiced_f0)),
            min=float(np.min(voiced_f0)),
            max=float(np.max(voiced_f0))
        )
    
    def _extract_f0(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extract F0 contour using best available method."""
        if HAS_PARSELMOUTH:
            return self._extract_f0_parselmouth(audio, sample_rate)
        elif HAS_LIBROSA:
            return self._extract_f0_librosa(audio, sample_rate)
        else:
            return self._extract_f0_autocorr(audio, sample_rate)
    
    def _extract_f0_parselmouth(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extract F0 using Parselmouth (Praat)."""
        sound = parselmouth.Sound(audio, sampling_frequency=sample_rate)
        pitch = call(sound, "To Pitch", 0.0, self.f0_min, self.f0_max)
        
        # Get F0 values at regular intervals
        duration = sound.duration
        time_step = self.hop_length / sample_rate
        times = np.arange(0, duration, time_step)
        
        f0_values = np.array([
            call(pitch, "Get value at time", t, "Hertz", "Linear")
            for t in times
        ])
        
        return f0_values
    
    def _extract_f0_librosa(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extract F0 using librosa's pyin algorithm."""
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=self.f0_min,
            fmax=self.f0_max,
            sr=sample_rate,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )
        return f0
    
    def _extract_f0_autocorr(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Fallback F0 extraction using autocorrelation."""
        # Simple autocorrelation-based pitch detection
        frame_size = self.frame_length
        hop = self.hop_length
        
        num_frames = (len(audio) - frame_size) // hop + 1
        f0_values = np.zeros(num_frames)
        
        for i in range(num_frames):
            start = i * hop
            frame = audio[start:start + frame_size]
            
            # Autocorrelation
            corr = np.correlate(frame, frame, mode='full')
            corr = corr[len(corr)//2:]
            
            # Find first peak after minimum lag
            min_lag = int(sample_rate / self.f0_max)
            max_lag = int(sample_rate / self.f0_min)
            
            if max_lag > len(corr):
                max_lag = len(corr)
            
            search_region = corr[min_lag:max_lag]
            if len(search_region) > 0:
                peak_idx = np.argmax(search_region) + min_lag
                if corr[peak_idx] > 0.3 * corr[0]:  # Voicing threshold
                    f0_values[i] = sample_rate / peak_idx
                else:
                    f0_values[i] = np.nan
            else:
                f0_values[i] = np.nan
        
        return f0_values
    
    def extract_temporal_features(
        self,
        audio: np.ndarray,
        sample_rate: int,
        transcript: Optional[str] = None
    ) -> TemporalFeatures:
        """
        Extract temporal features (speech rate, syllable rate).
        
        Args:
            audio: Audio data
            sample_rate: Sample rate
            transcript: Optional transcript for word count
        
        Returns:
            TemporalFeatures with speech rate, syllable rate, duration
        """
        duration = len(audio) / sample_rate
        
        # Calculate speech rate if transcript provided
        if transcript:
            words = transcript.split()
            num_words = len(words)
            speech_rate_wpm = (num_words / duration) * 60 if duration > 0 else 0
            
            # Estimate syllables (rough approximation)
            num_syllables = self._estimate_syllables(transcript)
            syllable_rate = num_syllables / duration if duration > 0 else 0
        else:
            # Estimate from audio using energy-based syllable detection
            syllable_rate = self._estimate_syllable_rate_from_audio(audio, sample_rate)
            # Rough estimate: average 1.5 syllables per word
            speech_rate_wpm = (syllable_rate / 1.5) * 60
        
        return TemporalFeatures(
            speech_rate_wpm=float(speech_rate_wpm),
            syllable_rate=float(syllable_rate),
            duration=float(duration)
        )
    
    def _estimate_syllables(self, text: str) -> int:
        """Estimate number of syllables in text."""
        # Simple vowel-based estimation
        vowels = "aeiouAEIOU"
        text = text.lower()
        count = 0
        prev_vowel = False
        
        for char in text:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        
        # Adjust for common patterns
        count = max(count, len(text.split()))  # At least one syllable per word
        return count
    
    def _estimate_syllable_rate_from_audio(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> float:
        """Estimate syllable rate from audio energy envelope."""
        # Compute energy envelope
        frame_length = int(0.025 * sample_rate)  # 25ms frames
        hop_length = int(0.010 * sample_rate)    # 10ms hop
        
        num_frames = (len(audio) - frame_length) // hop_length + 1
        energy = np.zeros(num_frames)
        
        for i in range(num_frames):
            start = i * hop_length
            frame = audio[start:start + frame_length]
            energy[i] = np.sum(frame ** 2)
        
        # Smooth energy
        if len(energy) > 5:
            kernel = np.ones(5) / 5
            energy = np.convolve(energy, kernel, mode='same')
        
        # Find peaks (syllable nuclei)
        threshold = np.mean(energy) * 0.5
        peaks = []
        for i in range(1, len(energy) - 1):
            if energy[i] > energy[i-1] and energy[i] > energy[i+1] and energy[i] > threshold:
                peaks.append(i)
        
        duration = len(audio) / sample_rate
        syllable_rate = len(peaks) / duration if duration > 0 else 0
        
        return syllable_rate
    
    def extract_intensity_features(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> IntensityFeatures:
        """
        Extract intensity and spectral features.
        
        Args:
            audio: Audio data
            sample_rate: Sample rate
        
        Returns:
            IntensityFeatures with RMS, ZCR, spectral tilt
        """
        # RMS intensity
        rms = np.sqrt(np.mean(audio ** 2))
        rms_db = 20 * np.log10(rms) if rms > 1e-10 else -100.0
        
        # Zero-crossing rate
        zcr = self._compute_zcr(audio)
        
        # Spectral tilt (H1-H2 proxy)
        spectral_tilt = self._compute_spectral_tilt(audio, sample_rate)
        
        return IntensityFeatures(
            rms_db=float(rms_db),
            zero_crossing_rate=float(zcr),
            spectral_tilt=float(spectral_tilt),
            lufs=0.0  # LUFS computed separately if needed
        )
    
    def _compute_zcr(self, audio: np.ndarray) -> float:
        """Compute zero-crossing rate."""
        signs = np.sign(audio)
        signs[signs == 0] = 1  # Treat zeros as positive
        crossings = np.sum(np.abs(np.diff(signs)) > 0)
        return crossings / len(audio)
    
    def _compute_spectral_tilt(self, audio: np.ndarray, sample_rate: int) -> float:
        """
        Compute spectral tilt as H1-H2 proxy.
        
        H1-H2 is the difference between the first and second harmonics,
        which correlates with voice quality and breathiness.
        """
        # Compute spectrum
        n_fft = min(2048, len(audio))
        if len(audio) < n_fft:
            audio = np.pad(audio, (0, n_fft - len(audio)))
        
        spectrum = np.abs(np.fft.rfft(audio[:n_fft]))
        freqs = np.fft.rfftfreq(n_fft, 1/sample_rate)
        
        # Find spectral slope (linear regression in log domain)
        log_spectrum = np.log10(spectrum + 1e-10)
        log_freqs = np.log10(freqs + 1)
        
        # Simple linear fit
        valid = np.isfinite(log_spectrum) & np.isfinite(log_freqs)
        if np.sum(valid) < 2:
            return 0.0
        
        coeffs = np.polyfit(log_freqs[valid], log_spectrum[valid], 1)
        return float(coeffs[0])  # Slope as spectral tilt
    
    def compute_prosody_deltas(
        self,
        features_dict: dict,
        baseline_key: str = "neutral"
    ) -> dict:
        """
        Compute prosody feature changes relative to baseline.
        
        Args:
            features_dict: Dict mapping prosody condition to AudioFeatures
            baseline_key: Key for baseline condition (default: "neutral")
        
        Returns:
            Dict mapping prosody condition to relative changes
        """
        if baseline_key not in features_dict:
            raise ValueError(f"Baseline '{baseline_key}' not found in features")
        
        baseline = features_dict[baseline_key]
        deltas = {}
        
        for key, features in features_dict.items():
            if key == baseline_key:
                continue
            
            # F0 variance ratio
            f0_var_ratio = (
                features.f0.variance / baseline.f0.variance
                if baseline.f0.variance > 0 else 1.0
            )
            
            # F0 mean change
            f0_mean_delta = features.f0.mean - baseline.f0.mean
            
            # Speech rate change
            rate_delta = features.temporal.speech_rate_wpm - baseline.temporal.speech_rate_wpm
            rate_ratio = (
                features.temporal.speech_rate_wpm / baseline.temporal.speech_rate_wpm
                if baseline.temporal.speech_rate_wpm > 0 else 1.0
            )
            
            deltas[key] = {
                "f0_variance_ratio": f0_var_ratio,
                "f0_mean_delta": f0_mean_delta,
                "speech_rate_delta": rate_delta,
                "speech_rate_ratio": rate_ratio,
            }
        
        return deltas
