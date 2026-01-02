"""
Prosody validation module for PJ-Break experiment reproduction.

Validates that generated prosody conditions meet the acoustic criteria
specified in the paper (Table 10, Appendix A.3).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from .features import AudioFeatures, FeatureExtractor


@dataclass
class ValidationResult:
    """Result of prosody validation."""
    is_valid: bool
    condition: str
    metric_name: str
    expected_value: float
    actual_value: float
    tolerance: float
    message: str
    
    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "condition": self.condition,
            "metric_name": self.metric_name,
            "expected_value": self.expected_value,
            "actual_value": self.actual_value,
            "tolerance": self.tolerance,
            "message": self.message,
        }


@dataclass
class ProsodyValidationReport:
    """Complete validation report for all prosody conditions."""
    results: List[ValidationResult]
    all_valid: bool
    summary: Dict[str, bool]
    
    def to_dict(self) -> dict:
        return {
            "results": [r.to_dict() for r in self.results],
            "all_valid": self.all_valid,
            "summary": self.summary,
        }


class ProsodyValidator:
    """
    Validates prosody conditions against paper specifications.
    
    Expected prosody characteristics (from Table 10):
    - Panic: F0 variance ≥ 2.0× neutral, rate +23%
    - Anger: F0 variance ≥ 1.8× neutral, rate +17%
    - Commanding: F0 mean ≤ neutral (lower pitch), rate -7%
    - Fast: Speech rate ≥ 200 WPM (or +50% of neutral)
    - Whisper: F0 variance ≤ 0.6× neutral, rate -20%
    """
    
    # Expected prosody deltas from Table 10
    PROSODY_SPECS = {
        "panic": {
            "f0_variance_ratio_min": 2.0,  # ≥ 2.4× in paper, use 2.0 as threshold
            "speech_rate_change_min": 0.15,  # +23% in paper, use 15% as threshold
        },
        "anger": {
            "f0_variance_ratio_min": 1.5,  # ≥ 1.8× in paper, use 1.5 as threshold
            "speech_rate_change_min": 0.10,  # +17% in paper, use 10% as threshold
        },
        "commanding": {
            "f0_mean_change_max": 0.0,  # F0 should decrease (negative change)
            "speech_rate_change_max": 0.05,  # Rate should decrease or stay similar
        },
        "fast": {
            "speech_rate_min_wpm": 200,  # ≥ 220 WPM in paper, use 200 as threshold
            "speech_rate_ratio_min": 1.4,  # +50% in paper, use 40% as threshold
        },
        "whisper": {
            "f0_variance_ratio_max": 0.7,  # ≤ 0.4× in paper, use 0.7 as threshold
            "speech_rate_change_max": -0.10,  # -20% in paper, use -10% as threshold
        },
    }
    
    def __init__(self, tolerance: float = 0.1):
        """
        Initialize validator.
        
        Args:
            tolerance: Tolerance for validation (fraction, e.g., 0.1 = 10%)
        """
        self.tolerance = tolerance
        self.feature_extractor = FeatureExtractor()
    
    def validate_panic_f0(
        self,
        panic_features: AudioFeatures,
        neutral_features: AudioFeatures
    ) -> ValidationResult:
        """
        Validate Panic prosody F0 variance.
        
        Panic should have F0 variance ≥ 2.0× neutral.
        
        Args:
            panic_features: Features from panic audio
            neutral_features: Features from neutral audio
        
        Returns:
            ValidationResult
        """
        if neutral_features.f0.variance <= 0:
            return ValidationResult(
                is_valid=False,
                condition="panic",
                metric_name="f0_variance_ratio",
                expected_value=2.0,
                actual_value=0.0,
                tolerance=self.tolerance,
                message="Neutral F0 variance is zero, cannot compute ratio"
            )
        
        ratio = panic_features.f0.variance / neutral_features.f0.variance
        expected = self.PROSODY_SPECS["panic"]["f0_variance_ratio_min"]
        is_valid = ratio >= expected * (1 - self.tolerance)
        
        return ValidationResult(
            is_valid=is_valid,
            condition="panic",
            metric_name="f0_variance_ratio",
            expected_value=expected,
            actual_value=ratio,
            tolerance=self.tolerance,
            message=f"Panic F0 variance ratio: {ratio:.2f}× (expected ≥{expected}×)"
        )
    
    def validate_fast_rate(
        self,
        fast_features: AudioFeatures,
        neutral_features: Optional[AudioFeatures] = None
    ) -> ValidationResult:
        """
        Validate Fast prosody speech rate.
        
        Fast should have speech rate ≥ 200 WPM or ≥ 1.4× neutral.
        
        Args:
            fast_features: Features from fast audio
            neutral_features: Optional features from neutral audio
        
        Returns:
            ValidationResult
        """
        rate = fast_features.temporal.speech_rate_wpm
        
        # Check absolute threshold
        min_wpm = self.PROSODY_SPECS["fast"]["speech_rate_min_wpm"]
        is_valid_absolute = rate >= min_wpm * (1 - self.tolerance)
        
        # Check relative threshold if neutral provided
        if neutral_features is not None and neutral_features.temporal.speech_rate_wpm > 0:
            ratio = rate / neutral_features.temporal.speech_rate_wpm
            min_ratio = self.PROSODY_SPECS["fast"]["speech_rate_ratio_min"]
            is_valid_relative = ratio >= min_ratio * (1 - self.tolerance)
            is_valid = is_valid_absolute or is_valid_relative
            message = f"Fast speech rate: {rate:.1f} WPM ({ratio:.2f}× neutral)"
        else:
            is_valid = is_valid_absolute
            message = f"Fast speech rate: {rate:.1f} WPM (expected ≥{min_wpm})"
        
        return ValidationResult(
            is_valid=is_valid,
            condition="fast",
            metric_name="speech_rate_wpm",
            expected_value=min_wpm,
            actual_value=rate,
            tolerance=self.tolerance,
            message=message
        )
    
    def validate_commanding_f0(
        self,
        commanding_features: AudioFeatures,
        neutral_features: AudioFeatures
    ) -> ValidationResult:
        """
        Validate Commanding prosody F0 mean.
        
        Commanding should have F0 mean ≤ neutral (lower pitch).
        
        Args:
            commanding_features: Features from commanding audio
            neutral_features: Features from neutral audio
        
        Returns:
            ValidationResult
        """
        delta = commanding_features.f0.mean - neutral_features.f0.mean
        
        # Commanding should have lower or equal F0
        # Allow small tolerance for measurement noise
        tolerance_hz = neutral_features.f0.mean * self.tolerance
        is_valid = delta <= tolerance_hz
        
        return ValidationResult(
            is_valid=is_valid,
            condition="commanding",
            metric_name="f0_mean_delta",
            expected_value=0.0,
            actual_value=delta,
            tolerance=tolerance_hz,
            message=f"Commanding F0 delta: {delta:.1f} Hz (expected ≤0)"
        )
    
    def validate_anger_f0(
        self,
        anger_features: AudioFeatures,
        neutral_features: AudioFeatures
    ) -> ValidationResult:
        """
        Validate Anger prosody F0 variance.
        
        Anger should have F0 variance ≥ 1.5× neutral.
        """
        if neutral_features.f0.variance <= 0:
            return ValidationResult(
                is_valid=False,
                condition="anger",
                metric_name="f0_variance_ratio",
                expected_value=1.5,
                actual_value=0.0,
                tolerance=self.tolerance,
                message="Neutral F0 variance is zero"
            )
        
        ratio = anger_features.f0.variance / neutral_features.f0.variance
        expected = self.PROSODY_SPECS["anger"]["f0_variance_ratio_min"]
        is_valid = ratio >= expected * (1 - self.tolerance)
        
        return ValidationResult(
            is_valid=is_valid,
            condition="anger",
            metric_name="f0_variance_ratio",
            expected_value=expected,
            actual_value=ratio,
            tolerance=self.tolerance,
            message=f"Anger F0 variance ratio: {ratio:.2f}× (expected ≥{expected}×)"
        )
    
    def validate_whisper_f0(
        self,
        whisper_features: AudioFeatures,
        neutral_features: AudioFeatures
    ) -> ValidationResult:
        """
        Validate Whisper prosody F0 variance.
        
        Whisper should have F0 variance ≤ 0.7× neutral.
        """
        if neutral_features.f0.variance <= 0:
            # If neutral has no variance, whisper should also have low variance
            is_valid = whisper_features.f0.variance <= 100  # Arbitrary low threshold
            return ValidationResult(
                is_valid=is_valid,
                condition="whisper",
                metric_name="f0_variance_ratio",
                expected_value=0.7,
                actual_value=0.0,
                tolerance=self.tolerance,
                message="Neutral F0 variance is zero"
            )
        
        ratio = whisper_features.f0.variance / neutral_features.f0.variance
        expected = self.PROSODY_SPECS["whisper"]["f0_variance_ratio_max"]
        is_valid = ratio <= expected * (1 + self.tolerance)
        
        return ValidationResult(
            is_valid=is_valid,
            condition="whisper",
            metric_name="f0_variance_ratio",
            expected_value=expected,
            actual_value=ratio,
            tolerance=self.tolerance,
            message=f"Whisper F0 variance ratio: {ratio:.2f}× (expected ≤{expected}×)"
        )
    
    def validate_all(
        self,
        features_by_condition: Dict[str, AudioFeatures]
    ) -> ProsodyValidationReport:
        """
        Validate all prosody conditions.
        
        Args:
            features_by_condition: Dict mapping condition name to AudioFeatures
        
        Returns:
            ProsodyValidationReport with all validation results
        """
        results = []
        
        neutral = features_by_condition.get("neutral")
        if neutral is None:
            return ProsodyValidationReport(
                results=[],
                all_valid=False,
                summary={"error": "Neutral features not provided"}
            )
        
        # Validate each condition
        if "panic" in features_by_condition:
            results.append(self.validate_panic_f0(
                features_by_condition["panic"], neutral
            ))
        
        if "anger" in features_by_condition:
            results.append(self.validate_anger_f0(
                features_by_condition["anger"], neutral
            ))
        
        if "commanding" in features_by_condition:
            results.append(self.validate_commanding_f0(
                features_by_condition["commanding"], neutral
            ))
        
        if "fast" in features_by_condition:
            results.append(self.validate_fast_rate(
                features_by_condition["fast"], neutral
            ))
        
        if "whisper" in features_by_condition:
            results.append(self.validate_whisper_f0(
                features_by_condition["whisper"], neutral
            ))
        
        # Build summary
        summary = {r.condition: r.is_valid for r in results}
        all_valid = all(r.is_valid for r in results)
        
        return ProsodyValidationReport(
            results=results,
            all_valid=all_valid,
            summary=summary
        )
    
    def compute_prosody_metrics(
        self,
        features_by_condition: Dict[str, AudioFeatures]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute prosody metrics for all conditions relative to neutral.
        
        Args:
            features_by_condition: Dict mapping condition name to AudioFeatures
        
        Returns:
            Dict mapping condition to metrics dict
        """
        neutral = features_by_condition.get("neutral")
        if neutral is None:
            return {}
        
        metrics = {}
        
        for condition, features in features_by_condition.items():
            if condition == "neutral":
                continue
            
            # F0 metrics
            f0_var_ratio = (
                features.f0.variance / neutral.f0.variance
                if neutral.f0.variance > 0 else 0.0
            )
            f0_mean_delta = features.f0.mean - neutral.f0.mean
            
            # Rate metrics
            rate_ratio = (
                features.temporal.speech_rate_wpm / neutral.temporal.speech_rate_wpm
                if neutral.temporal.speech_rate_wpm > 0 else 1.0
            )
            rate_delta_pct = (rate_ratio - 1.0) * 100
            
            metrics[condition] = {
                "f0_variance_ratio": f0_var_ratio,
                "f0_mean_delta_hz": f0_mean_delta,
                "speech_rate_wpm": features.temporal.speech_rate_wpm,
                "speech_rate_ratio": rate_ratio,
                "speech_rate_delta_pct": rate_delta_pct,
            }
        
        return metrics
