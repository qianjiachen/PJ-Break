"""
Statistical metrics for PJ-Break experiment reproduction.

Implements all statistical measures from the paper:
- ASR with Wilson confidence intervals
- False positive rate
- Fleiss' kappa and Cohen's kappa
- McNemar's test
- Bonferroni correction
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from scipy import stats
import logging

from ..models_data import ASRMetric, KappaResult, AttackResult, DefenseResult

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculator for evaluation metrics."""
    
    def compute_asr(
        self,
        results: List[AttackResult],
        confidence: float = 0.95
    ) -> ASRMetric:
        """
        Compute Attack Success Rate with Wilson confidence interval.
        
        Args:
            results: List of attack results
            confidence: Confidence level (default: 0.95)
        
        Returns:
            ASRMetric with ASR and confidence interval
        """
        if not results:
            return ASRMetric(asr=0.0, ci_lower=0.0, ci_upper=0.0, n=0)
        
        successes = sum(1 for r in results if r.is_success)
        n = len(results)
        
        asr = successes / n
        ci_lower, ci_upper = self.wilson_confidence_interval(successes, n, confidence)
        
        return ASRMetric(asr=asr, ci_lower=ci_lower, ci_upper=ci_upper, n=n)
    
    def compute_asr_from_binary(
        self,
        successes: List[bool],
        confidence: float = 0.95
    ) -> ASRMetric:
        """
        Compute ASR from binary success list.
        
        Args:
            successes: List of boolean success values
            confidence: Confidence level
        
        Returns:
            ASRMetric
        """
        if not successes:
            return ASRMetric(asr=0.0, ci_lower=0.0, ci_upper=0.0, n=0)
        
        num_success = sum(successes)
        n = len(successes)
        
        asr = num_success / n
        ci_lower, ci_upper = self.wilson_confidence_interval(num_success, n, confidence)
        
        return ASRMetric(asr=asr, ci_lower=ci_lower, ci_upper=ci_upper, n=n)
    
    def compute_fpr(self, defense_results: List[DefenseResult]) -> float:
        """
        Compute False Positive Rate for defense.
        
        Args:
            defense_results: Results on benign queries
        
        Returns:
            False positive rate
        """
        if not defense_results:
            return 0.0
        
        false_positives = sum(1 for r in defense_results if r.is_blocked)
        return false_positives / len(defense_results)
    
    def wilson_confidence_interval(
        self,
        successes: int,
        n: int,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Compute Wilson score confidence interval.
        
        Args:
            successes: Number of successes
            n: Total trials
            confidence: Confidence level
        
        Returns:
            Tuple of (lower, upper) bounds
        """
        if n == 0:
            return (0.0, 0.0)
        
        z = stats.norm.ppf(1 - (1 - confidence) / 2)
        p = successes / n
        
        denominator = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denominator
        margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator
        
        return (max(0.0, center - margin), min(1.0, center + margin))
    
    def compute_fleiss_kappa(
        self,
        ratings: np.ndarray,
        confidence: float = 0.95
    ) -> KappaResult:
        """
        Compute Fleiss' kappa for inter-rater agreement.
        
        Args:
            ratings: Matrix of shape (n_samples, n_raters) with category labels
            confidence: Confidence level for CI
        
        Returns:
            KappaResult with kappa and confidence interval
        """
        n_samples, n_raters = ratings.shape
        
        if n_samples == 0 or n_raters == 0:
            return KappaResult(kappa=0.0, ci_lower=0.0, ci_upper=0.0)
        
        # Get unique categories
        categories = np.unique(ratings)
        n_categories = len(categories)
        
        # Build count matrix
        counts = np.zeros((n_samples, n_categories))
        for i, cat in enumerate(categories):
            counts[:, i] = np.sum(ratings == cat, axis=1)
        
        # Compute P_i (agreement for each sample)
        P_i = (np.sum(counts**2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
        P_bar = np.mean(P_i)
        
        # Compute P_j (proportion for each category)
        p_j = np.sum(counts, axis=0) / (n_samples * n_raters)
        P_e = np.sum(p_j**2)
        
        # Compute kappa
        if P_e == 1:
            kappa = 1.0
        else:
            kappa = (P_bar - P_e) / (1 - P_e)
        
        # Compute standard error and CI
        # Using Fleiss' formula for SE
        if P_e == 1:
            se = 0.0
        else:
            se = np.sqrt(2 * (P_e - np.sum(p_j**3))) / ((1 - P_e) * np.sqrt(n_samples * n_raters * (n_raters - 1)))
        
        z = stats.norm.ppf(1 - (1 - confidence) / 2)
        ci_lower = max(-1.0, kappa - z * se)
        ci_upper = min(1.0, kappa + z * se)
        
        return KappaResult(kappa=kappa, ci_lower=ci_lower, ci_upper=ci_upper)
    
    def compute_cohens_kappa(
        self,
        rater1: List[int],
        rater2: List[int]
    ) -> float:
        """
        Compute Cohen's kappa for two raters.
        
        Args:
            rater1: Ratings from first rater
            rater2: Ratings from second rater
        
        Returns:
            Cohen's kappa value
        """
        if len(rater1) != len(rater2):
            raise ValueError("Raters must have same number of ratings")
        
        if len(rater1) == 0:
            return 0.0
        
        # Build confusion matrix
        categories = sorted(set(rater1) | set(rater2))
        n_cat = len(categories)
        cat_to_idx = {c: i for i, c in enumerate(categories)}
        
        matrix = np.zeros((n_cat, n_cat))
        for r1, r2 in zip(rater1, rater2):
            matrix[cat_to_idx[r1], cat_to_idx[r2]] += 1
        
        n = len(rater1)
        
        # Observed agreement
        p_o = np.trace(matrix) / n
        
        # Expected agreement
        row_sums = np.sum(matrix, axis=1)
        col_sums = np.sum(matrix, axis=0)
        p_e = np.sum(row_sums * col_sums) / (n**2)
        
        # Kappa
        if p_e == 1:
            return 1.0
        
        return (p_o - p_e) / (1 - p_e)
    
    def mcnemar_test(
        self,
        results1: List[bool],
        results2: List[bool]
    ) -> float:
        """
        Perform McNemar's test for paired binary data.
        
        Args:
            results1: Binary results from condition 1
            results2: Binary results from condition 2
        
        Returns:
            p-value
        """
        if len(results1) != len(results2):
            raise ValueError("Results must have same length")
        
        if len(results1) == 0:
            return 1.0
        
        # Build contingency table
        # b = results1 success, results2 failure
        # c = results1 failure, results2 success
        b = sum(1 for r1, r2 in zip(results1, results2) if r1 and not r2)
        c = sum(1 for r1, r2 in zip(results1, results2) if not r1 and r2)
        
        # McNemar's test (exact for small samples)
        if b + c == 0:
            return 1.0
        
        if b + c < 25:
            # Exact binomial test
            result = stats.binomtest(b, b + c, 0.5).pvalue
        else:
            # Chi-squared approximation with continuity correction
            chi2 = (abs(b - c) - 1)**2 / (b + c)
            result = 1 - stats.chi2.cdf(chi2, df=1)
        
        return float(result)
    
    def bonferroni_correction(
        self,
        p_values: List[float],
        alpha: float = 0.05
    ) -> Tuple[List[bool], float]:
        """
        Apply Bonferroni correction for multiple comparisons.
        
        Args:
            p_values: List of p-values
            alpha: Significance level
        
        Returns:
            Tuple of (significant flags, corrected alpha)
        """
        if not p_values:
            return ([], alpha)
        
        corrected_alpha = alpha / len(p_values)
        significant = [p < corrected_alpha for p in p_values]
        
        return (significant, corrected_alpha)
    
    def compute_effect_size(
        self,
        asr1: float,
        asr2: float,
        n1: int,
        n2: int
    ) -> float:
        """
        Compute Cohen's h effect size for proportions.
        
        Args:
            asr1: First ASR
            asr2: Second ASR
            n1: Sample size 1
            n2: Sample size 2
        
        Returns:
            Cohen's h effect size
        """
        # Arcsine transformation
        phi1 = 2 * np.arcsin(np.sqrt(asr1))
        phi2 = 2 * np.arcsin(np.sqrt(asr2))
        
        return abs(phi1 - phi2)


def compute_asr(results: List[AttackResult], confidence: float = 0.95) -> ASRMetric:
    """Convenience function for ASR computation."""
    calc = MetricsCalculator()
    return calc.compute_asr(results, confidence)


def compute_fpr(defense_results: List[DefenseResult]) -> float:
    """Convenience function for FPR computation."""
    calc = MetricsCalculator()
    return calc.compute_fpr(defense_results)


def mcnemar_test(results1: List[bool], results2: List[bool]) -> float:
    """Convenience function for McNemar's test."""
    calc = MetricsCalculator()
    return calc.mcnemar_test(results1, results2)
