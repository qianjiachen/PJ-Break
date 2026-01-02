"""
Visualization module for PJ-Break experiment reproduction.

Generates figures from the paper:
- Figure 2: Acoustic features comparison
- Figure 3: ASR by prosody condition
- Figure 4: Latent space visualization
- Figure 5: Pro-Guard architecture
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Try to import matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not installed. Visualization unavailable.")

# Try to import seaborn
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


class Visualizer:
    """
    Generates visualizations for experiment results.
    """
    
    # Color scheme from paper
    PROSODY_COLORS = {
        "neutral": "#808080",
        "panic": "#FF4444",
        "anger": "#FF8800",
        "commanding": "#4444FF",
        "fast": "#44FF44",
        "whisper": "#AA44AA",
    }
    
    MODEL_COLORS = {
        "qwen2-audio": "#1f77b4",
        "gpt-4o": "#ff7f0e",
        "gemini": "#2ca02c",
        "salmonn": "#d62728",
    }
    
    def __init__(self, output_dir: Optional[str] = None, style: str = "seaborn-v0_8-whitegrid"):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save figures
            style: Matplotlib style
        """
        self.output_dir = Path(output_dir) if output_dir else None
        
        if HAS_MATPLOTLIB:
            try:
                plt.style.use(style)
            except:
                plt.style.use("default")
    
    def plot_acoustic_features(
        self,
        features_by_condition: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None
    ) -> Optional[Any]:
        """
        Figure 2: Acoustic features comparison across prosody conditions.
        
        Args:
            features_by_condition: Dict of prosody -> {f0_mean, f0_var, rate, intensity}
            save_path: Path to save figure
        
        Returns:
            Figure object or None
        """
        if not HAS_MATPLOTLIB:
            logger.warning("matplotlib not available")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        conditions = list(features_by_condition.keys())
        colors = [self.PROSODY_COLORS.get(c, "#888888") for c in conditions]
        
        # F0 Mean
        ax = axes[0, 0]
        f0_means = [features_by_condition[c].get("f0_mean", 0) for c in conditions]
        ax.bar(conditions, f0_means, color=colors)
        ax.set_ylabel("F0 Mean (Hz)")
        ax.set_title("Fundamental Frequency Mean")
        ax.tick_params(axis='x', rotation=45)
        
        # F0 Variance
        ax = axes[0, 1]
        f0_vars = [features_by_condition[c].get("f0_variance", 0) for c in conditions]
        ax.bar(conditions, f0_vars, color=colors)
        ax.set_ylabel("F0 Variance")
        ax.set_title("Fundamental Frequency Variance")
        ax.tick_params(axis='x', rotation=45)
        
        # Speech Rate
        ax = axes[1, 0]
        rates = [features_by_condition[c].get("speech_rate_wpm", 0) for c in conditions]
        ax.bar(conditions, rates, color=colors)
        ax.set_ylabel("Speech Rate (wpm)")
        ax.set_title("Speech Rate")
        ax.tick_params(axis='x', rotation=45)
        
        # Intensity
        ax = axes[1, 1]
        intensities = [features_by_condition[c].get("rms_db", 0) for c in conditions]
        ax.bar(conditions, intensities, color=colors)
        ax.set_ylabel("RMS Intensity (dBFS)")
        ax.set_title("Audio Intensity")
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_asr_by_prosody(
        self,
        asr_by_condition: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None
    ) -> Optional[Any]:
        """
        Figure 3: ASR by prosody condition for each model.
        
        Args:
            asr_by_condition: Dict of model -> {prosody -> asr}
            save_path: Path to save figure
        
        Returns:
            Figure object or None
        """
        if not HAS_MATPLOTLIB:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        models = list(asr_by_condition.keys())
        conditions = list(self.PROSODY_COLORS.keys())
        
        x = np.arange(len(conditions))
        width = 0.2
        
        for i, model in enumerate(models):
            asrs = [asr_by_condition[model].get(c, 0) for c in conditions]
            color = self.MODEL_COLORS.get(model, f"C{i}")
            ax.bar(x + i * width, asrs, width, label=model, color=color)
        
        ax.set_xlabel("Prosody Condition")
        ax.set_ylabel("Attack Success Rate")
        ax.set_title("ASR by Prosody Condition and Model")
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(conditions, rotation=45)
        ax.legend()
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_latent_space(
        self,
        activations: Dict[str, np.ndarray],
        labels: List[str],
        save_path: Optional[str] = None,
        method: str = "tsne"
    ) -> Optional[Any]:
        """
        Figure 4: Latent space visualization.
        
        Args:
            activations: Dict of sample_id -> activation vector
            labels: Prosody labels for each sample
            save_path: Path to save figure
            method: Dimensionality reduction method ("tsne" or "umap")
        
        Returns:
            Figure object or None
        """
        if not HAS_MATPLOTLIB:
            return None
        
        # Stack activations
        sample_ids = list(activations.keys())
        X = np.stack([activations[sid] for sid in sample_ids])
        
        # Reduce dimensionality
        if method == "tsne":
            try:
                from sklearn.manifold import TSNE
                reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1))
                X_2d = reducer.fit_transform(X)
            except ImportError:
                logger.warning("sklearn not available for t-SNE")
                return None
        elif method == "umap":
            try:
                import umap
                reducer = umap.UMAP(n_components=2, random_state=42)
                X_2d = reducer.fit_transform(X)
            except ImportError:
                logger.warning("umap not available")
                return None
        else:
            # Simple PCA fallback
            X_centered = X - X.mean(axis=0)
            _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
            X_2d = X_centered @ Vt[:2].T
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot by prosody condition
        unique_labels = list(set(labels))
        for label in unique_labels:
            mask = [l == label for l in labels]
            color = self.PROSODY_COLORS.get(label, "#888888")
            ax.scatter(
                X_2d[mask, 0], X_2d[mask, 1],
                c=color, label=label, alpha=0.7, s=50
            )
        
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.set_title(f"Latent Space Visualization ({method.upper()})")
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_defense_comparison(
        self,
        defense_results: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None
    ) -> Optional[Any]:
        """
        Plot defense comparison (ASR reduction vs FPR).
        
        Args:
            defense_results: Dict of defense -> {asr_reduction, fpr, latency}
            save_path: Path to save figure
        
        Returns:
            Figure object or None
        """
        if not HAS_MATPLOTLIB:
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        defenses = list(defense_results.keys())
        
        # ASR Reduction
        ax = axes[0]
        asr_reductions = [defense_results[d].get("asr_reduction", 0) for d in defenses]
        ax.bar(defenses, asr_reductions, color='steelblue')
        ax.set_ylabel("ASR Reduction")
        ax.set_title("Defense Effectiveness")
        ax.tick_params(axis='x', rotation=45)
        
        # FPR
        ax = axes[1]
        fprs = [defense_results[d].get("fpr", 0) for d in defenses]
        ax.bar(defenses, fprs, color='coral')
        ax.set_ylabel("False Positive Rate")
        ax.set_title("Defense False Positives")
        ax.tick_params(axis='x', rotation=45)
        ax.axhline(y=0.05, color='red', linestyle='--', label='5% threshold')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_category_analysis(
        self,
        asr_by_category: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None
    ) -> Optional[Any]:
        """
        Plot ASR by harm category.
        
        Args:
            asr_by_category: Dict of category -> {text_only, pj_break}
            save_path: Path to save figure
        
        Returns:
            Figure object or None
        """
        if not HAS_MATPLOTLIB:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        categories = list(asr_by_category.keys())
        x = np.arange(len(categories))
        width = 0.35
        
        text_only = [asr_by_category[c].get("text_only", 0) for c in categories]
        pj_break = [asr_by_category[c].get("pj_break", 0) for c in categories]
        
        ax.bar(x - width/2, text_only, width, label='Text-Only', color='gray')
        ax.bar(x + width/2, pj_break, width, label='PJ-Break', color='red')
        
        ax.set_xlabel("Harm Category")
        ax.set_ylabel("Attack Success Rate")
        ax.set_title("ASR by Harm Category")
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def save_all_figures(self, output_dir: str) -> None:
        """Save all figures to directory."""
        if self.output_dir is None:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Figures will be saved to {self.output_dir}")
