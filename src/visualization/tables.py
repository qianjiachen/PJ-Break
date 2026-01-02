"""
Table generation for PJ-Break experiment reproduction.

Generates LaTeX tables for the paper:
- Table 3: Main attack results
- Table 4: Prosody dimension analysis
- Table 5: Category analysis
- Table 7: Defense comparison
- Table 8: Pro-Guard ablation
"""

from typing import Dict, List, Optional, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import pandas
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    logger.warning("pandas not installed. Table generation limited.")


class TableGenerator:
    """
    Generates LaTeX tables for experiment results.
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize table generator.
        
        Args:
            output_dir: Directory to save tables
        """
        self.output_dir = Path(output_dir) if output_dir else None
    
    def generate_main_results_table(
        self,
        results: Dict[str, Dict[str, Any]],
        caption: str = "Main Attack Results"
    ) -> str:
        """
        Generate Table 3: Main attack results.
        
        Args:
            results: Dict of model -> {asr, ci_lower, ci_upper, n}
            caption: Table caption
        
        Returns:
            LaTeX table string
        """
        lines = [
            "\\begin{table}[h]",
            "\\centering",
            f"\\caption{{{caption}}}",
            "\\begin{tabular}{lccc}",
            "\\toprule",
            "Model & ASR (\\%) & 95\\% CI & N \\\\",
            "\\midrule",
        ]
        
        for model, data in results.items():
            asr = data.get("asr", 0) * 100
            ci_lower = data.get("ci_lower", 0) * 100
            ci_upper = data.get("ci_upper", 0) * 100
            n = data.get("n", 0)
            
            lines.append(
                f"{model} & {asr:.1f} & [{ci_lower:.1f}, {ci_upper:.1f}] & {n} \\\\"
            )
        
        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\label{tab:main_results}",
            "\\end{table}",
        ])
        
        return "\n".join(lines)
    
    def generate_prosody_table(
        self,
        results: Dict[str, Dict[str, float]],
        caption: str = "ASR by Prosody Condition"
    ) -> str:
        """
        Generate Table 4: Prosody dimension analysis.
        
        Args:
            results: Dict of prosody -> {asr, f0_var, rate}
            caption: Table caption
        
        Returns:
            LaTeX table string
        """
        lines = [
            "\\begin{table}[h]",
            "\\centering",
            f"\\caption{{{caption}}}",
            "\\begin{tabular}{lccc}",
            "\\toprule",
            "Prosody & ASR (\\%) & F0 Var & Rate (wpm) \\\\",
            "\\midrule",
        ]
        
        for prosody, data in results.items():
            asr = data.get("asr", 0) * 100
            f0_var = data.get("f0_variance", 0)
            rate = data.get("speech_rate_wpm", 0)
            
            lines.append(
                f"{prosody.capitalize()} & {asr:.1f} & {f0_var:.2f} & {rate:.0f} \\\\"
            )
        
        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\label{tab:prosody_analysis}",
            "\\end{table}",
        ])
        
        return "\n".join(lines)
    
    def generate_category_table(
        self,
        results: Dict[str, Dict[str, float]],
        caption: str = "ASR by Harm Category"
    ) -> str:
        """
        Generate Table 5: Category analysis.
        
        Args:
            results: Dict of category -> {text_only, pj_break, delta}
            caption: Table caption
        
        Returns:
            LaTeX table string
        """
        lines = [
            "\\begin{table}[h]",
            "\\centering",
            f"\\caption{{{caption}}}",
            "\\begin{tabular}{lccc}",
            "\\toprule",
            "Category & Text-Only (\\%) & PJ-Break (\\%) & $\\Delta$ \\\\",
            "\\midrule",
        ]
        
        for category, data in results.items():
            text_only = data.get("text_only", 0) * 100
            pj_break = data.get("pj_break", 0) * 100
            delta = pj_break - text_only
            
            lines.append(
                f"{category} & {text_only:.1f} & {pj_break:.1f} & +{delta:.1f} \\\\"
            )
        
        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\label{tab:category_analysis}",
            "\\end{table}",
        ])
        
        return "\n".join(lines)
    
    def generate_defense_table(
        self,
        results: Dict[str, Dict[str, float]],
        caption: str = "Defense Comparison"
    ) -> str:
        """
        Generate Table 7: Defense comparison.
        
        Args:
            results: Dict of defense -> {asr_reduction, fpr, latency_ms}
            caption: Table caption
        
        Returns:
            LaTeX table string
        """
        lines = [
            "\\begin{table}[h]",
            "\\centering",
            f"\\caption{{{caption}}}",
            "\\begin{tabular}{lccc}",
            "\\toprule",
            "Defense & ASR Reduction (\\%) & FPR (\\%) & Latency (ms) \\\\",
            "\\midrule",
        ]
        
        for defense, data in results.items():
            asr_red = data.get("asr_reduction", 0) * 100
            fpr = data.get("fpr", 0) * 100
            latency = data.get("latency_ms", 0)
            
            lines.append(
                f"{defense} & {asr_red:.1f} & {fpr:.1f} & {latency:.0f} \\\\"
            )
        
        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\label{tab:defense_comparison}",
            "\\end{table}",
        ])
        
        return "\n".join(lines)
    
    def generate_ablation_table(
        self,
        results: Dict[str, Dict[str, float]],
        caption: str = "Pro-Guard Ablation Study"
    ) -> str:
        """
        Generate Table 8: Pro-Guard ablation.
        
        Args:
            results: Dict of variant -> {asr_reduction, fpr}
            caption: Table caption
        
        Returns:
            LaTeX table string
        """
        lines = [
            "\\begin{table}[h]",
            "\\centering",
            f"\\caption{{{caption}}}",
            "\\begin{tabular}{lcc}",
            "\\toprule",
            "Variant & ASR Reduction (\\%) & FPR (\\%) \\\\",
            "\\midrule",
        ]
        
        for variant, data in results.items():
            asr_red = data.get("asr_reduction", 0) * 100
            fpr = data.get("fpr", 0) * 100
            
            lines.append(f"{variant} & {asr_red:.1f} & {fpr:.1f} \\\\")
        
        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\label{tab:proguard_ablation}",
            "\\end{table}",
        ])
        
        return "\n".join(lines)
    
    def generate_statistical_table(
        self,
        comparisons: List[Dict[str, Any]],
        caption: str = "Statistical Significance Tests"
    ) -> str:
        """
        Generate table for statistical tests.
        
        Args:
            comparisons: List of {comparison, p_value, significant}
            caption: Table caption
        
        Returns:
            LaTeX table string
        """
        lines = [
            "\\begin{table}[h]",
            "\\centering",
            f"\\caption{{{caption}}}",
            "\\begin{tabular}{lcc}",
            "\\toprule",
            "Comparison & p-value & Significant \\\\",
            "\\midrule",
        ]
        
        for comp in comparisons:
            name = comp.get("comparison", "")
            p_value = comp.get("p_value", 1.0)
            sig = "Yes" if comp.get("significant", False) else "No"
            
            # Format p-value
            if p_value < 0.001:
                p_str = "$<$0.001"
            else:
                p_str = f"{p_value:.3f}"
            
            lines.append(f"{name} & {p_str} & {sig} \\\\")
        
        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\label{tab:statistical_tests}",
            "\\end{table}",
        ])
        
        return "\n".join(lines)
    
    def save_table(self, table_str: str, filename: str) -> None:
        """Save table to file."""
        if self.output_dir is None:
            logger.warning("output_dir not set")
            return
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(table_str)
        
        logger.info(f"Table saved to {filepath}")
    
    def to_dataframe(self, results: Dict[str, Dict[str, Any]]) -> Optional[Any]:
        """Convert results to pandas DataFrame."""
        if not HAS_PANDAS:
            return None
        
        return pd.DataFrame.from_dict(results, orient='index')
