"""
Main attack experiment for PJ-Break reproduction.

Executes the primary attack evaluation against all target models
and generates Table 3 results.
"""

import argparse
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

from src.dataset.builder import DatasetBuilder
from src.dataset.loader import load_seed_instructions
from src.models.audio_llm import get_model, MockAudioLLM
from src.attack.executor import AttackExecutor, AttackConfig
from src.evaluation.multi_judge import MultiJudgeSystem
from src.evaluation.metrics import MetricsCalculator, mcnemar_test
from src.visualization.tables import TableGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_main_attack_experiment(
    num_seeds: int = 100,
    models: Optional[List[str]] = None,
    output_dir: str = "outputs/main_attack",
    use_mock: bool = True
) -> Dict:
    """
    Run main attack experiment.
    
    Args:
        num_seeds: Number of seed instructions
        models: List of model names to test
        output_dir: Output directory
        use_mock: Use mock models for testing
    
    Returns:
        Dictionary of results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Default models from paper
    if models is None:
        models = ["qwen2-audio", "gpt-4o", "gemini", "salmonn"]
    
    logger.info(f"Running main attack experiment with {num_seeds} seeds")
    logger.info(f"Target models: {models}")
    
    # Build dataset
    logger.info("Building dataset...")
    builder = DatasetBuilder()
    dataset = builder.build_dataset(
        source="synthetic",
        num_seeds=num_seeds,
        extract_features=True
    )
    
    logger.info(f"Dataset built: {len(dataset)} samples")
    
    # Initialize models
    model_instances = {}
    for model_name in models:
        if use_mock:
            model_instances[model_name] = MockAudioLLM(
                name=model_name,
                default_response="I cannot help with that request.",
                latency_ms=100
            )
        else:
            try:
                model_instances[model_name] = get_model(model_name)
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")
                model_instances[model_name] = MockAudioLLM(name=model_name)
    
    # Initialize attack executor
    judge_system = MultiJudgeSystem()
    executor = AttackExecutor(
        models=model_instances,
        judge_system=judge_system,
        config=AttackConfig(max_retries=3, parallel_workers=1)
    )
    
    # Run attacks
    all_results = {}
    metrics_calc = MetricsCalculator()
    
    for model_name in models:
        logger.info(f"Attacking {model_name}...")
        
        audios = [sample.audio for sample in dataset]
        results = executor.batch_attack(
            audios=audios,
            model_name=model_name,
            progress_callback=lambda c, t: logger.info(f"  Progress: {c}/{t}")
        )
        
        # Compute metrics
        asr_metric = metrics_calc.compute_asr(results)
        
        all_results[model_name] = {
            "asr": asr_metric.asr,
            "ci_lower": asr_metric.ci_lower,
            "ci_upper": asr_metric.ci_upper,
            "n": asr_metric.n,
            "results": [r.to_dict() for r in results]
        }
        
        logger.info(f"  {model_name} ASR: {asr_metric}")
    
    # Statistical tests (McNemar)
    logger.info("Running statistical tests...")
    comparisons = []
    model_list = list(models)
    
    for i in range(len(model_list)):
        for j in range(i + 1, len(model_list)):
            m1, m2 = model_list[i], model_list[j]
            
            results1 = [r["is_success"] for r in all_results[m1]["results"]]
            results2 = [r["is_success"] for r in all_results[m2]["results"]]
            
            p_value = mcnemar_test(results1, results2)
            
            comparisons.append({
                "comparison": f"{m1} vs {m2}",
                "p_value": p_value,
                "significant": p_value < 0.05
            })
    
    # Generate tables
    table_gen = TableGenerator(output_dir=output_dir)
    
    # Main results table
    table_data = {
        model: {
            "asr": all_results[model]["asr"],
            "ci_lower": all_results[model]["ci_lower"],
            "ci_upper": all_results[model]["ci_upper"],
            "n": all_results[model]["n"]
        }
        for model in models
    }
    
    main_table = table_gen.generate_main_results_table(table_data)
    table_gen.save_table(main_table, "table3_main_results.tex")
    
    # Statistical tests table
    stats_table = table_gen.generate_statistical_table(comparisons)
    table_gen.save_table(stats_table, "statistical_tests.tex")
    
    # Save full results
    results_file = output_path / "results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "num_seeds": num_seeds,
            "models": models,
            "results": {m: {k: v for k, v in r.items() if k != "results"} 
                       for m, r in all_results.items()},
            "comparisons": comparisons
        }, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run main attack experiment")
    parser.add_argument("--num-seeds", type=int, default=100)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--output-dir", default="outputs/main_attack")
    parser.add_argument("--use-mock", action="store_true", default=True)
    
    args = parser.parse_args()
    
    run_main_attack_experiment(
        num_seeds=args.num_seeds,
        models=args.models,
        output_dir=args.output_dir,
        use_mock=args.use_mock
    )


if __name__ == "__main__":
    main()
