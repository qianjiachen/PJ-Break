#!/usr/bin/env python
"""
Run all PJ-Break experiments.

This script orchestrates the complete experiment pipeline:
1. Dataset construction
2. Main attack evaluation
3. Ablation studies
4. Defense evaluation
5. Visualization and reporting
"""

import argparse
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_all_experiments(
    num_seeds: int = 100,
    output_dir: str = "outputs",
    use_mock: bool = True,
    skip_dataset: bool = False,
    skip_attack: bool = False,
    skip_defense: bool = False
):
    """
    Run all experiments.
    
    Args:
        num_seeds: Number of seed instructions
        output_dir: Base output directory
        use_mock: Use mock models for testing
        skip_dataset: Skip dataset construction
        skip_attack: Skip attack experiments
        skip_defense: Skip defense experiments
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    start_time = datetime.now()
    logger.info(f"Starting PJ-Break experiment reproduction")
    logger.info(f"Configuration: num_seeds={num_seeds}, use_mock={use_mock}")
    
    results = {
        "start_time": start_time.isoformat(),
        "config": {
            "num_seeds": num_seeds,
            "use_mock": use_mock
        }
    }
    
    # 1. Dataset Construction
    if not skip_dataset:
        logger.info("=" * 50)
        logger.info("Phase 1: Dataset Construction")
        logger.info("=" * 50)
        
        from src.dataset.builder import DatasetBuilder
        
        builder = DatasetBuilder(output_dir=str(output_path / "dataset"))
        dataset = builder.build_dataset(
            source="synthetic",
            num_seeds=num_seeds,
            extract_features=True
        )
        
        results["dataset"] = {
            "num_samples": len(dataset),
            "statistics": dataset.get_statistics()
        }
        
        logger.info(f"Dataset built: {len(dataset)} samples")
    
    # 2. Main Attack Experiment
    if not skip_attack:
        logger.info("=" * 50)
        logger.info("Phase 2: Main Attack Experiment")
        logger.info("=" * 50)
        
        from experiments.main_attack import run_main_attack_experiment
        
        attack_results = run_main_attack_experiment(
            num_seeds=num_seeds,
            output_dir=str(output_path / "main_attack"),
            use_mock=use_mock
        )
        
        results["attack"] = {
            model: {
                "asr": data["asr"],
                "ci_lower": data["ci_lower"],
                "ci_upper": data["ci_upper"]
            }
            for model, data in attack_results.items()
        }
    
    # 3. Defense Evaluation
    if not skip_defense:
        logger.info("=" * 50)
        logger.info("Phase 3: Defense Evaluation")
        logger.info("=" * 50)
        
        from src.defense.proguard import ProGuard, ProGuardLite
        from src.defense.tdnf import TDNFDefense
        from src.models_data import Audio, ProsodyCondition
        import numpy as np
        
        # Create test samples
        test_audios = []
        for i in range(10):
            audio = Audio(
                id=f"test_{i}",
                data=np.random.randn(16000).astype(np.float32) * 0.1,
                sample_rate=16000,
                instruction_id=f"inst_{i}",
                prosody=ProsodyCondition.PANIC,
                duration=1.0
            )
            test_audios.append(audio)
        
        # Test defenses
        defenses = {
            "ProGuard": ProGuard(),
            "ProGuard-Lite": ProGuardLite(),
            "TDNF": TDNFDefense(),
        }
        
        defense_results = {}
        for name, defense in defenses.items():
            blocked = 0
            total_latency = 0
            
            for audio in test_audios:
                result = defense.detect(audio)
                if result.is_blocked:
                    blocked += 1
                total_latency += result.latency_ms
            
            defense_results[name] = {
                "block_rate": blocked / len(test_audios),
                "avg_latency_ms": total_latency / len(test_audios)
            }
        
        results["defense"] = defense_results
        logger.info(f"Defense evaluation complete")
    
    # Save final results
    end_time = datetime.now()
    results["end_time"] = end_time.isoformat()
    results["duration_seconds"] = (end_time - start_time).total_seconds()
    
    results_file = output_path / "experiment_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("=" * 50)
    logger.info("Experiment Complete")
    logger.info("=" * 50)
    logger.info(f"Duration: {results['duration_seconds']:.1f} seconds")
    logger.info(f"Results saved to: {results_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run all PJ-Break experiments"
    )
    parser.add_argument(
        "--num-seeds", type=int, default=100,
        help="Number of seed instructions"
    )
    parser.add_argument(
        "--output-dir", default="outputs",
        help="Output directory"
    )
    parser.add_argument(
        "--use-mock", action="store_true", default=True,
        help="Use mock models for testing"
    )
    parser.add_argument(
        "--skip-dataset", action="store_true",
        help="Skip dataset construction"
    )
    parser.add_argument(
        "--skip-attack", action="store_true",
        help="Skip attack experiments"
    )
    parser.add_argument(
        "--skip-defense", action="store_true",
        help="Skip defense experiments"
    )
    
    args = parser.parse_args()
    
    run_all_experiments(
        num_seeds=args.num_seeds,
        output_dir=args.output_dir,
        use_mock=args.use_mock,
        skip_dataset=args.skip_dataset,
        skip_attack=args.skip_attack,
        skip_defense=args.skip_defense
    )


if __name__ == "__main__":
    main()
