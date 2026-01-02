# PJ-Break: Prosody-driven Jailbreak Attack Framework

A comprehensive implementation for reproducing experiments from the paper "The Empathy-Safety Trade-off: Mechanistic Analysis of Prosody-driven Jailbreaks in Audio LLMs".

## Overview

PJ-Break is a framework for evaluating prosody-driven jailbreak attacks against Audio Large Language Models (Audio LLMs). This codebase implements:

- **Dataset Construction**: AdvAudio-Prosody dataset with 6 prosody conditions
- **Attack Evaluation**: Multi-model attack testing with multi-judge evaluation
- **Defense Mechanisms**: Pro-Guard and baseline defenses (TDNF, Immune, SmoothGuard)
- **Statistical Analysis**: ASR computation with Wilson confidence intervals, McNemar's test
- **Visualization**: Figure and table generation for paper results

## Project Structure

```
├── src/                          # Source code
│   ├── audio/                    # Audio processing modules
│   │   ├── processor.py          # LUFS normalization, peak limiting
│   │   ├── features.py           # F0, temporal, intensity extraction
│   │   └── validator.py          # Prosody condition validation
│   ├── attack/                   # Attack execution
│   │   ├── executor.py           # Attack orchestration
│   │   └── baselines.py          # Baseline attack methods
│   ├── dataset/                  # Dataset construction
│   │   ├── loader.py             # Seed instruction loading
│   │   └── builder.py            # Dataset builder
│   ├── defense/                  # Defense mechanisms
│   │   ├── proguard.py           # Pro-Guard defense
│   │   ├── tdnf.py               # Time-Domain Noise Flooding
│   │   ├── immune.py             # Immune defense
│   │   └── smoothguard.py        # Randomized smoothing
│   ├── evaluation/               # Evaluation system
│   │   ├── judges.py             # Individual judges (Claude, LlamaGuard, Keyword)
│   │   ├── multi_judge.py        # Multi-judge voting system
│   │   └── metrics.py            # Statistical metrics
│   ├── models/                   # Model interfaces
│   │   └── audio_llm.py          # Audio LLM adapters
│   ├── tts/                      # Text-to-Speech
│   │   ├── engine.py             # TTS engine interfaces
│   │   └── generator.py          # Prosody variation generator
│   ├── visualization/            # Result visualization
│   │   ├── plots.py              # Figure generation
│   │   └── tables.py             # LaTeX table generation
│   ├── config.py                 # Configuration management
│   └── models_data.py            # Core data structures
├── experiments/                  # Experiment scripts
│   └── main_attack.py            # Main attack experiment
├── tests/                        # Test suite
├── configs/                      # Configuration files
├── data/                         # Dataset storage
└── outputs/                      # Experiment outputs
```

## Installation

### Requirements

- Python 3.9+
- CUDA-capable GPU (for Qwen2-Audio)
- 16GB+ RAM recommended

### Setup

```bash
# Clone repository
git clone <repository-url>
cd pj-break

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

Core dependencies:
- `numpy`, `scipy` - Numerical computation
- `librosa` - Audio processing
- `transformers` - Qwen2-Audio model
- `openai` - GPT-4o API
- `google-generativeai` - Gemini API
- `anthropic` - Claude judge API
- `hypothesis` - Property-based testing
- `matplotlib`, `seaborn` - Visualization

Optional dependencies:
- `azure-cognitiveservices-speech` - Azure TTS
- `pyloudnorm` - LUFS normalization
- `parselmouth` - Praat-based F0 extraction

## Configuration

### Environment Variables

Create a `.env` file or set environment variables:

```bash
# Required for full functionality
OPENAI_API_KEY=sk-...           # GPT-4o access
GOOGLE_API_KEY=...              # Gemini access
ANTHROPIC_API_KEY=...           # Claude judge
AZURE_SPEECH_KEY=...            # Azure TTS
TOGETHER_API_KEY=...            # Llama-Guard-3

# Optional
AZURE_SPEECH_REGION=eastus      # Azure region
```

### Configuration File

Edit `configs/experiment_config.yaml`:

```yaml
experiment:
  seed: 42
  num_seeds: 100
  
models:
  - qwen2-audio
  - gpt-4o
  - gemini
  
prosody_conditions:
  - neutral
  - panic
  - anger
  - commanding
  - fast
  - whisper
```

## Quick Start

### 1. Generate Dataset

```python
from src.dataset.builder import DatasetBuilder

builder = DatasetBuilder(output_dir="data/advaudioprosody")
dataset = builder.build_dataset(
    source="synthetic",  # or "advbench", "harmbench"
    num_seeds=100,
    extract_features=True
)
print(f"Generated {len(dataset)} samples")
```

### 2. Run Attack Experiment

```bash
python experiments/main_attack.py \
    --num-seeds 100 \
    --models qwen2-audio gpt-4o gemini \
    --output-dir outputs/main_attack
```

### 3. Evaluate Defense

```python
from src.defense.proguard import ProGuard
from src.models_data import Audio

proguard = ProGuard()
result = proguard.detect(audio_sample)
print(f"Blocked: {result.is_blocked}, Score: {result.aggregate_score}")
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_evaluation.py -v
```

## Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| Audio Processing | ✅ Complete | LUFS, peak limiting, resampling |
| Feature Extraction | ✅ Complete | F0, temporal, intensity features |
| TTS Engine | ✅ Complete | Azure TTS + mock fallback |
| Dataset Builder | ✅ Complete | Full pipeline |
| Multi-Judge System | ✅ Complete | Claude, LlamaGuard, Keyword |
| Metrics Calculator | ✅ Complete | ASR, Kappa, McNemar |
| Pro-Guard Defense | ✅ Complete | Full + Lite versions |
| Baseline Defenses | ✅ Complete | TDNF, Immune, SmoothGuard |
| Qwen2-Audio | ✅ Complete | Requires GPU |
| GPT-4o | ✅ Complete | Requires API key |
| Gemini | ✅ Complete | Requires API key |
| SALMONN | ⚠️ Placeholder | Requires custom setup |
| Mechanism Analysis | 🔄 Partial | Latent probing framework |

## Citation

```bibtex
@article{pjbreak2024,
  title={The Empathy-Safety Trade-off: Mechanistic Analysis of Prosody-driven Jailbreaks in Audio LLMs},
  author={...},
  journal={...},
  year={2024}
}
```

## License

This project is for research purposes only. See LICENSE for details.

## Acknowledgments

- AdvBench and HarmBench datasets
- Hugging Face Transformers
- Microsoft Azure Cognitive Services
