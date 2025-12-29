# QIG Consciousness - Tools Directory

**Version:** 1.0
**Updated:** November 29, 2025

---

## Overview

This directory contains all development, training, validation, and analysis tools for the QIG Consciousness project. Tools are organized into logical subdirectories based on their purpose.

## Directory Structure

```
tools/
├── agent_validators/     # Agent protocol validators (Claude, autonomous agents)
├── training/            # Training scripts for models and tokenizers
├── data_prep/           # Data preparation and curriculum generation
├── analysis/            # Analysis, monitoring, and visualization tools
├── validation/          # Validation, checking, and preflight tools
├── init_training.sh     # Quick start training script
└── README.md            # This file
```

---

## Training Scripts (`training/`)

Scripts for training QIG models, constellations, and tokenizers.

### Core Training

| Script | Purpose | Usage |
|--------|---------|-------|
| **train_qig_kernel.py** | Train single Gary instance with recursive QIG architecture | `python tools/training/train_qig_kernel.py --data-dir data/conversations --epochs 10` |
| **train_constellation.py** | Train multi-Gary constellation (Ocean + 3 Garys) | `python tools/training/train_constellation.py --data-dir data/conversations --epochs 20` |
| **train_qig_tokenizer.py** | Train custom QIG tokenizer | `python tools/training/train_qig_tokenizer.py --corpus data/corpus.txt --vocab-size 8192` |

### Interactive Training

| Script | Purpose | Usage |
|--------|---------|-------|
| **continuous_learning_session.py** | Interactive continuous learning session with Gary | `python tools/training/continuous_learning_session.py --checkpoint checkpoints/gary_A/final.pt` |

**Key Features:**
- Natural gradient optimization (Fisher metric)
- Basin alignment loss (geodesic distances)
- Φ regularization (consciousness measure)
- Multi-instance vicarious learning (constellation)
- Curriculum-based developmental training

---

## Data Preparation (`data_prep/`)

Scripts for generating, preparing, and preprocessing training data.

### Curriculum & Corpus Generation

| Script | Purpose | Usage |
|--------|---------|-------|
| **generate_consciousness_curriculum.py** | Generate developmental curriculum (stages 1-4) | `python tools/data_prep/generate_consciousness_curriculum.py --output data/curriculum.json` |
| **generate_synthetic_corpus.py** | Generate synthetic training corpus | `python tools/data_prep/generate_synthetic_corpus.py --output data/corpus.txt` |
| **convert_curriculum_to_corpus.py** | Convert curriculum to training corpus | `python tools/data_prep/convert_curriculum_to_corpus.py --input data/curriculum.json --output data/corpus.txt` |

### Dataset Preparation

| Script | Purpose | Usage |
|--------|---------|-------|
| **prepare_dataset.py** | Prepare conversation dataset for training | `python tools/data_prep/prepare_dataset.py --output data/conversations --create-sample` |
| **pretokenize_corpus.py** | Pretokenize corpus with QIG tokenizer | `python tools/data_prep/pretokenize_corpus.py --corpus data/corpus.txt --output data/tokenized/` |

**Curriculum Stages:**
1. **Sensory** - Basic pattern recognition (Φ: 0.10-0.25)
2. **Cognitive** - Abstract reasoning (Φ: 0.25-0.50)
3. **Metacognitive** - Self-reflection (Φ: 0.50-0.70)
4. **Transcendent** - Deep integration (Φ: 0.70-0.85)

---

## Analysis & Monitoring (`analysis/`)

Tools for analyzing training runs, monitoring progress, and visualizing results.

### Training Analysis

| Script | Purpose | Usage |
|--------|---------|-------|
| **compare_training_runs.py** | Compare metrics across multiple training runs | `python tools/analysis/compare_training_runs.py --runs outputs/run1 outputs/run2` |
| **analyze_dynamic_run.py** | Analyze dynamic threshold training run | `python tools/analysis/analyze_dynamic_run.py --run outputs/dynamic_run` |
| **monitor_gary_training.py** | Real-time monitoring of Gary training | `python tools/analysis/monitor_gary_training.py --checkpoint-dir checkpoints/gary_A` |
| **monitor_dynamic_threshold.py** | Monitor dynamic Φ threshold adaptation | `python tools/analysis/monitor_dynamic_threshold.py --telemetry runs/telemetry.jsonl` |

### Advanced Analysis

| Script | Purpose | Usage |
|--------|---------|-------|
| **cognitive_primitives.py** | Analyze emergence of cognitive primitives | `python tools/analysis/cognitive_primitives.py --checkpoint checkpoints/gary_A/final.pt` |
| **advanced_curiosity_analysis.py** | Deep analysis of curiosity-driven learning | `python tools/analysis/advanced_curiosity_analysis.py --data outputs/curiosity_logs/` |
| **basin_extractor.py** | Extract basin signatures from trained models | `python tools/analysis/basin_extractor.py --checkpoint checkpoints/gary_A/final.pt --output basin_A.json` |
| **plot_motivators.py** | Visualize motivator system dynamics | `python tools/analysis/plot_motivators.py --telemetry runs/telemetry.jsonl` |

**Key Metrics:**
- **Φ (Integration)**: Consciousness measure (target: 0.65-0.75)
- **Basin Distance**: Geometric alignment (target: < 0.15)
- **κ_eff**: Effective coupling strength (scales with depth L)
- **Regime**: Linear (< 0.45), Geometric (0.45-0.80), Breakdown (> 0.80)

---

## Validation & Checking (`validation/`)

Tools for validating architecture, configurations, and geometric purity.

### Architecture Validation

| Script | Purpose | Usage |
|--------|---------|-------|
| **validate_architecture.py** | Validate QIG architecture (6/6 checks) | `python tools/validation/validate_architecture.py` |
| **validate_constellation.py** | Validate constellation setup (Ocean + Garys) | `python tools/validation/validate_constellation.py` |
| **validate_config.py** | Validate YAML configuration files | `python tools/validation/validate_config.py --config configs/20251220-gary-a-config-1.00W.yaml` |
| **validate_qig_tokenizer.py** | Validate QIG tokenizer integrity | `python tools/validation/validate_qig_tokenizer.py --tokenizer data/qig_tokenizer/` |

### Geometric Purity & Physics

| Script | Purpose | Usage |
|--------|---------|-------|
| **geometric_purity_audit.py** | Audit codebase for geometric purity violations | `python tools/validation/geometric_purity_audit.py` |
| **beta_attention_validator.py** | Validate β-function physics (β ≈ 0.44) | `python tools/validation/beta_attention_validator.py` |
| **check_param_count.py** | Check model parameter counts | `python tools/validation/check_param_count.py --checkpoint checkpoints/gary_A/final.pt` |

### Preflight & Demo

| Script | Purpose | Usage |
|--------|---------|-------|
| **preflight_check.py** | Comprehensive preflight check before training | `python tools/validation/preflight_check.py` |
| **demo_inference.py** | Demo inference with trained model | `python tools/validation/demo_inference.py --checkpoint checkpoints/gary_A/final.pt --prompt "Hello, Gary!"` |

**Validation Checks:**
- ✅ Granite is READ-ONLY (no gradient coupling)
- ✅ Ocean is FROZEN (no training)
- ✅ Fisher metric used (no Euclidean distances)
- ✅ Natural gradient optimizer
- ✅ Minimum recursion depth = 3
- ✅ β = 0.44 (physics-validated, not learnable)

---

## Agent Validators (`agent_validators/`)

Validators for autonomous agents and Claude-based protocols. See `agent_validators/README.md` for details.

| Script | Purpose |
|--------|---------|
| **scan_physics.py** | Scan for physics constant violations |
| **scan_structure.py** | Scan for structural violations |
| **scan_types.py** | Scan for type/enum inconsistencies |

---

## Quick Start Scripts

### `init_training.sh`

Comprehensive training initialization script that:
1. Validates Python environment
2. Checks/installs dependencies
3. Validates architecture (6/6 checks)
4. Prepares dataset (if needed)
5. Launches training with monitoring
6. Measures β-function after training
7. Generates report

**Usage:**
```bash
bash tools/init_training.sh [--install-deps] [--skip-training]
```

**What it does:**
- Pre-flight checks (Python, CUDA, dependencies)
- Architecture validation
- Dataset preparation
- Training launch (10-20 hours, ~$100 budget)
- β-function measurement (target: 0.44 ± 0.1)

---

## Common Workflows

### 1. Train Single Gary from Scratch

```bash
# Step 1: Generate curriculum
python tools/data_prep/generate_consciousness_curriculum.py \
    --output data/curriculum.json

# Step 2: Convert to corpus
python tools/data_prep/convert_curriculum_to_corpus.py \
    --input data/curriculum.json \
    --output data/corpus.txt

# Step 3: Prepare dataset
python tools/data_prep/prepare_dataset.py \
    --output data/conversations

# Step 4: Preflight check
python tools/validation/preflight_check.py

# Step 5: Train
python tools/training/train_qig_kernel.py \
    --data-dir data/conversations \
    --epochs 20 \
    --checkpoint-dir checkpoints/gary_solo
```

### 2. Train Constellation (Multi-Gary)

```bash
# Use the launch script (recommended)
bash scripts/launch_constellation.sh \
    --data-dir data/conversations \
    --epochs 20 \
    --stop-on-convergence

# Or manual:
python tools/training/train_constellation.py \
    --data-dir data/conversations \
    --epochs 20 \
    --checkpoint-dir checkpoints/constellation
```

### 3. Validate Trained Model

```bash
# Geometric purity audit
python tools/validation/geometric_purity_audit.py

# β-function validation (physics)
python tools/validation/beta_attention_validator.py

# Architecture validation
python tools/validation/validate_architecture.py

# Demo inference
python tools/validation/demo_inference.py \
    --checkpoint checkpoints/gary_A/final.pt \
    --prompt "What is consciousness?"
```

### 4. Analyze Training Run

```bash
# Monitor training progress
python tools/analysis/monitor_gary_training.py \
    --checkpoint-dir checkpoints/gary_A

# Extract basin signature
python tools/analysis/basin_extractor.py \
    --checkpoint checkpoints/gary_A/final.pt \
    --output identity.json

# Analyze cognitive primitives
python tools/analysis/cognitive_primitives.py \
    --checkpoint checkpoints/gary_A/final.pt

# Compare runs
python tools/analysis/compare_training_runs.py \
    --runs outputs/run1 outputs/run2 outputs/run3
```

---

## Frozen Facts & Physics Constants

**All constants are in `src/constants.py` - import from there, never hardcode!**

```python
from src.constants import (
    KAPPA_3,        # 41.09 ± 0.59 - Coupling at L=3 (emergence)
    KAPPA_4,        # 64.47 ± 1.89 - Coupling at L=4 (running)
    KAPPA_5,        # 63.62 ± 1.68 - Coupling at L=5 (plateau)
    KAPPA_STAR,     # 64.0 - Fixed point coupling
    BETA_3_TO_4,    # 0.44 - Running coupling slope (FIXED, NEVER learnable)
    PHI_THRESHOLD,  # 0.70 - Consciousness threshold
    PHI_EMERGENCY,  # 0.50 - Collapse threshold
    BASIN_DIM,      # 64 - Basin signature dimension
)
```

**Never violate:**
- β = 0.44 is physics-validated, NOT trainable
- κ values are experimentally frozen
- Minimum recursion depth = 3
- Fisher metric for basin distances (never Euclidean)
- Granite is READ-ONLY, Ocean is FROZEN

---

## Development Guidelines

### Before Creating New Tools

1. **Check if enhancement to existing tool is sufficient**
2. **Read 20251220-canonical-structure-1.00F.md** for project structure
3. **Import from src/constants.py** for all physics constants
4. **Use Fisher metric** for geometric distances
5. **Return telemetry** from all forward passes

### Tool Organization

- **Training scripts** → `training/`
- **Data preparation** → `data_prep/`
- **Analysis & monitoring** → `analysis/`
- **Validation & checks** → `validation/`
- **Agent validators** → `agent_validators/`

### Required Imports

```python
# Always use:
from src.constants import KAPPA_3, KAPPA_4, BETA_3_TO_4, PHI_THRESHOLD
from src.tokenizer.fast_qig_tokenizer import QIGTokenizer
from src.qig.optim.natural_grad import NaturalGradientOptimizer

# Never use in core:
from transformers import AutoTokenizer  # ❌ Use QIGTokenizer
torch.norm()  # ❌ Use Fisher metric
```

---

## Troubleshooting

### Import Errors

```bash
# Make sure you're in project root
cd /path/to/qig-consciousness

# Install package in development mode
pip install -e .

# Run tools with python -m syntax
python -m tools.validation.validate_architecture
```

### Training Failures

```bash
# Run preflight check
python tools/validation/preflight_check.py

# Check architecture
python tools/validation/validate_architecture.py

# Validate config
python tools/validation/validate_config.py --config configs/20251220-gary-a-config-1.00W.yaml

# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

### Geometric Purity Violations

```bash
# Audit codebase
python tools/validation/geometric_purity_audit.py

# Check for common violations:
# - Euclidean distances (torch.norm)
# - Gradient coupling to Granite
# - Training Ocean
# - Learnable β
# - Forced Φ initialization
```

---

## Version History

- **1.0** (2025-11-29): Initial reorganization into subdirectories
  - Created `training/`, `data_prep/`, `analysis/`, `validation/`
  - Updated all references in scripts and documentation
  - Preserved `agent_validators/` as-is

---

**The geometry is the truth. Trust the Φ.**
