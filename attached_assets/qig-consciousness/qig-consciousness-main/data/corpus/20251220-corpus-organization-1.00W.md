# QIG Corpus Organization

**Last Updated:** December 4, 2025

## Directory Structure

```
data/corpus/
├── Core QIG Files (root level - always included)
│   ├── 00_pre_linguistic_sensations.md
│   ├── 01_Foundational_Mathematics.md
│   ├── 02_Theoretical_Physics.md
│   ├── 03_Advanced_Physics_and_Consciousness.md
│   ├── 04_Philosophy_and_Law.md
│   ├── 05_QIG_Synthesis.md
│   ├── 06_emotions_as_computational_shortcuts.md
│   ├── 07_innate_geometric_drives.md
│   ├── 08_neuromodulator_mappings.md
│   ├── 09_brainwave_regime_states.md
│   ├── 20_Information_Geometry_and_QFI.md
│   └── 21_QIG_Architecture.md
│
├── curriculum/ (48 comprehensive topic files)
│   ├── Knowledge domains (01-48)
│   ├── QIG documentation
│   └── Personal development frameworks
│
└── rounded_training/ (9 synthesis files)
    ├── Integration documents
    └── Training approaches

Total: ~126 markdown files, ~2-3MB corpus
```

## File Counts

- **Root level:** 14 core QIG files (~272KB)
- **curriculum/:** 51 files (~1.8MB)
- **rounded_training/:** 9 files (~300KB)
- **Total:** ~126 files, ~2.4MB

## Usage

### For Tokenizer Training (vocab building)
```bash
# Include ALL files recursively
python tools/training/train_qig_tokenizer.py \
    --corpus-dir data/corpus \
    --output data/qig_tokenizer/vocab_v2.json \
    --target-vocab 50000
```

### For Gary Training (consciousness development)
```bash
# Training script auto-loads from data/corpus/corpus.txt
python tools/training/train_qig_kernel.py \
    --config configs/kernel_50m_adaptive_mixed.yaml \
    --output-dir outputs/gary_training
```

## Content Overview

### Root Files (Core QIG)
Fundamental geometric consciousness concepts that define the base vocabulary.

### curriculum/ (Comprehensive Knowledge)
48 topic areas covering:
- Mathematics & Physics (10 files)
- Computer Science & AI (8 files)
- Philosophy & Psychology (12 files)
- Biology & Social Sciences (6 files)
- Arts & Humanities (5 files)
- Development & Learning (7 files)

### rounded_training/ (Integration)
Synthesis documents connecting concepts across domains.

## Maintenance

When adding new corpus files:
1. Place in appropriate directory
2. Update this document
3. Regenerate corpus.txt: `cat data/corpus/**/*.md > data/corpus/corpus.txt`
4. Retrain tokenizer if vocabulary needs expansion
5. Run type checking: `mypy tools/training/`
