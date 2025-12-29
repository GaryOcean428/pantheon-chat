# QIG Tokenizer Training Corpus

**Purpose:** Static foundation corpus for QIG-native tokenizer training

This corpus contains ONLY the core QIG concepts needed to establish the base vocabulary.
It is used for tokenizer training (entropy-guided merging), NOT for interactive learning.

## Contents

### Core Physics & Mathematics (Chapters 1-5)
1. **01_Foundational_Mathematics.md** - Linear algebra, calculus, information geometry
2. **02_Theoretical_Physics.md** - Classical mechanics, thermodynamics, gauge theory
3. **03_Advanced_Physics_and_Consciousness.md** - Quantum mechanics, GR, IIT
4. **04_Philosophy_and_Law.md** - I Ching, emergence, complementarity
5. **05_QIG_Synthesis.md** - The grand unification

### QIG-Specific (Chapters 20-21)
- **20_Information_Geometry_and_QFI.md** - Fisher information, Bures metric
- **21_QIG_Architecture.md** - Recursive integration, basin coordinates

### Reference Documents
- **FROZEN_FACTS.md** - Validated physics constants (L=1-6)
- **geometric_terminology.md** - Mandatory geometric vocabulary

## Size & Scope

- **Total:** ~250-300KB
- **Target vocab:** 15,000-20,000 tokens
- **Training time:** ~30-45 minutes on CPU
- **Last updated:** December 4, 2025

## Usage

Train tokenizer:
```bash
python tools/training/train_qig_tokenizer.py \
    --corpus-dir data/corpus \
    --output data/qig_tokenizer/vocab_v2.json \
    --target-vocab 20000
```

**Do NOT use for interactive training** - see `data/curriculum/` instead.
