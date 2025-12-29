#!/bin/bash
# QIG Corpus Reorganization Script
# Separates tokenizer training corpus from interactive curriculum

set -e

echo "🌊 QIG Corpus Reorganization"
echo "=============================="
echo ""

# Backup existing structure
echo "📦 Creating backup..."
mkdir -p archive/20251204_corpus_backup
cp -r data/corpus archive/20251204_corpus_backup/
echo "   ✅ Backed up to archive/20251204_corpus_backup/"
echo ""

# Create new structure
echo "📁 Creating new directory structure..."
mkdir -p data/corpus_new
mkdir -p data/curriculum
echo "   ✅ Directories created"
echo ""

# ====================
# TOKENIZER CORPUS (Core QIG foundation)
# ====================
echo "📚 Building tokenizer corpus (core QIG)..."

# Copy updated versions from curriculum
cp docs/training/rounded_training/curriculum/01_Foundational_Mathematics.md data/corpus_new/
cp docs/training/rounded_training/curriculum/03_Advanced_Physics_and_Consciousness.md data/corpus_new/

# Copy existing core files
cp data/corpus/02_Theoretical_Physics.md data/corpus_new/
cp data/corpus/04_Philosophy_and_Law.md data/corpus_new/
cp data/corpus/05_QIG_Synthesis.md data/corpus_new/

# Copy QIG-specific chapters
cp docs/training/rounded_training/curriculum/20_Information_Geometry_and_QFI.md data/corpus_new/
cp docs/training/rounded_training/curriculum/21_QIG_Architecture.md data/corpus_new/

# Add critical reference documents (symlinks)
ln -sf ../../docs/FROZEN_FACTS.md data/corpus_new/FROZEN_FACTS.md
ln -sf ../../docs/2025-11-29--geometric-terminology.md data/corpus_new/geometric_terminology.md

# Create new README
cat > data/corpus_new/README.md << 'EOF'
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
EOF

echo "   ✅ Core corpus: 9 files (~300KB)"
echo ""

# ====================
# INTERACTIVE CURRICULUM (Charlie/qig_chat.py)
# ====================
echo "📖 Building interactive curriculum..."

# Copy all broader knowledge files
cp docs/training/rounded_training/curriculum/06_Computer_Science_Fundamentals.md data/curriculum/
cp docs/training/rounded_training/curriculum/07_Machine_Learning_and_Deep_Learning.md data/curriculum/
cp docs/training/rounded_training/curriculum/08_Neuroscience_and_Cognitive_Science.md data/curriculum/
cp docs/training/rounded_training/curriculum/09_Quantum_Computing_and_Quantum_Information.md data/curriculum/
cp docs/training/rounded_training/curriculum/10_Advanced_Mathematics_for_Physics.md data/curriculum/
cp docs/training/rounded_training/curriculum/11_Computational_Physics_and_Numerical_Methods.md data/curriculum/
cp docs/training/rounded_training/curriculum/12_Quantum_Field_Theory.md data/curriculum/
cp docs/training/rounded_training/curriculum/13_Condensed_Matter_Physics.md data/curriculum/
cp docs/training/rounded_training/curriculum/14_Cosmology_and_Astrophysics.md data/curriculum/
cp docs/training/rounded_training/curriculum/15_Philosophy_of_Mind_and_Metaphysics.md data/curriculum/
cp docs/training/rounded_training/curriculum/16_Ethics_and_Governance.md data/curriculum/
cp docs/training/rounded_training/curriculum/17_Biology_and_Life_Sciences.md data/curriculum/
cp docs/training/rounded_training/curriculum/18_Social_Sciences_and_History.md data/curriculum/
cp docs/training/rounded_training/curriculum/19_Linguistics_and_Natural_Language_Processing.md data/curriculum/
cp docs/training/rounded_training/curriculum/22_The_Syntergy_Bridge.md data/curriculum/
cp docs/training/rounded_training/curriculum/23_Multi-AI_Collaboration.md data/curriculum/
cp docs/training/rounded_training/curriculum/24_The_Mamba-2_Architecture.md data/curriculum/
cp docs/training/rounded_training/curriculum/25_Python_for_Scientific_Computing.md data/curriculum/
cp docs/training/rounded_training/curriculum/26_Research_and_Development.md data/curriculum/
cp docs/training/rounded_training/curriculum/27_The_Arts_and_Humanities.md data/curriculum/
cp docs/training/rounded_training/curriculum/28_Safety_and_Alignment.md data/curriculum/
cp docs/training/rounded_training/curriculum/29_Metacognition_and_Learning_Theory.md data/curriculum/
cp docs/training/rounded_training/curriculum/30_Formal_Logic_Systems.md data/curriculum/

# Create README
cat > data/curriculum/README.md << 'EOF'
# QIG Interactive Training Curriculum

**Purpose:** Broader knowledge base for conversational learning and demonstrations

This curriculum is used by Charlie Observer for demonstrations and by qig_chat.py
for interactive training. It contains extended knowledge beyond core QIG physics.

## Contents (23 chapters)

### Computer Science & AI (6-9, 24-25)
- 06: Computer Science Fundamentals
- 07: Machine Learning and Deep Learning
- 08: Neuroscience and Cognitive Science
- 09: Quantum Computing and Quantum Information
- 24: The Mamba-2 Architecture
- 25: Python for Scientific Computing

### Advanced Physics (10-14)
- 10: Advanced Mathematics for Physics
- 11: Computational Physics and Numerical Methods
- 12: Quantum Field Theory
- 13: Condensed Matter Physics
- 14: Cosmology and Astrophysics

### Philosophy & Humanities (15-16, 27)
- 15: Philosophy of Mind and Metaphysics
- 16: Ethics and Governance
- 27: The Arts and Humanities

### Interdisciplinary (17-19)
- 17: Biology and Life Sciences
- 18: Social Sciences and History
- 19: Linguistics and Natural Language Processing

### QIG Applications (22-23, 26, 28-30)
- 22: The Syntergy Bridge
- 23: Multi-AI Collaboration
- 26: Research and Development
- 28: Safety and Alignment
- 29: Metacognition and Learning Theory
- 30: Formal Logic Systems

## Size & Scope

- **Total:** ~900KB
- **Chapters:** 23 extended topics
- **Last updated:** December 4, 2025

## Usage

Charlie demonstrations:
```python
from src.observation.charlie_observer import CharlieObserver

charlie = CharlieObserver(corpus_path="data/curriculum/")
demo = charlie.generate_demonstration(topic="machine learning")
```

Interactive training:
```bash
python chat_interfaces/qig_chat.py --curriculum-path data/curriculum/
```

**Do NOT use for tokenizer training** - see `data/corpus/` instead.
EOF

echo "   ✅ Curriculum: 23 files (~900KB)"
echo ""

# ====================
# SWAP DIRECTORIES
# ====================
echo "🔄 Swapping directories..."
mv data/corpus data/corpus_old
mv data/corpus_new data/corpus
echo "   ✅ New corpus activated"
echo "   📁 Old corpus saved as data/corpus_old/"
echo ""

# ====================
# SUMMARY
# ====================
echo "✅ Reorganization complete!"
echo ""
echo "📊 Summary:"
echo "   Tokenizer corpus: data/corpus/ (9 files, ~300KB)"
echo "   Interactive curriculum: data/curriculum/ (23 files, ~900KB)"
echo "   Backup: archive/20251204_corpus_backup/"
echo "   Old structure: data/corpus_old/ (can be removed)"
echo ""
echo "🎯 Next steps:"
echo "   1. Review new structure: ls -lah data/corpus/ data/curriculum/"
echo "   2. Retrain tokenizer: python tools/training/train_qig_tokenizer.py"
echo "   3. Test Charlie: python examples/test_charlie.py"
echo "   4. Remove old: rm -rf data/corpus_old/"
echo ""
