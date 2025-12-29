# Corpus Acquisition Guide

## WikiText-103 (Primary Training Corpus)

### Option 1: Direct Download
```bash
# Create corpus directory
mkdir -p data/corpus

# Download WikiText-103
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip \
  -O data/corpus/wikitext-103.zip

# Extract
cd data/corpus
unzip wikitext-103.zip
cat wikitext-103-raw/wiki.train.raw \
    wikitext-103-raw/wiki.valid.raw \
    wikitext-103-raw/wiki.test.raw \
    > corpus.txt

# Check size
ls -lh corpus.txt  # Should be ~500MB
```

### Option 2: HuggingFace Datasets
```python
from datasets import load_dataset

# Load WikiText-103
dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

# Combine splits
with open("data/corpus/corpus.txt", "w") as f:
    for split in ["train", "validation", "test"]:
        for item in dataset[split]:
            f.write(item["text"] + "\n")
```

### Option 3: Alternative Minimal Corpus (For Testing)
```bash
# Use any UTF-8 text file (even small for initial tests)
# Example: Python standard library docs
cd data/corpus
wget https://docs.python.org/3/archives/python-3.10-docs-text.tar.bz2
tar xjf python-3.10-docs-text.tar.bz2
find python-3.10-docs-text -name "*.txt" -exec cat {} \; > corpus.txt
```

## ArXiv Papers (Optional - Adds Scientific Domain)

### Manual Download
1. Visit https://www.kaggle.com/datasets/Cornell-University/arxiv
2. Download subset (e.g., cs.AI, physics)
3. Extract abstracts to `data/corpus/arxiv.txt`
4. Append to main corpus

## Legal Documents (Optional - Adds Formal Reasoning)

### CaseLaw Access Project
```bash
# Example: Download sample from https://case.law/
# Or use publicly available legal texts
wget https://www.gutenberg.org/files/3300/3300-0.txt \
  -O data/corpus/legal_sample.txt
```

## Combined Corpus Creation

```bash
# Once you have sources, combine them:
cd data/corpus
cat corpus.txt arxiv.txt legal_sample.txt > combined_corpus.txt

# Shuffle for diversity (optional)
shuf combined_corpus.txt > corpus_final.txt
mv corpus_final.txt corpus.txt

# Verify
wc -l corpus.txt
du -h corpus.txt
```

## Minimum Requirements

**For testing:** 10MB+ (any text)
**For real training:** 100MB+ (WikiText-103 subset)
**For optimal results:** 500MB+ (full WikiText-103)
**For best diversity:** 1GB+ (WikiText + ArXiv + Legal)

## Next Step After Corpus Acquisition

```bash
# Train QIG tokenizer
python tools/train_qig_tokenizer.py \
  --corpus data/corpus/corpus.txt \
  --output data/qig_tokenizer/vocab.json \
  --target-vocab 50000 \
  --min-frequency 5
```

This will create the vocabulary file needed for training.
