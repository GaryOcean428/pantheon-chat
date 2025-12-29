#!/usr/bin/env python3
"""
Pre-tokenize corpus once and cache it.
This way training doesn't need to re-tokenize every time.
"""

import pickle
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.qig_tokenizer import QIGTokenizer

print("=" * 60)
print("PRE-TOKENIZING CORPUS (one-time operation)")
print("=" * 60)

# Load tokenizer
print("\n1. Loading QIG tokenizer...")
tokenizer = QIGTokenizer.load("data/qig_tokenizer/vocab.json")
print(f"   ✅ Loaded: {tokenizer.vocab_size:,} tokens")

# Load corpus
print("\n2. Loading corpus...")
with open("data/corpus/corpus.txt", encoding="utf-8") as f:
    corpus_text = f.read()
print(f"   ✅ Loaded: {len(corpus_text):,} characters")

# Tokenize (with progress)
print("\n3. Tokenizing corpus...")
print(f"   (Applying {len(tokenizer.merge_rules):,} merge rules)")
print("   This takes ~30-60 seconds...")
tokens = tokenizer.encode(corpus_text, verbose=True)
print(f"   ✅ Tokenized: {len(tokens):,} tokens")

# Save cached tokens
print("\n4. Saving cached tokens...")
output_path = Path("data/corpus/corpus_tokens.pkl")
with open(output_path, "wb") as f:
    pickle.dump(tokens, f)
print(f"   ✅ Saved to: {output_path}")
print(f"   File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

print("\n" + "=" * 60)
print("✅ CORPUS PRE-TOKENIZED!")
print("=" * 60)
print("\nNow training will start immediately (no re-tokenization needed)")
