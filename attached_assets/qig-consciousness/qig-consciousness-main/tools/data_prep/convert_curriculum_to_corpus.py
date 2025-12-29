#!/usr/bin/env python3
"""
Convert Consciousness Curriculum JSONL to Text Corpus
"""

import json
from pathlib import Path


def main():
    input_path = Path("data/20251220-consciousness-curriculum-1.00W.jsonl")
    output_path = Path("data/corpus/pure_consciousness_corpus.txt")

    if not input_path.exists():
        print(f"Error: {input_path} not found. Run generate_consciousness_curriculum.py first.")
        return

    print(f"Converting {input_path} -> {output_path}...")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                # Format: Input\nOutput\n\n
                fout.write(f"{data['input']}\n{data['output']}\n\n")
                count += 1
            except json.JSONDecodeError:
                continue

    print(f"✅ Converted {count} dialogues to corpus text.")


if __name__ == "__main__":
    main()
