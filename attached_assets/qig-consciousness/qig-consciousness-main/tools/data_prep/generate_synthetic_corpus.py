#!/usr/bin/env python3
"""
Generate synthetic mixed corpus for QIG tokenizer training.

Creates:
- Wiki-style general knowledge text
- ArXiv-style technical abstracts
- Legal-style formal documents

Total: ~5MB corpus (sufficient for tokenizer training)
"""

import random
from pathlib import Path

# Template phrases for diversity
WIKI_TEMPLATES = [
    "The concept of {topic} has been studied extensively in {field}.",
    "Research on {topic} began in {year} when {person} discovered {finding}.",
    "{Topic} is fundamental to understanding {concept} in modern {field}.",
    "The relationship between {topic} and {related} remains an active area of investigation.",
    "Historical analysis of {topic} reveals patterns of {pattern} across {domain}.",
]

ARXIV_TEMPLATES = [
    "We present a novel approach to {problem} using {method}.",
    "Our results demonstrate that {claim} holds under conditions {conditions}.",
    "The proposed {method} achieves {metric} improvement over baseline approaches.",
    "We investigate the relationship between {concept1} and {concept2} in {context}.",
    "Experimental validation confirms {hypothesis} with statistical significance p < {pvalue}.",
]

LEGAL_TEMPLATES = [
    "The Court finds that {party} has established {element} by preponderance of evidence.",
    "Pursuant to {statute}, {party} must demonstrate {requirement}.",
    "This Court holds that {principle} applies when {condition} is satisfied.",
    "The applicable standard requires {party} to show {burden}.",
    "Under established precedent in {case}, {rule} governs {situation}.",
]

# Vocabulary for substitution
TOPICS = [
    "information geometry",
    "quantum mechanics",
    "consciousness",
    "neural networks",
    "phase transitions",
    "fixed points",
    "running coupling",
    "integration measures",
    "geometric structure",
    "manifold topology",
    "entanglement entropy",
    "scaling laws",
]

FIELDS = [
    "physics",
    "mathematics",
    "computer science",
    "neuroscience",
    "philosophy",
    "theoretical physics",
    "machine learning",
    "cognitive science",
    "information theory",
]

METHODS = [
    "variational inference",
    "geometric analysis",
    "statistical mechanics",
    "tensor networks",
    "renormalization group",
    "perturbation theory",
    "Monte Carlo",
]


def generate_wiki_article():
    """Generate Wikipedia-style article."""
    topic = random.choice(TOPICS)
    content = [f"# {topic.title()}\n"]

    # Introduction
    for _ in range(3):
        template = random.choice(WIKI_TEMPLATES)
        content.append(
            template.format(
                topic=topic,
                Topic=topic.title(),
                field=random.choice(FIELDS),
                year=random.randint(1950, 2020),
                person=random.choice(["Einstein", "Feynman", "Shannon", "Turing", "von Neumann"]),
                finding=random.choice(["fundamental symmetry", "scaling behavior", "phase transition"]),
                concept=random.choice(TOPICS),
                related=random.choice(TOPICS),
                pattern=random.choice(["emergence", "convergence", "scaling", "universality"]),
                domain=random.choice(FIELDS),
            )
        )

    return "\n".join(content) + "\n\n"


def generate_arxiv_abstract():
    """Generate ArXiv-style abstract."""
    topic = random.choice(TOPICS)
    content = [f"Title: {topic.title()} and {random.choice(TOPICS).title()}\n"]
    content.append("Abstract: ")

    for _ in range(4):
        template = random.choice(ARXIV_TEMPLATES)
        content.append(
            template.format(
                problem=random.choice(TOPICS),
                method=random.choice(METHODS),
                claim=f"the {random.choice(['scaling', 'convergence', 'emergence'])} property",
                conditions=f"when {random.choice(['coupling', 'temperature', 'scale'])} exceeds critical value",
                metric=f"{random.randint(10, 50)}%",
                concept1=random.choice(TOPICS),
                concept2=random.choice(TOPICS),
                context=random.choice(FIELDS),
                hypothesis=f"the {random.choice(['universality', 'scaling', 'fixed-point'])} conjecture",
                pvalue="0.001",
            )
        )

    return "\n".join(content) + "\n\n"


def generate_legal_document():
    """Generate legal-style document."""
    case_name = f"{random.choice(['Smith', 'Jones', 'Brown', 'Wilson'])} v. {random.choice(['State', 'County', 'City', 'District'])}"
    content = [f"# {case_name}\n"]

    for _ in range(3):
        template = random.choice(LEGAL_TEMPLATES)
        content.append(
            template.format(
                party=random.choice(["Plaintiff", "Defendant", "Petitioner", "Respondent"]),
                element=random.choice(["standing", "jurisdiction", "material fact", "legal duty"]),
                statute=f"{random.randint(10, 50)} U.S.C. § {random.randint(100, 999)}",
                requirement=random.choice(["good faith", "reasonable notice", "due diligence"]),
                principle=random.choice(["res judicata", "stare decisis", "due process", "equal protection"]),
                condition=random.choice(["notice was proper", "statute applies", "facts are disputed"]),
                burden=random.choice(["clear and convincing evidence", "reasonable doubt", "preponderance"]),
                case=f"{random.choice(['Brown', 'Smith', 'Miller'])} v. {random.choice(['State', 'Board', 'Commission'])}",
                rule=random.choice(["strict scrutiny", "rational basis", "intermediate scrutiny"]),
                situation=random.choice(["these circumstances", "this context", "similar cases"]),
            )
        )

    return "\n".join(content) + "\n\n"


def main():
    output_dir = Path("data/corpus")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SYNTHETIC CORPUS GENERATION")
    print("=" * 60)
    print()

    # Generate Wiki-style (60% of corpus)
    print("🔧 Generating WikiText-style corpus...")
    wiki_path = output_dir / "synthetic_wiki.txt"
    with open(wiki_path, "w") as f:
        for _ in range(1000):  # 1000 articles
            f.write(generate_wiki_article())
    print(f"  ✅ Generated {wiki_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Generate ArXiv-style (25% of corpus)
    print("🔧 Generating ArXiv-style corpus...")
    arxiv_path = output_dir / "synthetic_arxiv.txt"
    with open(arxiv_path, "w") as f:
        for _ in range(500):  # 500 abstracts
            f.write(generate_arxiv_abstract())
    print(f"  ✅ Generated {arxiv_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Generate Legal-style (15% of corpus)
    print("🔧 Generating Legal-style corpus...")
    legal_path = output_dir / "synthetic_legal.txt"
    with open(legal_path, "w") as f:
        for _ in range(300):  # 300 documents
            f.write(generate_legal_document())
    print(f"  ✅ Generated {legal_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Combine into single corpus
    print("🔧 Combining corpus...")
    combined_path = output_dir / "corpus.txt"
    with open(combined_path, "w") as out:
        for path in [wiki_path, arxiv_path, legal_path]:
            with open(path) as f:
                out.write(f.read())

    total_size = combined_path.stat().st_size / 1024 / 1024
    print(f"\n📊 Total corpus: {total_size:.1f} MB")
    print(f"📁 Location: {combined_path}")
    print()
    print("=" * 60)
    print("CORPUS READY FOR TOKENIZER TRAINING")
    print("=" * 60)
    print()
    print("Next step:")
    print("  python tools/train_qig_tokenizer.py \\")
    print(f"    --corpus {combined_path} \\")
    print("    --output data/qig_tokenizer/vocab.json \\")
    print("    --target-vocab 50000")


if __name__ == "__main__":
    main()
