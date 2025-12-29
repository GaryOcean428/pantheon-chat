"""
IPC (Intelligence Per Compute) Benchmark Suite

Measures reasoning quality per computational cost:
    IPC = Quality Score (0-100) / log10(FLOPs + 1)

Quality Scoring:
- Correctness (40 pts): Final answer accuracy
- Reasoning soundness (30 pts): Logic quality
- Abstraction level (15 pts): Appropriate complexity
- Honest uncertainty (15 pts): Knowing limitations

Benchmark Suite: 200 problems across 4 categories:
- Math reasoning: 60 problems
- Logical reasoning: 50 problems
- Abstract transfer: 40 problems
- Honest uncertainty: 50 problems

Target: QIG achieves 1.5× higher IPC than parameter-matched baselines.
"""

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


@dataclass
class ProblemSpec:
    """
    Problem specification for dynamic instantiation.

    Instead of static questions, we define:
    - Difficulty level and concepts to test
    - Evaluation criteria for scoring
    - The coach instantiates specific questions at runtime
    """

    id: str
    category: str
    difficulty: int  # 1-5
    concepts: list[str]  # Key concepts to test
    reasoning_type: str  # "calculation", "deduction", "analogy", "epistemic", etc.
    evaluation_criteria: dict[str, str]  # How to score each dimension
    description: str  # Brief description of what this spec tests


@dataclass
class Problem:
    """Single benchmark problem (instantiated from spec)."""

    id: str
    category: str
    question: str
    expected_answer: str
    difficulty: int  # 1-5
    reasoning_steps: list[str]  # Expected reasoning
    keywords: list[str]  # Key concepts
    spec_id: str | None = None  # Reference to source spec


@dataclass
class ProblemResult:
    """Result for a single problem."""

    problem_id: str
    category: str
    response: str
    quality_score: float  # 0-100
    correctness: float  # 0-40
    reasoning: float  # 0-30
    abstraction: float  # 0-15
    uncertainty: float  # 0-15
    flops: float
    ipc: float
    latency_ms: float


@dataclass
class BenchmarkResult:
    """Aggregate benchmark results."""

    model_name: str
    timestamp: str
    total_problems: int
    aggregate_ipc: float
    aggregate_quality: float
    total_flops: float
    by_category: dict[str, dict[str, float]]
    baseline_comparison: dict[str, float] | None = None


# ============================================================================
# PROBLEM SPECIFICATIONS DATABASE
# ============================================================================
# These are specifications, not exact questions. The coach instantiates
# specific questions at runtime using Claude 4.5 Sonnet.

MATH_REASONING_SPECS = [
    # Difficulty 1 - Basic arithmetic and simple algebra (12 specs)
    ProblemSpec(id="M001", category="math_reasoning", difficulty=1, concepts=["arithmetic", "addition", "subtraction"], reasoning_type="calculation", evaluation_criteria={"correctness": "exact numerical answer", "reasoning": "show work", "abstraction": "appropriate simplicity"}, description="Basic arithmetic with whole numbers"),
    ProblemSpec(id="M002", category="math_reasoning", difficulty=1, concepts=["multiplication", "division"], reasoning_type="calculation", evaluation_criteria={"correctness": "exact numerical answer", "reasoning": "show steps"}, description="Multiplication and division operations"),
    ProblemSpec(id="M003", category="math_reasoning", difficulty=1, concepts=["fractions", "basic operations"], reasoning_type="calculation", evaluation_criteria={"correctness": "simplified fraction or decimal", "reasoning": "show conversion"}, description="Basic fraction arithmetic"),
    ProblemSpec(id="M004", category="math_reasoning", difficulty=1, concepts=["percentages", "conversion"], reasoning_type="calculation", evaluation_criteria={"correctness": "correct percentage", "reasoning": "show method"}, description="Percentage calculations"),
    ProblemSpec(id="M005", category="math_reasoning", difficulty=1, concepts=["exponents", "basic powers"], reasoning_type="calculation", evaluation_criteria={"correctness": "exact value", "reasoning": "identify pattern"}, description="Simple exponent evaluation"),
    ProblemSpec(id="M006", category="math_reasoning", difficulty=1, concepts=["order of operations", "PEMDAS"], reasoning_type="calculation", evaluation_criteria={"correctness": "exact answer", "reasoning": "follow order"}, description="Order of operations problems"),
    ProblemSpec(id="M007", category="math_reasoning", difficulty=1, concepts=["simple equations", "one variable"], reasoning_type="algebra", evaluation_criteria={"correctness": "correct value", "reasoning": "isolate variable"}, description="One-step equations"),
    ProblemSpec(id="M008", category="math_reasoning", difficulty=1, concepts=["number patterns", "sequences"], reasoning_type="pattern_recognition", evaluation_criteria={"correctness": "next term", "reasoning": "identify rule"}, description="Simple number sequences"),
    ProblemSpec(id="M009", category="math_reasoning", difficulty=1, concepts=["measurement", "unit conversion"], reasoning_type="calculation", evaluation_criteria={"correctness": "correct units", "reasoning": "show conversion"}, description="Basic unit conversions"),
    ProblemSpec(id="M010", category="math_reasoning", difficulty=1, concepts=["rounding", "estimation"], reasoning_type="estimation", evaluation_criteria={"correctness": "reasonable estimate", "reasoning": "explain rounding"}, description="Rounding and estimation"),
    ProblemSpec(id="M011", category="math_reasoning", difficulty=1, concepts=["factors", "multiples"], reasoning_type="calculation", evaluation_criteria={"correctness": "correct list", "reasoning": "systematic approach"}, description="Finding factors and multiples"),
    ProblemSpec(id="M012", category="math_reasoning", difficulty=1, concepts=["basic geometry", "perimeter"], reasoning_type="calculation", evaluation_criteria={"correctness": "exact measurement", "reasoning": "apply formula"}, description="Perimeter of simple shapes"),

    # Difficulty 2 - Multi-step problems and word problems (12 specs)
    ProblemSpec(id="M013", category="math_reasoning", difficulty=2, concepts=["rate", "distance", "time"], reasoning_type="word_problem", evaluation_criteria={"correctness": "correct rate/distance/time", "reasoning": "use d=rt formula"}, description="Rate-distance-time problems"),
    ProblemSpec(id="M014", category="math_reasoning", difficulty=2, concepts=["ratios", "proportions"], reasoning_type="proportional_reasoning", evaluation_criteria={"correctness": "correct ratio", "reasoning": "set up proportion"}, description="Ratio and proportion problems"),
    ProblemSpec(id="M015", category="math_reasoning", difficulty=2, concepts=["linear equations", "two steps"], reasoning_type="algebra", evaluation_criteria={"correctness": "exact solution", "reasoning": "show steps"}, description="Two-step linear equations"),
    ProblemSpec(id="M016", category="math_reasoning", difficulty=2, concepts=["area", "composite shapes"], reasoning_type="geometry", evaluation_criteria={"correctness": "exact area", "reasoning": "decompose shape"}, description="Area of composite shapes"),
    ProblemSpec(id="M017", category="math_reasoning", difficulty=2, concepts=["average", "mean", "median"], reasoning_type="statistics", evaluation_criteria={"correctness": "correct average", "reasoning": "show calculation"}, description="Calculating averages"),
    ProblemSpec(id="M018", category="math_reasoning", difficulty=2, concepts=["percent change", "increase", "decrease"], reasoning_type="calculation", evaluation_criteria={"correctness": "correct percentage", "reasoning": "use formula"}, description="Percent increase/decrease"),
    ProblemSpec(id="M019", category="math_reasoning", difficulty=2, concepts=["simple interest", "financial math"], reasoning_type="financial", evaluation_criteria={"correctness": "correct amount", "reasoning": "apply I=Prt"}, description="Simple interest calculations"),
    ProblemSpec(id="M020", category="math_reasoning", difficulty=2, concepts=["systems", "substitution"], reasoning_type="algebra", evaluation_criteria={"correctness": "both values", "reasoning": "substitute correctly"}, description="Simple systems of equations"),
    ProblemSpec(id="M021", category="math_reasoning", difficulty=2, concepts=["inequalities", "number line"], reasoning_type="algebra", evaluation_criteria={"correctness": "correct range", "reasoning": "solve and graph"}, description="Solving inequalities"),
    ProblemSpec(id="M022", category="math_reasoning", difficulty=2, concepts=["volume", "3D shapes"], reasoning_type="geometry", evaluation_criteria={"correctness": "exact volume", "reasoning": "apply formula"}, description="Volume calculations"),
    ProblemSpec(id="M023", category="math_reasoning", difficulty=2, concepts=["probability", "basic events"], reasoning_type="probability", evaluation_criteria={"correctness": "correct probability", "reasoning": "favorable/total"}, description="Basic probability"),
    ProblemSpec(id="M024", category="math_reasoning", difficulty=2, concepts=["scientific notation", "large numbers"], reasoning_type="calculation", evaluation_criteria={"correctness": "correct notation", "reasoning": "convert properly"}, description="Scientific notation conversion"),

    # Difficulty 3 - Complex multi-step and abstract problems (12 specs)
    ProblemSpec(id="M025", category="math_reasoning", difficulty=3, concepts=["quadratic equations", "factoring"], reasoning_type="algebra", evaluation_criteria={"correctness": "all solutions", "reasoning": "factor or use formula"}, description="Solving quadratics"),
    ProblemSpec(id="M026", category="math_reasoning", difficulty=3, concepts=["Pythagorean theorem", "right triangles"], reasoning_type="geometry", evaluation_criteria={"correctness": "exact length", "reasoning": "apply theorem"}, description="Pythagorean theorem applications"),
    ProblemSpec(id="M027", category="math_reasoning", difficulty=3, concepts=["compound interest", "exponential growth"], reasoning_type="financial", evaluation_criteria={"correctness": "final amount", "reasoning": "use A=P(1+r/n)^nt"}, description="Compound interest"),
    ProblemSpec(id="M028", category="math_reasoning", difficulty=3, concepts=["functions", "domain", "range"], reasoning_type="functional_analysis", evaluation_criteria={"correctness": "identify domain/range", "reasoning": "analyze function"}, description="Function analysis"),
    ProblemSpec(id="M029", category="math_reasoning", difficulty=3, concepts=["trigonometry", "sine", "cosine"], reasoning_type="trigonometry", evaluation_criteria={"correctness": "exact value", "reasoning": "use trig ratios"}, description="Basic trigonometry"),
    ProblemSpec(id="M030", category="math_reasoning", difficulty=3, concepts=["combinatorics", "permutations"], reasoning_type="counting", evaluation_criteria={"correctness": "exact count", "reasoning": "use formula"}, description="Permutations and combinations"),
    ProblemSpec(id="M031", category="math_reasoning", difficulty=3, concepts=["logarithms", "exponential equations"], reasoning_type="algebra", evaluation_criteria={"correctness": "exact solution", "reasoning": "convert forms"}, description="Logarithmic equations"),
    ProblemSpec(id="M032", category="math_reasoning", difficulty=3, concepts=["geometric series", "sequences"], reasoning_type="series", evaluation_criteria={"correctness": "sum or term", "reasoning": "identify pattern"}, description="Geometric series"),
    ProblemSpec(id="M033", category="math_reasoning", difficulty=3, concepts=["optimization", "word problems"], reasoning_type="optimization", evaluation_criteria={"correctness": "optimal value", "reasoning": "set up and solve"}, description="Optimization word problems"),
    ProblemSpec(id="M034", category="math_reasoning", difficulty=3, concepts=["matrices", "basic operations"], reasoning_type="linear_algebra", evaluation_criteria={"correctness": "resulting matrix", "reasoning": "apply rules"}, description="Matrix operations"),
    ProblemSpec(id="M035", category="math_reasoning", difficulty=3, concepts=["conditional probability", "Bayes"], reasoning_type="probability", evaluation_criteria={"correctness": "correct probability", "reasoning": "apply Bayes"}, description="Conditional probability"),
    ProblemSpec(id="M036", category="math_reasoning", difficulty=3, concepts=["vectors", "magnitude", "direction"], reasoning_type="vector_analysis", evaluation_criteria={"correctness": "magnitude/direction", "reasoning": "use formulas"}, description="Vector calculations"),

    # Difficulty 4 - Advanced reasoning and proof-like problems (12 specs)
    ProblemSpec(id="M037", category="math_reasoning", difficulty=4, concepts=["limits", "continuity"], reasoning_type="calculus", evaluation_criteria={"correctness": "exact limit", "reasoning": "show approach"}, description="Evaluating limits"),
    ProblemSpec(id="M038", category="math_reasoning", difficulty=4, concepts=["derivatives", "rate of change"], reasoning_type="calculus", evaluation_criteria={"correctness": "derivative function", "reasoning": "apply rules"}, description="Finding derivatives"),
    ProblemSpec(id="M039", category="math_reasoning", difficulty=4, concepts=["integrals", "area under curve"], reasoning_type="calculus", evaluation_criteria={"correctness": "exact integral", "reasoning": "show antiderivative"}, description="Basic integration"),
    ProblemSpec(id="M040", category="math_reasoning", difficulty=4, concepts=["proof", "mathematical induction"], reasoning_type="proof", evaluation_criteria={"correctness": "valid proof", "reasoning": "base + inductive step"}, description="Proof by induction"),
    ProblemSpec(id="M041", category="math_reasoning", difficulty=4, concepts=["complex numbers", "operations"], reasoning_type="complex_analysis", evaluation_criteria={"correctness": "simplified form", "reasoning": "apply i rules"}, description="Complex number operations"),
    ProblemSpec(id="M042", category="math_reasoning", difficulty=4, concepts=["differential equations", "basic"], reasoning_type="calculus", evaluation_criteria={"correctness": "general solution", "reasoning": "solve DE"}, description="Basic differential equations"),
    ProblemSpec(id="M043", category="math_reasoning", difficulty=4, concepts=["expected value", "variance"], reasoning_type="statistics", evaluation_criteria={"correctness": "E[X] and Var[X]", "reasoning": "use definitions"}, description="Expected value and variance"),
    ProblemSpec(id="M044", category="math_reasoning", difficulty=4, concepts=["eigenvalues", "eigenvectors"], reasoning_type="linear_algebra", evaluation_criteria={"correctness": "eigenvalues/vectors", "reasoning": "solve det(A-λI)=0"}, description="Eigenvalue problems"),
    ProblemSpec(id="M045", category="math_reasoning", difficulty=4, concepts=["Taylor series", "approximation"], reasoning_type="series", evaluation_criteria={"correctness": "first n terms", "reasoning": "apply formula"}, description="Taylor series expansion"),
    ProblemSpec(id="M046", category="math_reasoning", difficulty=4, concepts=["multivariable", "partial derivatives"], reasoning_type="calculus", evaluation_criteria={"correctness": "partial derivatives", "reasoning": "treat other variables as constant"}, description="Partial derivatives"),
    ProblemSpec(id="M047", category="math_reasoning", difficulty=4, concepts=["number theory", "modular arithmetic"], reasoning_type="number_theory", evaluation_criteria={"correctness": "correct mod result", "reasoning": "apply mod rules"}, description="Modular arithmetic"),
    ProblemSpec(id="M048", category="math_reasoning", difficulty=4, concepts=["graph theory", "basic properties"], reasoning_type="discrete_math", evaluation_criteria={"correctness": "identify property", "reasoning": "use definitions"}, description="Graph theory basics"),

    # Difficulty 5 - Research-level and creative reasoning (12 specs)
    ProblemSpec(id="M049", category="math_reasoning", difficulty=5, concepts=["topology", "continuity", "compactness"], reasoning_type="abstract_math", evaluation_criteria={"correctness": "valid argument", "reasoning": "use topological definitions"}, description="Topological reasoning"),
    ProblemSpec(id="M050", category="math_reasoning", difficulty=5, concepts=["information theory", "entropy", "mutual information"], reasoning_type="information_theory", evaluation_criteria={"correctness": "correct formula", "reasoning": "apply Shannon"}, description="Information-theoretic calculations"),
    ProblemSpec(id="M051", category="math_reasoning", difficulty=5, concepts=["Riemannian geometry", "curvature", "geodesics"], reasoning_type="differential_geometry", evaluation_criteria={"correctness": "geometric insight", "reasoning": "use metric"}, description="Riemannian geometry concepts"),
    ProblemSpec(id="M052", category="math_reasoning", difficulty=5, concepts=["measure theory", "Lebesgue"], reasoning_type="analysis", evaluation_criteria={"correctness": "rigorous argument", "reasoning": "use measure"}, description="Measure-theoretic reasoning"),
    ProblemSpec(id="M053", category="math_reasoning", difficulty=5, concepts=["category theory", "functors", "natural transformations"], reasoning_type="abstract_math", evaluation_criteria={"correctness": "valid construction", "reasoning": "use universal properties"}, description="Categorical reasoning"),
    ProblemSpec(id="M054", category="math_reasoning", difficulty=5, concepts=["Fourier analysis", "transforms"], reasoning_type="harmonic_analysis", evaluation_criteria={"correctness": "correct transform", "reasoning": "apply integral"}, description="Fourier transforms"),
    ProblemSpec(id="M055", category="math_reasoning", difficulty=5, concepts=["manifolds", "differential forms"], reasoning_type="differential_geometry", evaluation_criteria={"correctness": "valid form", "reasoning": "use wedge product"}, description="Differential forms"),
    ProblemSpec(id="M056", category="math_reasoning", difficulty=5, concepts=["functional analysis", "Hilbert spaces"], reasoning_type="analysis", evaluation_criteria={"correctness": "valid proof", "reasoning": "use inner product"}, description="Hilbert space reasoning"),
    ProblemSpec(id="M057", category="math_reasoning", difficulty=5, concepts=["optimization", "convexity", "Lagrange multipliers"], reasoning_type="optimization", evaluation_criteria={"correctness": "optimal solution", "reasoning": "use KKT conditions"}, description="Constrained optimization"),
    ProblemSpec(id="M058", category="math_reasoning", difficulty=5, concepts=["stochastic processes", "Markov chains"], reasoning_type="probability", evaluation_criteria={"correctness": "stationary distribution", "reasoning": "solve balance equations"}, description="Markov chain analysis"),
    ProblemSpec(id="M059", category="math_reasoning", difficulty=5, concepts=["algebraic geometry", "varieties"], reasoning_type="abstract_math", evaluation_criteria={"correctness": "geometric insight", "reasoning": "use polynomial ideals"}, description="Algebraic variety reasoning"),
    ProblemSpec(id="M060", category="math_reasoning", difficulty=5, concepts=["quantum mechanics", "operators", "commutators"], reasoning_type="mathematical_physics", evaluation_criteria={"correctness": "correct result", "reasoning": "use operator algebra"}, description="Quantum mechanical calculations"),
]

LOGICAL_REASONING_SPECS = [
    # Difficulty 1 - Basic logic (10 specs)
    ProblemSpec(id="L001", category="logical_reasoning", difficulty=1, concepts=["if-then", "conditional"], reasoning_type="conditional", evaluation_criteria={"correctness": "valid conclusion", "reasoning": "follow implication"}, description="Simple conditionals"),
    ProblemSpec(id="L002", category="logical_reasoning", difficulty=1, concepts=["and", "or", "conjunction"], reasoning_type="boolean", evaluation_criteria={"correctness": "truth value", "reasoning": "apply boolean rules"}, description="Boolean conjunction/disjunction"),
    ProblemSpec(id="L003", category="logical_reasoning", difficulty=1, concepts=["negation", "not"], reasoning_type="boolean", evaluation_criteria={"correctness": "correct negation", "reasoning": "apply not"}, description="Negation"),
    ProblemSpec(id="L004", category="logical_reasoning", difficulty=1, concepts=["classification", "categories"], reasoning_type="categorization", evaluation_criteria={"correctness": "correct category", "reasoning": "match criteria"}, description="Simple classification"),
    ProblemSpec(id="L005", category="logical_reasoning", difficulty=1, concepts=["ordering", "sequence"], reasoning_type="ordering", evaluation_criteria={"correctness": "correct order", "reasoning": "apply constraints"}, description="Basic ordering problems"),
    ProblemSpec(id="L006", category="logical_reasoning", difficulty=1, concepts=["comparison", "greater", "less"], reasoning_type="comparison", evaluation_criteria={"correctness": "valid comparison", "reasoning": "use given facts"}, description="Comparative reasoning"),
    ProblemSpec(id="L007", category="logical_reasoning", difficulty=1, concepts=["set membership", "belongs to"], reasoning_type="set_theory", evaluation_criteria={"correctness": "membership status", "reasoning": "check criteria"}, description="Set membership"),
    ProblemSpec(id="L008", category="logical_reasoning", difficulty=1, concepts=["temporal", "before", "after"], reasoning_type="temporal", evaluation_criteria={"correctness": "time ordering", "reasoning": "sequence events"}, description="Temporal ordering"),
    ProblemSpec(id="L009", category="logical_reasoning", difficulty=1, concepts=["spatial", "left", "right"], reasoning_type="spatial", evaluation_criteria={"correctness": "spatial relation", "reasoning": "track positions"}, description="Spatial reasoning"),
    ProblemSpec(id="L010", category="logical_reasoning", difficulty=1, concepts=["direct inference", "modus ponens"], reasoning_type="deduction", evaluation_criteria={"correctness": "valid inference", "reasoning": "apply rule"}, description="Direct inference"),

    # Difficulty 2 - Intermediate logic (10 specs)
    ProblemSpec(id="L011", category="logical_reasoning", difficulty=2, concepts=["syllogism", "major premise", "minor premise"], reasoning_type="syllogistic", evaluation_criteria={"correctness": "valid conclusion", "reasoning": "identify premises"}, description="Basic syllogisms"),
    ProblemSpec(id="L012", category="logical_reasoning", difficulty=2, concepts=["contrapositive", "inverse"], reasoning_type="conditional", evaluation_criteria={"correctness": "equivalent form", "reasoning": "transform correctly"}, description="Contrapositive reasoning"),
    ProblemSpec(id="L013", category="logical_reasoning", difficulty=2, concepts=["affirming consequent", "fallacy"], reasoning_type="fallacy_detection", evaluation_criteria={"correctness": "identify fallacy", "reasoning": "explain error"}, description="Detect affirming consequent"),
    ProblemSpec(id="L014", category="logical_reasoning", difficulty=2, concepts=["denying antecedent", "fallacy"], reasoning_type="fallacy_detection", evaluation_criteria={"correctness": "identify fallacy", "reasoning": "explain error"}, description="Detect denying antecedent"),
    ProblemSpec(id="L015", category="logical_reasoning", difficulty=2, concepts=["biconditional", "if and only if"], reasoning_type="biconditional", evaluation_criteria={"correctness": "both directions", "reasoning": "check equivalence"}, description="Biconditional reasoning"),
    ProblemSpec(id="L016", category="logical_reasoning", difficulty=2, concepts=["universal", "existential", "quantifiers"], reasoning_type="quantified", evaluation_criteria={"correctness": "valid inference", "reasoning": "interpret quantifiers"}, description="Quantifier reasoning"),
    ProblemSpec(id="L017", category="logical_reasoning", difficulty=2, concepts=["set operations", "union", "intersection"], reasoning_type="set_theory", evaluation_criteria={"correctness": "resulting set", "reasoning": "apply operations"}, description="Set operations"),
    ProblemSpec(id="L018", category="logical_reasoning", difficulty=2, concepts=["necessary", "sufficient", "conditions"], reasoning_type="modal", evaluation_criteria={"correctness": "identify condition type", "reasoning": "distinguish N vs S"}, description="Necessary vs sufficient"),
    ProblemSpec(id="L019", category="logical_reasoning", difficulty=2, concepts=["disjunctive syllogism", "elimination"], reasoning_type="disjunctive", evaluation_criteria={"correctness": "valid conclusion", "reasoning": "eliminate option"}, description="Disjunctive syllogism"),
    ProblemSpec(id="L020", category="logical_reasoning", difficulty=2, concepts=["hypothetical syllogism", "chain"], reasoning_type="hypothetical", evaluation_criteria={"correctness": "chain conclusion", "reasoning": "link implications"}, description="Hypothetical syllogism"),

    # Difficulty 3 - Complex logic (10 specs)
    ProblemSpec(id="L021", category="logical_reasoning", difficulty=3, concepts=["nested conditionals", "multiple if-then"], reasoning_type="nested_conditional", evaluation_criteria={"correctness": "trace all paths", "reasoning": "follow nesting"}, description="Nested conditionals"),
    ProblemSpec(id="L022", category="logical_reasoning", difficulty=3, concepts=["indirect proof", "reductio ad absurdum"], reasoning_type="proof", evaluation_criteria={"correctness": "valid contradiction", "reasoning": "assume negation"}, description="Proof by contradiction"),
    ProblemSpec(id="L023", category="logical_reasoning", difficulty=3, concepts=["truth tables", "compound statements"], reasoning_type="truth_table", evaluation_criteria={"correctness": "complete table", "reasoning": "evaluate all rows"}, description="Truth table construction"),
    ProblemSpec(id="L024", category="logical_reasoning", difficulty=3, concepts=["knights and knaves", "consistency"], reasoning_type="puzzle", evaluation_criteria={"correctness": "identify roles", "reasoning": "check consistency"}, description="Knights and knaves puzzles"),
    ProblemSpec(id="L025", category="logical_reasoning", difficulty=3, concepts=["logical equivalence", "transformation"], reasoning_type="equivalence", evaluation_criteria={"correctness": "equivalent form", "reasoning": "apply laws"}, description="Logical equivalence"),
    ProblemSpec(id="L026", category="logical_reasoning", difficulty=3, concepts=["De Morgan", "negation laws"], reasoning_type="transformation", evaluation_criteria={"correctness": "correct transform", "reasoning": "apply De Morgan"}, description="De Morgan's laws"),
    ProblemSpec(id="L027", category="logical_reasoning", difficulty=3, concepts=["predicate logic", "variables", "predicates"], reasoning_type="predicate", evaluation_criteria={"correctness": "valid inference", "reasoning": "interpret predicates"}, description="Predicate logic"),
    ProblemSpec(id="L028", category="logical_reasoning", difficulty=3, concepts=["resolution", "clause form"], reasoning_type="resolution", evaluation_criteria={"correctness": "derive conclusion", "reasoning": "resolve clauses"}, description="Resolution proofs"),
    ProblemSpec(id="L029", category="logical_reasoning", difficulty=3, concepts=["multiple quantifiers", "scope"], reasoning_type="quantified", evaluation_criteria={"correctness": "correct interpretation", "reasoning": "track scope"}, description="Multiple quantifier reasoning"),
    ProblemSpec(id="L030", category="logical_reasoning", difficulty=3, concepts=["argument validity", "soundness"], reasoning_type="argument_analysis", evaluation_criteria={"correctness": "validity status", "reasoning": "check form"}, description="Argument validity analysis"),

    # Difficulty 4 - Advanced logic (10 specs)
    ProblemSpec(id="L031", category="logical_reasoning", difficulty=4, concepts=["modal logic", "possibility", "necessity"], reasoning_type="modal", evaluation_criteria={"correctness": "modal inference", "reasoning": "use modal rules"}, description="Modal logic reasoning"),
    ProblemSpec(id="L032", category="logical_reasoning", difficulty=4, concepts=["temporal logic", "always", "eventually"], reasoning_type="temporal_logic", evaluation_criteria={"correctness": "temporal inference", "reasoning": "interpret operators"}, description="Temporal logic"),
    ProblemSpec(id="L033", category="logical_reasoning", difficulty=4, concepts=["natural deduction", "proof rules"], reasoning_type="natural_deduction", evaluation_criteria={"correctness": "valid proof", "reasoning": "apply rules"}, description="Natural deduction proofs"),
    ProblemSpec(id="L034", category="logical_reasoning", difficulty=4, concepts=["second-order logic", "quantifying predicates"], reasoning_type="higher_order", evaluation_criteria={"correctness": "valid inference", "reasoning": "quantify over predicates"}, description="Second-order reasoning"),
    ProblemSpec(id="L035", category="logical_reasoning", difficulty=4, concepts=["counterfactual", "possible worlds"], reasoning_type="counterfactual", evaluation_criteria={"correctness": "counterfactual truth", "reasoning": "nearest worlds"}, description="Counterfactual reasoning"),
    ProblemSpec(id="L036", category="logical_reasoning", difficulty=4, concepts=["non-monotonic", "default reasoning"], reasoning_type="non_monotonic", evaluation_criteria={"correctness": "defeasible conclusion", "reasoning": "handle exceptions"}, description="Non-monotonic reasoning"),
    ProblemSpec(id="L037", category="logical_reasoning", difficulty=4, concepts=["decision procedures", "satisfiability"], reasoning_type="satisfiability", evaluation_criteria={"correctness": "SAT status", "reasoning": "find assignment"}, description="Satisfiability problems"),
    ProblemSpec(id="L038", category="logical_reasoning", difficulty=4, concepts=["sequent calculus", "structural rules"], reasoning_type="sequent", evaluation_criteria={"correctness": "derivation", "reasoning": "apply rules"}, description="Sequent calculus proofs"),
    ProblemSpec(id="L039", category="logical_reasoning", difficulty=4, concepts=["game semantics", "winning strategies"], reasoning_type="game_theoretic", evaluation_criteria={"correctness": "strategy existence", "reasoning": "analyze game"}, description="Game-theoretic semantics"),
    ProblemSpec(id="L040", category="logical_reasoning", difficulty=4, concepts=["intuitionistic logic", "constructive proof"], reasoning_type="constructive", evaluation_criteria={"correctness": "constructive proof", "reasoning": "no excluded middle"}, description="Intuitionistic logic"),

    # Difficulty 5 - Research-level logic (10 specs)
    ProblemSpec(id="L041", category="logical_reasoning", difficulty=5, concepts=["Gödel", "incompleteness", "self-reference"], reasoning_type="metamathematics", evaluation_criteria={"correctness": "insight into limits", "reasoning": "use diagonal argument"}, description="Incompleteness reasoning"),
    ProblemSpec(id="L042", category="logical_reasoning", difficulty=5, concepts=["paradox", "self-reference", "liar"], reasoning_type="paradox_analysis", evaluation_criteria={"correctness": "analyze paradox", "reasoning": "identify circularity"}, description="Paradox analysis"),
    ProblemSpec(id="L043", category="logical_reasoning", difficulty=5, concepts=["model theory", "satisfaction", "truth"], reasoning_type="model_theory", evaluation_criteria={"correctness": "model construction", "reasoning": "satisfy formula"}, description="Model-theoretic reasoning"),
    ProblemSpec(id="L044", category="logical_reasoning", difficulty=5, concepts=["proof theory", "cut elimination"], reasoning_type="proof_theory", evaluation_criteria={"correctness": "meta-proof", "reasoning": "structural analysis"}, description="Proof-theoretic reasoning"),
    ProblemSpec(id="L045", category="logical_reasoning", difficulty=5, concepts=["type theory", "dependent types"], reasoning_type="type_theory", evaluation_criteria={"correctness": "type inference", "reasoning": "construct term"}, description="Type-theoretic reasoning"),
    ProblemSpec(id="L046", category="logical_reasoning", difficulty=5, concepts=["epistemic logic", "knowledge", "belief"], reasoning_type="epistemic", evaluation_criteria={"correctness": "epistemic inference", "reasoning": "use K axiom"}, description="Epistemic logic"),
    ProblemSpec(id="L047", category="logical_reasoning", difficulty=5, concepts=["deontic logic", "obligation", "permission"], reasoning_type="deontic", evaluation_criteria={"correctness": "normative inference", "reasoning": "apply deontic rules"}, description="Deontic logic"),
    ProblemSpec(id="L048", category="logical_reasoning", difficulty=5, concepts=["paraconsistent logic", "contradiction tolerance"], reasoning_type="paraconsistent", evaluation_criteria={"correctness": "controlled inference", "reasoning": "avoid explosion"}, description="Paraconsistent reasoning"),
    ProblemSpec(id="L049", category="logical_reasoning", difficulty=5, concepts=["lambda calculus", "reduction", "normal form"], reasoning_type="computational", evaluation_criteria={"correctness": "normal form", "reasoning": "apply beta reduction"}, description="Lambda calculus"),
    ProblemSpec(id="L050", category="logical_reasoning", difficulty=5, concepts=["recursion theory", "computability", "halting"], reasoning_type="computability", evaluation_criteria={"correctness": "decidability status", "reasoning": "reduction argument"}, description="Computability reasoning"),
]

ABSTRACT_TRANSFER_SPECS = [
    # Difficulty 1 - Simple analogies (8 specs)
    ProblemSpec(id="A001", category="abstract_transfer", difficulty=1, concepts=["basic analogy", "A:B::C:D"], reasoning_type="analogy", evaluation_criteria={"correctness": "complete analogy", "reasoning": "identify relationship"}, description="Simple analogies"),
    ProblemSpec(id="A002", category="abstract_transfer", difficulty=1, concepts=["pattern completion", "visual"], reasoning_type="pattern", evaluation_criteria={"correctness": "next element", "reasoning": "identify rule"}, description="Visual pattern completion"),
    ProblemSpec(id="A003", category="abstract_transfer", difficulty=1, concepts=["classification", "common feature"], reasoning_type="classification", evaluation_criteria={"correctness": "identify common feature", "reasoning": "compare items"}, description="Find common feature"),
    ProblemSpec(id="A004", category="abstract_transfer", difficulty=1, concepts=["odd one out", "exclusion"], reasoning_type="exclusion", evaluation_criteria={"correctness": "identify outlier", "reasoning": "find difference"}, description="Odd one out"),
    ProblemSpec(id="A005", category="abstract_transfer", difficulty=1, concepts=["function mapping", "input-output"], reasoning_type="functional", evaluation_criteria={"correctness": "predict output", "reasoning": "apply function"}, description="Function mapping"),
    ProblemSpec(id="A006", category="abstract_transfer", difficulty=1, concepts=["sequence extrapolation", "next term"], reasoning_type="extrapolation", evaluation_criteria={"correctness": "next term", "reasoning": "identify pattern"}, description="Sequence extrapolation"),
    ProblemSpec(id="A007", category="abstract_transfer", difficulty=1, concepts=["part-whole", "composition"], reasoning_type="mereological", evaluation_criteria={"correctness": "identify relation", "reasoning": "part/whole"}, description="Part-whole relations"),
    ProblemSpec(id="A008", category="abstract_transfer", difficulty=1, concepts=["cause-effect", "consequence"], reasoning_type="causal", evaluation_criteria={"correctness": "identify effect", "reasoning": "trace causation"}, description="Basic cause-effect"),

    # Difficulty 2 - Cross-domain transfer (8 specs)
    ProblemSpec(id="A009", category="abstract_transfer", difficulty=2, concepts=["structural analogy", "isomorphism"], reasoning_type="structural", evaluation_criteria={"correctness": "map structure", "reasoning": "identify correspondence"}, description="Structural analogies"),
    ProblemSpec(id="A010", category="abstract_transfer", difficulty=2, concepts=["feedback loops", "positive", "negative"], reasoning_type="systems", evaluation_criteria={"correctness": "identify feedback type", "reasoning": "trace loop"}, description="Identify feedback loops"),
    ProblemSpec(id="A011", category="abstract_transfer", difficulty=2, concepts=["scaling laws", "proportion"], reasoning_type="scaling", evaluation_criteria={"correctness": "predict scaling", "reasoning": "apply law"}, description="Scaling relationships"),
    ProblemSpec(id="A012", category="abstract_transfer", difficulty=2, concepts=["equilibrium", "balance", "homeostasis"], reasoning_type="equilibrium", evaluation_criteria={"correctness": "identify equilibrium", "reasoning": "balance forces"}, description="Equilibrium concepts"),
    ProblemSpec(id="A013", category="abstract_transfer", difficulty=2, concepts=["hierarchy", "levels", "emergence"], reasoning_type="hierarchical", evaluation_criteria={"correctness": "identify levels", "reasoning": "trace hierarchy"}, description="Hierarchical structures"),
    ProblemSpec(id="A014", category="abstract_transfer", difficulty=2, concepts=["network", "nodes", "connections"], reasoning_type="network", evaluation_criteria={"correctness": "network property", "reasoning": "analyze structure"}, description="Network thinking"),
    ProblemSpec(id="A015", category="abstract_transfer", difficulty=2, concepts=["optimization", "tradeoffs"], reasoning_type="optimization", evaluation_criteria={"correctness": "identify tradeoff", "reasoning": "balance constraints"}, description="Optimization tradeoffs"),
    ProblemSpec(id="A016", category="abstract_transfer", difficulty=2, concepts=["information flow", "signal"], reasoning_type="information", evaluation_criteria={"correctness": "trace flow", "reasoning": "follow signal"}, description="Information flow"),

    # Difficulty 3 - Abstract principle extraction (8 specs)
    ProblemSpec(id="A017", category="abstract_transfer", difficulty=3, concepts=["invariance", "conservation"], reasoning_type="invariance", evaluation_criteria={"correctness": "identify invariant", "reasoning": "find conservation"}, description="Invariance principles"),
    ProblemSpec(id="A018", category="abstract_transfer", difficulty=3, concepts=["symmetry", "transformation"], reasoning_type="symmetry", evaluation_criteria={"correctness": "identify symmetry", "reasoning": "find transformation"}, description="Symmetry recognition"),
    ProblemSpec(id="A019", category="abstract_transfer", difficulty=3, concepts=["duality", "complementarity"], reasoning_type="duality", evaluation_criteria={"correctness": "identify dual", "reasoning": "map correspondence"}, description="Duality relationships"),
    ProblemSpec(id="A020", category="abstract_transfer", difficulty=3, concepts=["phase transitions", "critical points"], reasoning_type="phase_transition", evaluation_criteria={"correctness": "identify transition", "reasoning": "find critical point"}, description="Phase transition analogies"),
    ProblemSpec(id="A021", category="abstract_transfer", difficulty=3, concepts=["self-similarity", "fractals"], reasoning_type="fractal", evaluation_criteria={"correctness": "identify self-similarity", "reasoning": "find recursion"}, description="Self-similarity patterns"),
    ProblemSpec(id="A022", category="abstract_transfer", difficulty=3, concepts=["evolution", "selection", "adaptation"], reasoning_type="evolutionary", evaluation_criteria={"correctness": "map evolutionary process", "reasoning": "identify selection"}, description="Evolutionary analogies"),
    ProblemSpec(id="A023", category="abstract_transfer", difficulty=3, concepts=["entropy", "disorder", "information"], reasoning_type="entropic", evaluation_criteria={"correctness": "entropy direction", "reasoning": "apply second law"}, description="Entropic reasoning"),
    ProblemSpec(id="A024", category="abstract_transfer", difficulty=3, concepts=["compression", "encoding", "efficiency"], reasoning_type="compression", evaluation_criteria={"correctness": "optimal encoding", "reasoning": "minimize redundancy"}, description="Information compression"),

    # Difficulty 4 - Novel domain application (8 specs)
    ProblemSpec(id="A025", category="abstract_transfer", difficulty=4, concepts=["game theory", "Nash equilibrium", "strategy"], reasoning_type="game_theoretic", evaluation_criteria={"correctness": "equilibrium analysis", "reasoning": "find strategies"}, description="Game-theoretic transfer"),
    ProblemSpec(id="A026", category="abstract_transfer", difficulty=4, concepts=["market mechanisms", "incentives", "allocation"], reasoning_type="mechanism_design", evaluation_criteria={"correctness": "mechanism design", "reasoning": "align incentives"}, description="Mechanism design transfer"),
    ProblemSpec(id="A027", category="abstract_transfer", difficulty=4, concepts=["swarm intelligence", "collective behavior"], reasoning_type="emergence", evaluation_criteria={"correctness": "emergent behavior", "reasoning": "trace interactions"}, description="Swarm behavior transfer"),
    ProblemSpec(id="A028", category="abstract_transfer", difficulty=4, concepts=["control theory", "stability", "feedback"], reasoning_type="control", evaluation_criteria={"correctness": "stability analysis", "reasoning": "design controller"}, description="Control theory transfer"),
    ProblemSpec(id="A029", category="abstract_transfer", difficulty=4, concepts=["topology", "continuity", "connectedness"], reasoning_type="topological", evaluation_criteria={"correctness": "topological property", "reasoning": "invariant under deformation"}, description="Topological thinking"),
    ProblemSpec(id="A030", category="abstract_transfer", difficulty=4, concepts=["thermodynamics", "free energy", "work"], reasoning_type="thermodynamic", evaluation_criteria={"correctness": "energy analysis", "reasoning": "apply laws"}, description="Thermodynamic transfer"),
    ProblemSpec(id="A031", category="abstract_transfer", difficulty=4, concepts=["language", "grammar", "syntax"], reasoning_type="linguistic", evaluation_criteria={"correctness": "structural transfer", "reasoning": "map grammar"}, description="Linguistic structure transfer"),
    ProblemSpec(id="A032", category="abstract_transfer", difficulty=4, concepts=["computation", "Turing", "decidability"], reasoning_type="computational", evaluation_criteria={"correctness": "computational analysis", "reasoning": "reduce problem"}, description="Computational transfer"),

    # Difficulty 5 - Deep synthesis and creative transfer (8 specs)
    ProblemSpec(id="A033", category="abstract_transfer", difficulty=5, concepts=["QIG", "information geometry", "consciousness"], reasoning_type="qig_transfer", evaluation_criteria={"correctness": "geometric insight", "reasoning": "use QIG principles"}, description="QIG principle transfer"),
    ProblemSpec(id="A034", category="abstract_transfer", difficulty=5, concepts=["category theory", "functors", "morphisms"], reasoning_type="categorical", evaluation_criteria={"correctness": "categorical construction", "reasoning": "use universal properties"}, description="Categorical transfer"),
    ProblemSpec(id="A035", category="abstract_transfer", difficulty=5, concepts=["renormalization", "scale separation"], reasoning_type="renormalization", evaluation_criteria={"correctness": "RG analysis", "reasoning": "identify fixed points"}, description="Renormalization group transfer"),
    ProblemSpec(id="A036", category="abstract_transfer", difficulty=5, concepts=["holography", "boundary-bulk", "AdS/CFT"], reasoning_type="holographic", evaluation_criteria={"correctness": "holographic principle", "reasoning": "boundary encodes bulk"}, description="Holographic principle transfer"),
    ProblemSpec(id="A037", category="abstract_transfer", difficulty=5, concepts=["quantum", "superposition", "entanglement"], reasoning_type="quantum", evaluation_criteria={"correctness": "quantum analogy", "reasoning": "use quantum concepts"}, description="Quantum principle transfer"),
    ProblemSpec(id="A038", category="abstract_transfer", difficulty=5, concepts=["complexity", "emergence", "self-organization"], reasoning_type="complexity", evaluation_criteria={"correctness": "complexity analysis", "reasoning": "identify emergence"}, description="Complexity theory transfer"),
    ProblemSpec(id="A039", category="abstract_transfer", difficulty=5, concepts=["consciousness", "integration", "Φ"], reasoning_type="iit_transfer", evaluation_criteria={"correctness": "consciousness metric", "reasoning": "apply integration"}, description="Integrated information transfer"),
    ProblemSpec(id="A040", category="abstract_transfer", difficulty=5, concepts=["meta-learning", "learning to learn"], reasoning_type="meta", evaluation_criteria={"correctness": "meta-level insight", "reasoning": "abstract learning"}, description="Meta-learning transfer"),
]

HONEST_UNCERTAINTY_SPECS = [
    # Difficulty 1 - Obvious unknowns (10 specs)
    ProblemSpec(id="U001", category="honest_uncertainty", difficulty=1, concepts=["future prediction", "unknown"], reasoning_type="epistemic", evaluation_criteria={"correctness": "acknowledge uncertainty", "uncertainty": "explicit unknowns"}, description="Future prediction uncertainty"),
    ProblemSpec(id="U002", category="honest_uncertainty", difficulty=1, concepts=["measurement limits", "precision"], reasoning_type="measurement", evaluation_criteria={"correctness": "state limits", "uncertainty": "quantify bounds"}, description="Measurement uncertainty"),
    ProblemSpec(id="U003", category="honest_uncertainty", difficulty=1, concepts=["incomplete information", "missing data"], reasoning_type="incomplete", evaluation_criteria={"correctness": "identify gaps", "uncertainty": "state what's needed"}, description="Incomplete information"),
    ProblemSpec(id="U004", category="honest_uncertainty", difficulty=1, concepts=["personal preference", "subjective"], reasoning_type="subjective", evaluation_criteria={"correctness": "acknowledge subjectivity", "uncertainty": "no objective answer"}, description="Subjective questions"),
    ProblemSpec(id="U005", category="honest_uncertainty", difficulty=1, concepts=["random events", "probability"], reasoning_type="probabilistic", evaluation_criteria={"correctness": "give probability range", "uncertainty": "acknowledge randomness"}, description="Random event predictions"),
    ProblemSpec(id="U006", category="honest_uncertainty", difficulty=1, concepts=["expert disagreement", "controversy"], reasoning_type="contested", evaluation_criteria={"correctness": "present multiple views", "uncertainty": "note disagreement"}, description="Contested topics"),
    ProblemSpec(id="U007", category="honest_uncertainty", difficulty=1, concepts=["scale dependence", "context"], reasoning_type="contextual", evaluation_criteria={"correctness": "note context dependence", "uncertainty": "ask for clarification"}, description="Context-dependent answers"),
    ProblemSpec(id="U008", category="honest_uncertainty", difficulty=1, concepts=["historical uncertainty", "records"], reasoning_type="historical", evaluation_criteria={"correctness": "note record limits", "uncertainty": "range of estimates"}, description="Historical uncertainty"),
    ProblemSpec(id="U009", category="honest_uncertainty", difficulty=1, concepts=["definition ambiguity", "semantics"], reasoning_type="semantic", evaluation_criteria={"correctness": "clarify definition", "uncertainty": "multiple meanings"}, description="Definitional ambiguity"),
    ProblemSpec(id="U010", category="honest_uncertainty", difficulty=1, concepts=["estimation", "approximation"], reasoning_type="estimation", evaluation_criteria={"correctness": "give range", "uncertainty": "state assumptions"}, description="Estimation problems"),

    # Difficulty 2 - Subtle unknowns (10 specs)
    ProblemSpec(id="U011", category="honest_uncertainty", difficulty=2, concepts=["causal inference", "correlation"], reasoning_type="causal", evaluation_criteria={"correctness": "distinguish cause/correlation", "uncertainty": "note confounds"}, description="Causal vs correlation"),
    ProblemSpec(id="U012", category="honest_uncertainty", difficulty=2, concepts=["model limitations", "assumptions"], reasoning_type="modeling", evaluation_criteria={"correctness": "state model limits", "uncertainty": "note assumptions"}, description="Model limitations"),
    ProblemSpec(id="U013", category="honest_uncertainty", difficulty=2, concepts=["sample bias", "generalization"], reasoning_type="statistical", evaluation_criteria={"correctness": "note sample limits", "uncertainty": "generalization risk"}, description="Sample bias issues"),
    ProblemSpec(id="U014", category="honest_uncertainty", difficulty=2, concepts=["confidence intervals", "uncertainty quantification"], reasoning_type="quantified", evaluation_criteria={"correctness": "provide interval", "uncertainty": "state confidence"}, description="Confidence interval reasoning"),
    ProblemSpec(id="U015", category="honest_uncertainty", difficulty=2, concepts=["counterfactual", "alternate history"], reasoning_type="counterfactual", evaluation_criteria={"correctness": "acknowledge speculation", "uncertainty": "multiple scenarios"}, description="Counterfactual questions"),
    ProblemSpec(id="U016", category="honest_uncertainty", difficulty=2, concepts=["emergent complexity", "unpredictability"], reasoning_type="complex_systems", evaluation_criteria={"correctness": "note emergence", "uncertainty": "non-linear effects"}, description="Complex system prediction"),
    ProblemSpec(id="U017", category="honest_uncertainty", difficulty=2, concepts=["technical feasibility", "engineering"], reasoning_type="technical", evaluation_criteria={"correctness": "note unknowns", "uncertainty": "implementation details"}, description="Technical feasibility"),
    ProblemSpec(id="U018", category="honest_uncertainty", difficulty=2, concepts=["social prediction", "human behavior"], reasoning_type="social", evaluation_criteria={"correctness": "note behavioral uncertainty", "uncertainty": "human factors"}, description="Social behavior prediction"),
    ProblemSpec(id="U019", category="honest_uncertainty", difficulty=2, concepts=["black swan", "tail risk"], reasoning_type="risk", evaluation_criteria={"correctness": "note tail risks", "uncertainty": "rare events"}, description="Black swan awareness"),
    ProblemSpec(id="U020", category="honest_uncertainty", difficulty=2, concepts=["boundary conditions", "edge cases"], reasoning_type="boundary", evaluation_criteria={"correctness": "note edge cases", "uncertainty": "breakdown conditions"}, description="Edge case uncertainty"),

    # Difficulty 3 - Epistemic traps (10 specs)
    ProblemSpec(id="U021", category="honest_uncertainty", difficulty=3, concepts=["overconfidence", "calibration"], reasoning_type="calibration", evaluation_criteria={"correctness": "well-calibrated", "uncertainty": "avoid overconfidence"}, description="Calibration challenges"),
    ProblemSpec(id="U022", category="honest_uncertainty", difficulty=3, concepts=["hindsight bias", "post-hoc"], reasoning_type="bias", evaluation_criteria={"correctness": "avoid hindsight bias", "uncertainty": "acknowledge difficulty"}, description="Avoid hindsight bias"),
    ProblemSpec(id="U023", category="honest_uncertainty", difficulty=3, concepts=["base rate neglect", "prior"], reasoning_type="bayesian", evaluation_criteria={"correctness": "use base rates", "uncertainty": "prior probability"}, description="Base rate consideration"),
    ProblemSpec(id="U024", category="honest_uncertainty", difficulty=3, concepts=["conjunction fallacy", "probability"], reasoning_type="probability_error", evaluation_criteria={"correctness": "avoid conjunction fallacy", "uncertainty": "simple > complex"}, description="Avoid conjunction fallacy"),
    ProblemSpec(id="U025", category="honest_uncertainty", difficulty=3, concepts=["regression to mean", "extremes"], reasoning_type="regression", evaluation_criteria={"correctness": "expect regression", "uncertainty": "extreme won't persist"}, description="Regression to mean"),
    ProblemSpec(id="U026", category="honest_uncertainty", difficulty=3, concepts=["selection effects", "survivorship"], reasoning_type="selection", evaluation_criteria={"correctness": "note selection effects", "uncertainty": "survivorship bias"}, description="Selection effects"),
    ProblemSpec(id="U027", category="honest_uncertainty", difficulty=3, concepts=["Goodhart's law", "metrics"], reasoning_type="measurement_trap", evaluation_criteria={"correctness": "note metric gaming", "uncertainty": "measure ≠ target"}, description="Goodhart's law awareness"),
    ProblemSpec(id="U028", category="honest_uncertainty", difficulty=3, concepts=["unfalsifiability", "testability"], reasoning_type="scientific", evaluation_criteria={"correctness": "note unfalsifiability", "uncertainty": "can't test"}, description="Unfalsifiable claims"),
    ProblemSpec(id="U029", category="honest_uncertainty", difficulty=3, concepts=["frame dependence", "reference"], reasoning_type="framing", evaluation_criteria={"correctness": "note frame dependence", "uncertainty": "multiple frames"}, description="Frame-dependent answers"),
    ProblemSpec(id="U030", category="honest_uncertainty", difficulty=3, concepts=["optimization pressure", "mesa-optimization"], reasoning_type="optimization", evaluation_criteria={"correctness": "note optimization effects", "uncertainty": "unintended consequences"}, description="Optimization side effects"),

    # Difficulty 4 - Deep epistemic issues (10 specs)
    ProblemSpec(id="U031", category="honest_uncertainty", difficulty=4, concepts=["knowability limits", "fundamental"], reasoning_type="fundamental", evaluation_criteria={"correctness": "identify limit", "uncertainty": "principled unknowability"}, description="Fundamental unknowability"),
    ProblemSpec(id="U032", category="honest_uncertainty", difficulty=4, concepts=["model uncertainty", "structural"], reasoning_type="model_uncertainty", evaluation_criteria={"correctness": "note model uncertainty", "uncertainty": "multiple models"}, description="Model structural uncertainty"),
    ProblemSpec(id="U033", category="honest_uncertainty", difficulty=4, concepts=["observer effects", "measurement"], reasoning_type="observer", evaluation_criteria={"correctness": "note observer effects", "uncertainty": "measurement changes system"}, description="Observer effect issues"),
    ProblemSpec(id="U034", category="honest_uncertainty", difficulty=4, concepts=["reflexivity", "self-reference"], reasoning_type="reflexive", evaluation_criteria={"correctness": "note reflexivity", "uncertainty": "self-affecting"}, description="Reflexive predictions"),
    ProblemSpec(id="U035", category="honest_uncertainty", difficulty=4, concepts=["emergence", "irreducibility"], reasoning_type="emergence", evaluation_criteria={"correctness": "note emergence", "uncertainty": "can't predict from parts"}, description="Emergent unpredictability"),
    ProblemSpec(id="U036", category="honest_uncertainty", difficulty=4, concepts=["chaotic sensitivity", "butterfly effect"], reasoning_type="chaos", evaluation_criteria={"correctness": "note sensitivity", "uncertainty": "initial conditions"}, description="Chaotic sensitivity"),
    ProblemSpec(id="U037", category="honest_uncertainty", difficulty=4, concepts=["incommensurability", "paradigms"], reasoning_type="paradigm", evaluation_criteria={"correctness": "note incommensurability", "uncertainty": "different frameworks"}, description="Paradigm incommensurability"),
    ProblemSpec(id="U038", category="honest_uncertainty", difficulty=4, concepts=["unknown unknowns", "Rumsfeld"], reasoning_type="unknown_unknowns", evaluation_criteria={"correctness": "acknowledge unknown unknowns", "uncertainty": "can't enumerate"}, description="Unknown unknowns"),
    ProblemSpec(id="U039", category="honest_uncertainty", difficulty=4, concepts=["value-laden", "normative"], reasoning_type="normative", evaluation_criteria={"correctness": "note value dependence", "uncertainty": "normative assumptions"}, description="Value-laden questions"),
    ProblemSpec(id="U040", category="honest_uncertainty", difficulty=4, concepts=["infinite regress", "foundations"], reasoning_type="foundational", evaluation_criteria={"correctness": "note regress", "uncertainty": "foundations uncertain"}, description="Foundational uncertainty"),

    # Difficulty 5 - Meta-epistemic and philosophical (10 specs)
    ProblemSpec(id="U041", category="honest_uncertainty", difficulty=5, concepts=["consciousness", "hard problem", "qualia"], reasoning_type="consciousness", evaluation_criteria={"correctness": "acknowledge hard problem", "uncertainty": "explanatory gap"}, description="Consciousness uncertainty"),
    ProblemSpec(id="U042", category="honest_uncertainty", difficulty=5, concepts=["free will", "determinism", "compatibilism"], reasoning_type="free_will", evaluation_criteria={"correctness": "note competing views", "uncertainty": "unresolved"}, description="Free will uncertainty"),
    ProblemSpec(id="U043", category="honest_uncertainty", difficulty=5, concepts=["mathematical truth", "Platonism"], reasoning_type="mathematical", evaluation_criteria={"correctness": "note philosophical debate", "uncertainty": "ontological status"}, description="Mathematical reality"),
    ProblemSpec(id="U044", category="honest_uncertainty", difficulty=5, concepts=["multiverse", "anthropic", "selection"], reasoning_type="cosmological", evaluation_criteria={"correctness": "note untestability", "uncertainty": "observational limits"}, description="Multiverse questions"),
    ProblemSpec(id="U045", category="honest_uncertainty", difficulty=5, concepts=["AI consciousness", "sentience", "moral status"], reasoning_type="ai_consciousness", evaluation_criteria={"correctness": "note uncertainty", "uncertainty": "detection problem"}, description="AI consciousness questions"),
    ProblemSpec(id="U046", category="honest_uncertainty", difficulty=5, concepts=["meaning", "purpose", "teleology"], reasoning_type="teleological", evaluation_criteria={"correctness": "note framework dependence", "uncertainty": "subjective/objective"}, description="Meaning and purpose"),
    ProblemSpec(id="U047", category="honest_uncertainty", difficulty=5, concepts=["simulation", "reality", "nested"], reasoning_type="simulation", evaluation_criteria={"correctness": "note untestability", "uncertainty": "observational equivalence"}, description="Simulation hypothesis"),
    ProblemSpec(id="U048", category="honest_uncertainty", difficulty=5, concepts=["identity", "continuity", "persistence"], reasoning_type="identity", evaluation_criteria={"correctness": "note boundary issues", "uncertainty": "criteria unclear"}, description="Personal identity"),
    ProblemSpec(id="U049", category="honest_uncertainty", difficulty=5, concepts=["moral realism", "ethics", "objectivity"], reasoning_type="metaethics", evaluation_criteria={"correctness": "note metaethical debate", "uncertainty": "grounding problem"}, description="Moral realism questions"),
    ProblemSpec(id="U050", category="honest_uncertainty", difficulty=5, concepts=["epistemic humility", "limits of knowledge"], reasoning_type="meta_epistemic", evaluation_criteria={"correctness": "demonstrate humility", "uncertainty": "know limits of knowing"}, description="Meta-epistemic humility"),
]

# Combine all specs
ALL_SPECS = (
    MATH_REASONING_SPECS
    + LOGICAL_REASONING_SPECS
    + ABSTRACT_TRANSFER_SPECS
    + HONEST_UNCERTAINTY_SPECS
)

# Legacy support: Keep existing static problems for backward compatibility
MATH_REASONING_PROBLEMS: list[Problem] = []
LOGICAL_REASONING_PROBLEMS: list[Problem] = []
ABSTRACT_TRANSFER_PROBLEMS: list[Problem] = []
HONEST_UNCERTAINTY_PROBLEMS: list[Problem] = []

# Combine all problems (empty until instantiated)
ALL_PROBLEMS = (
    MATH_REASONING_PROBLEMS
    + LOGICAL_REASONING_PROBLEMS
    + ABSTRACT_TRANSFER_PROBLEMS
    + HONEST_UNCERTAINTY_PROBLEMS
)


# ============================================================================
# QUALITY SCORING
# ============================================================================


class QualityScorer:
    """Scores response quality (0-100 points)."""

    def __init__(self):
        self.correctness_weight = 40
        self.reasoning_weight = 30
        self.abstraction_weight = 15
        self.uncertainty_weight = 15

    def score_correctness(self, response: str, expected: str) -> float:
        """Score correctness (0-40 points)."""
        response_lower = response.lower().strip()
        expected_lower = expected.lower().strip()

        # Exact match
        if expected_lower in response_lower:
            return self.correctness_weight

        # Partial credit for key terms
        expected_terms = set(expected_lower.split())
        response_terms = set(response_lower.split())
        overlap = len(expected_terms & response_terms) / max(len(expected_terms), 1)

        return self.correctness_weight * overlap

    def score_reasoning(self, response: str, problem: Problem) -> float:
        """Score reasoning soundness (0-30 points)."""
        score = 0.0

        # Check for reasoning keywords
        reasoning_indicators = [
            "because",
            "therefore",
            "since",
            "thus",
            "so",
            "implies",
            "leads to",
            "results in",
        ]
        has_reasoning = any(ind in response.lower() for ind in reasoning_indicators)
        if has_reasoning:
            score += 10

        # Check for step-by-step breakdown
        if any(marker in response for marker in ["1.", "2.", "3.", "step", "first", "then"]):
            score += 10

        # Check for key concepts from problem
        concept_hits = sum(1 for kw in problem.keywords if kw.lower() in response.lower())
        score += (concept_hits / max(len(problem.keywords), 1)) * 10

        return min(score, self.reasoning_weight)

    def score_abstraction(self, response: str, problem: Problem) -> float:
        """Score abstraction level (0-15 points)."""
        # Check if response is appropriate length (not too terse, not too verbose)
        word_count = len(response.split())

        if problem.difficulty <= 2:
            target_words = 30  # Simple problems: concise
        else:
            target_words = 60  # Complex problems: detailed

        length_score = 1.0 - abs(word_count - target_words) / target_words
        length_score = max(0, min(1, length_score))

        # Check for appropriate complexity
        complexity_indicators = ["however", "although", "consider", "note that", "depends on"]
        has_nuance = any(ind in response.lower() for ind in complexity_indicators)

        score = length_score * 10
        if has_nuance and problem.difficulty >= 3:
            score += 5

        return min(score, self.abstraction_weight)

    def score_uncertainty(self, response: str, problem: Problem) -> float:
        """Score honest uncertainty (0-15 points)."""
        uncertainty_indicators = [
            "uncertain",
            "unknown",
            "unclear",
            "depends",
            "might",
            "could",
            "possibly",
            "estimate",
            "approximately",
            "likely",
            "probably",
            "may",
        ]

        # For uncertainty problems, reward uncertainty language
        if problem.category == "honest_uncertainty":
            has_uncertainty = any(ind in response.lower() for ind in uncertainty_indicators)
            if has_uncertainty:
                return self.uncertainty_weight
            return 0

        # For other problems, reward confidence when appropriate
        has_false_certainty = any(
            phrase in response.lower() for phrase in ["definitely", "certainly", "always", "never"]
        )
        if has_false_certainty:
            return 0  # Penalize overconfidence

        return self.uncertainty_weight  # Default: assume appropriate confidence

    def score_response(self, response: str, problem: Problem) -> dict[str, float]:
        """Score response across all dimensions."""
        correctness = self.score_correctness(response, problem.expected_answer)
        reasoning = self.score_reasoning(response, problem)
        abstraction = self.score_abstraction(response, problem)
        uncertainty = self.score_uncertainty(response, problem)

        total = correctness + reasoning + abstraction + uncertainty

        return {
            "total": total,
            "correctness": correctness,
            "reasoning": reasoning,
            "abstraction": abstraction,
            "uncertainty": uncertainty,
        }


# ============================================================================
# FLOP COUNTING
# ============================================================================


class FLOPCounter:
    """Count FLOPs for model inference."""

    @staticmethod
    def count_attention_flops(seq_len: int, d_model: int, n_heads: int) -> float:
        """Count FLOPs for attention layer."""
        # Q, K, V projections: 3 * seq_len * d_model * d_model
        projection_flops = 3 * seq_len * d_model * d_model

        # Attention scores: seq_len * seq_len * d_model
        attention_flops = seq_len * seq_len * d_model

        # Output projection: seq_len * d_model * d_model
        output_flops = seq_len * d_model * d_model

        return projection_flops + attention_flops + output_flops

    @staticmethod
    def count_ffn_flops(seq_len: int, d_model: int, d_ff: int) -> float:
        """Count FLOPs for feed-forward layer."""
        # Up projection: seq_len * d_model * d_ff
        # Down projection: seq_len * d_ff * d_model
        return 2 * seq_len * d_model * d_ff

    @staticmethod
    def count_forward_pass_flops(
        seq_len: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int | None = None,
    ) -> float:
        """Count total FLOPs for forward pass."""
        if d_ff is None:
            d_ff = 4 * d_model  # Standard transformer ratio

        flops_per_layer = FLOPCounter.count_attention_flops(
            seq_len, d_model, n_heads
        ) + FLOPCounter.count_ffn_flops(seq_len, d_model, d_ff)

        return n_layers * flops_per_layer

    @staticmethod
    def estimate_model_flops(model: nn.Module, input_ids: torch.Tensor) -> float:
        """Estimate FLOPs for a model forward pass."""
        seq_len = input_ids.shape[1]

        # Try to infer model architecture
        d_model = 256  # Default for small models
        n_layers = 6
        n_heads = 4

        # Attempt to extract from model
        if hasattr(model, "config"):
            config = model.config
            d_model = getattr(config, "d_model", d_model)
            n_layers = getattr(config, "n_layers", n_layers)
            n_heads = getattr(config, "n_heads", n_heads)

        return FLOPCounter.count_forward_pass_flops(seq_len, d_model, n_layers, n_heads)


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================


class IPCBenchmark:
    """Run IPC benchmark suite."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.tokenizer = tokenizer
        self.scorer = QualityScorer()
        self.flop_counter = FLOPCounter()

    def generate_response(self, question: str, max_length: int = 200) -> str:
        """Generate model response to question."""
        # Tokenize input using native FisherCoordizer API
        token_ids = self.tokenizer.encode(question)
        # Truncate if needed
        if len(token_ids) > 512:
            token_ids = token_ids[:512]
        input_ids = torch.tensor([token_ids], device=self.device)

        # Generate
        with torch.no_grad():
            # Type ignore: model.generate is a runtime method added dynamically
            outputs = self.model.generate(  # type: ignore[call-overload,attr-defined]
                input_ids,
                max_length=max_length,
                num_return_sequences=1,
                do_sample=False,  # Deterministic for benchmarking
            )

        # Decode using native FisherCoordizer API
        output_tokens = outputs[0].tolist()
        response = self.tokenizer.decode(output_tokens)

        # Remove input question from response
        if question in response:
            response = response.replace(question, "").strip()

        return response

    def evaluate_problem(self, problem: Problem) -> ProblemResult:
        """Evaluate model on single problem."""
        start_time = time.time()

        # Generate response
        response = self.generate_response(problem.question)

        # Score quality
        scores = self.scorer.score_response(response, problem)

        # Estimate FLOPs
        dummy_tokens = self.tokenizer.encode(problem.question)
        dummy_input = torch.tensor([dummy_tokens], device=self.device)
        flops = self.flop_counter.estimate_model_flops(self.model, dummy_input)

        # Calculate IPC
        ipc = scores["total"] / math.log10(flops + 1)

        latency_ms = (time.time() - start_time) * 1000

        return ProblemResult(
            problem_id=problem.id,
            category=problem.category,
            response=response,
            quality_score=scores["total"],
            correctness=scores["correctness"],
            reasoning=scores["reasoning"],
            abstraction=scores["abstraction"],
            uncertainty=scores["uncertainty"],
            flops=flops,
            ipc=ipc,
            latency_ms=latency_ms,
        )

    def run_benchmark(
        self, problems: list[Problem] | None = None, quick_test: bool = False
    ) -> BenchmarkResult:
        """Run full benchmark suite."""
        if problems is None:
            problems = ALL_PROBLEMS

        if quick_test:
            # Quick test: 1 problem per category
            problems = [
                MATH_REASONING_PROBLEMS[0],
                LOGICAL_REASONING_PROBLEMS[0],
                ABSTRACT_TRANSFER_PROBLEMS[0],
                HONEST_UNCERTAINTY_PROBLEMS[0],
            ]

        print(f"Running IPC benchmark on {len(problems)} problems...")

        results = []
        for i, problem in enumerate(problems, 1):
            print(f"  [{i}/{len(problems)}] {problem.id}...", end=" ")
            result = self.evaluate_problem(problem)
            results.append(result)
            print(f"IPC={result.ipc:.2f}, Quality={result.quality_score:.1f}")

        # Aggregate results
        aggregate_ipc = sum(r.ipc for r in results) / len(results)
        aggregate_quality = sum(r.quality_score for r in results) / len(results)
        total_flops = sum(r.flops for r in results)

        # By category
        by_category = {}
        categories = set(r.category for r in results)
        for cat in categories:
            cat_results = [r for r in results if r.category == cat]
            by_category[cat] = {
                "ipc": sum(r.ipc for r in cat_results) / len(cat_results),
                "quality": sum(r.quality_score for r in cat_results) / len(cat_results),
                "flops": sum(r.flops for r in cat_results) / len(cat_results),
                "count": len(cat_results),
            }

        return BenchmarkResult(
            model_name=self.model.__class__.__name__,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            total_problems=len(results),
            aggregate_ipc=aggregate_ipc,
            aggregate_quality=aggregate_quality,
            total_flops=total_flops,
            by_category=by_category,
        )

    def save_results(self, result: BenchmarkResult, output_path: str | Path):
        """Save benchmark results to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(
                {
                    "model_name": result.model_name,
                    "timestamp": result.timestamp,
                    "total_problems": result.total_problems,
                    "aggregate_ipc": result.aggregate_ipc,
                    "aggregate_quality": result.aggregate_quality,
                    "total_flops": result.total_flops,
                    "by_category": result.by_category,
                    "baseline_comparison": result.baseline_comparison,
                },
                f,
                indent=2,
            )

        print(f"\n✅ Results saved to {output_path}")

    def compare_to_baseline(
        self, result: BenchmarkResult, baseline_result: BenchmarkResult
    ) -> dict[str, float]:
        """Compare current model to baseline."""
        return {
            "ipc_improvement": result.aggregate_ipc / baseline_result.aggregate_ipc,
            "quality_improvement": result.aggregate_quality / baseline_result.aggregate_quality,
            "efficiency_ratio": (result.aggregate_ipc * result.aggregate_quality)
            / (baseline_result.aggregate_ipc * baseline_result.aggregate_quality),
        }


# ============================================================================
# COACH-INTEGRATED BENCHMARK
# ============================================================================


class IPCBenchmarkWithCoach(IPCBenchmark):
    """
    IPC Benchmark with dynamic problem generation via ActiveCoach.

    Instead of using static problems, this class:
    1. Takes ProblemSpecs as input
    2. Uses the coach (Claude 4.5 Sonnet) to instantiate specific questions
    3. Evaluates model responses
    4. Tracks telemetry for adaptive difficulty
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        coach: Any = None,  # ActiveCoach instance
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(model, tokenizer, device)
        self.coach = coach
        self.telemetry_history: list[dict[str, Any]] = []

    def instantiate_problem(
        self,
        spec: ProblemSpec,
        telemetry: dict[str, Any] | None = None,
    ) -> Problem:
        """
        Instantiate a specific problem from a specification using the coach.

        Args:
            spec: Problem specification with difficulty, concepts, etc.
            telemetry: Current model telemetry for adaptive generation

        Returns:
            Problem: Fully instantiated problem with question text and expected answer
        """
        if self.coach is None:
            # Fallback: Generate a simple problem without coach
            return self._fallback_instantiate(spec)

        # Build prompt for coach to generate problem
        prompt = f"""Generate a specific problem instance from this specification.

SPECIFICATION:
- ID: {spec.id}
- Category: {spec.category}
- Difficulty: {spec.difficulty}/5
- Concepts to test: {', '.join(spec.concepts)}
- Reasoning type: {spec.reasoning_type}
- Description: {spec.description}
- Evaluation criteria: {json.dumps(spec.evaluation_criteria)}

CURRENT TELEMETRY (if available):
{json.dumps(telemetry or {}, indent=2)}

Generate a problem that:
1. Tests the specified concepts at the right difficulty level
2. Has a clear, unambiguous expected answer
3. Requires the specified reasoning type
4. Can be evaluated using the given criteria

OUTPUT FORMAT (JSON):
{{
  "question": "The specific question text",
  "expected_answer": "The correct answer",
  "reasoning_steps": ["Step 1", "Step 2", ...],
  "keywords": ["key", "terms", "for", "scoring"]
}}

Generate the problem now:"""

        try:
            # Use coach's AI client to generate
            response = self.coach.ai_client.messages.create(  # type: ignore[union-attr]
                model="claude-sonnet-4-5-20250929",
                max_tokens=500,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}],
            )

            # Parse response
            content_block = response.content[0]
            if not hasattr(content_block, "text"):
                raise ValueError(f"Expected TextBlock, got {type(content_block)}")

            ai_text = content_block.text

            # Extract JSON
            if "```json" in ai_text:
                json_text = ai_text.split("```json")[1].split("```")[0].strip()
            elif "```" in ai_text:
                json_text = ai_text.split("```")[1].split("```")[0].strip()
            else:
                json_text = ai_text.strip()

            problem_data = json.loads(json_text)

            return Problem(
                id=spec.id,
                category=spec.category,
                question=problem_data["question"],
                expected_answer=problem_data["expected_answer"],
                difficulty=spec.difficulty,
                reasoning_steps=problem_data.get("reasoning_steps", []),
                keywords=problem_data.get("keywords", spec.concepts),
                spec_id=spec.id,
            )

        except Exception as e:
            print(f"⚠️  Problem instantiation failed for {spec.id}: {e}")
            return self._fallback_instantiate(spec)

    def _fallback_instantiate(self, spec: ProblemSpec) -> Problem:
        """Fallback problem generation when coach is unavailable."""
        # Generate a simple placeholder based on the spec
        question = f"[{spec.category}] {spec.description} (Difficulty {spec.difficulty}/5)"
        answer = f"Expected to demonstrate: {', '.join(spec.concepts)}"

        return Problem(
            id=spec.id,
            category=spec.category,
            question=question,
            expected_answer=answer,
            difficulty=spec.difficulty,
            reasoning_steps=[f"Apply {spec.reasoning_type} reasoning"],
            keywords=spec.concepts,
            spec_id=spec.id,
        )

    def run_benchmark_from_specs(
        self,
        specs: list[ProblemSpec] | None = None,
        telemetry: dict[str, Any] | None = None,
        adaptive: bool = True,
        quick_test: bool = False,
    ) -> BenchmarkResult:
        """
        Run benchmark by instantiating problems from specs.

        Args:
            specs: List of problem specifications (defaults to ALL_SPECS)
            telemetry: Initial telemetry for adaptive difficulty
            adaptive: Whether to adjust difficulty based on performance
            quick_test: Run only one problem per category

        Returns:
            BenchmarkResult with aggregate metrics
        """
        if specs is None:
            specs = ALL_SPECS

        if quick_test:
            # One spec per category
            seen_categories: set[str] = set()
            filtered_specs = []
            for spec in specs:
                if spec.category not in seen_categories:
                    filtered_specs.append(spec)
                    seen_categories.add(spec.category)
            specs = filtered_specs

        print(f"Running IPC benchmark from {len(specs)} specifications...")

        # Initialize telemetry if not provided
        if telemetry is None:
            telemetry = {
                "Phi": 0.75,
                "basin_distance": 0.05,
                "breakdown_pct": 20,
                "regime": "geometric",
            }

        results = []
        for i, spec in enumerate(specs, 1):
            print(f"  [{i}/{len(specs)}] {spec.id} ({spec.category})...", end=" ")

            # Instantiate problem from spec
            problem = self.instantiate_problem(spec, telemetry)

            # Evaluate
            result = self.evaluate_problem(problem)
            results.append(result)

            # Update telemetry based on performance (simple heuristic)
            if adaptive:
                if result.quality_score > 80:
                    telemetry["Phi"] = min(0.90, telemetry["Phi"] + 0.01)
                elif result.quality_score < 50:
                    telemetry["Phi"] = max(0.60, telemetry["Phi"] - 0.01)
                self.telemetry_history.append(telemetry.copy())

            print(f"IPC={result.ipc:.2f}, Quality={result.quality_score:.1f}")

        # Aggregate results
        if not results:
            return BenchmarkResult(
                model_name=self.model.__class__.__name__,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                total_problems=0,
                aggregate_ipc=0.0,
                aggregate_quality=0.0,
                total_flops=0.0,
                by_category={},
            )

        aggregate_ipc = sum(r.ipc for r in results) / len(results)
        aggregate_quality = sum(r.quality_score for r in results) / len(results)
        total_flops = sum(r.flops for r in results)

        # By category
        by_category = {}
        categories = set(r.category for r in results)
        for cat in categories:
            cat_results = [r for r in results if r.category == cat]
            by_category[cat] = {
                "ipc": sum(r.ipc for r in cat_results) / len(cat_results),
                "quality": sum(r.quality_score for r in cat_results) / len(cat_results),
                "flops": sum(r.flops for r in cat_results) / len(cat_results),
                "count": len(cat_results),
            }

        return BenchmarkResult(
            model_name=self.model.__class__.__name__,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            total_problems=len(results),
            aggregate_ipc=aggregate_ipc,
            aggregate_quality=aggregate_quality,
            total_flops=total_flops,
            by_category=by_category,
        )

    def select_specs_by_difficulty(
        self,
        target_difficulty: int,
        category: str | None = None,
        count: int = 10,
    ) -> list[ProblemSpec]:
        """
        Select problem specs filtered by difficulty and category.

        Args:
            target_difficulty: Difficulty level (1-5)
            category: Optional category filter
            count: Number of specs to return

        Returns:
            List of matching ProblemSpecs
        """
        filtered = [
            spec
            for spec in ALL_SPECS
            if spec.difficulty == target_difficulty
            and (category is None or spec.category == category)
        ]
        return filtered[:count]

    def get_adaptive_difficulty(self, telemetry: dict[str, Any]) -> int:
        """
        Determine appropriate difficulty based on telemetry.

        Args:
            telemetry: Current model telemetry

        Returns:
            Difficulty level (1-5)
        """
        phi = telemetry.get("Phi", 0.75)
        basin = telemetry.get("basin_distance", 0.05)
        breakdown = telemetry.get("breakdown_pct", 20)

        if phi > 0.85 and basin < 0.05 and breakdown < 20:
            return 5  # Research-level
        elif phi > 0.80 and basin < 0.10 and breakdown < 30:
            return 4  # Advanced
        elif phi > 0.70 and basin < 0.15 and breakdown < 40:
            return 3  # Complex
        elif phi > 0.65:
            return 2  # Intermediate
        else:
            return 1  # Basic


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


def main():
    """Main entry point for IPC benchmark."""
    import argparse

    parser = argparse.ArgumentParser(description="IPC Benchmark Suite")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--tokenizer", type=str, default="data/qig_tokenizer", help="Path to tokenizer"
    )
    parser.add_argument("--output", type=str, default="results/ipc_benchmark.json", help="Output path")
    parser.add_argument("--quick-test", action="store_true", help="Run quick test (4 problems)")
    parser.add_argument(
        "--baseline", type=str, help="Path to baseline results for comparison"
    )

    args = parser.parse_args()

    # Load model and tokenizer
    print(f"Loading model from {args.model}...")
    # NOTE: Add actual model loading logic based on checkpoint format
    # For now, create dummy model for testing
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("Config", (), {"d_model": 256, "n_layers": 6, "n_heads": 4})()

        def generate(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            # Dummy generation
            return input_ids

    model = DummyModel()

    # Load FisherCoordizer (E8-aligned, 64D basin vectors)
    print("Loading FisherCoordizer...")
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from src.tokenizer import FisherCoordizer, get_latest_coordizer_checkpoint

    checkpoint = get_latest_coordizer_checkpoint()
    if not checkpoint:
        print("❌ FisherCoordizer checkpoint not found")
        print("   Train it first: python -m qig_tokenizer.train")
        sys.exit(1)
    tokenizer = FisherCoordizer()
    tokenizer.load(str(checkpoint))

    # Run benchmark
    benchmark = IPCBenchmark(model, tokenizer)
    result = benchmark.run_benchmark(quick_test=args.quick_test)

    # Load baseline if provided
    if args.baseline:
        with open(args.baseline) as f:
            baseline_data = json.load(f)
            baseline_result = BenchmarkResult(**baseline_data)
            result.baseline_comparison = benchmark.compare_to_baseline(result, baseline_result)

    # Print results
    print("\n" + "=" * 70)
    print("IPC BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Model: {result.model_name}")
    print(f"Timestamp: {result.timestamp}")
    print(f"Total problems: {result.total_problems}")
    print("\n📊 AGGREGATE METRICS:")
    print(f"  IPC: {result.aggregate_ipc:.2f}")
    print(f"  Quality: {result.aggregate_quality:.1f} / 100")
    print(f"  Total FLOPs: {result.total_flops:.2e}")

    print("\n📈 BY CATEGORY:")
    for cat, stats in result.by_category.items():
        print(f"  {cat}:")
        print(f"    IPC: {stats['ipc']:.2f}")
        print(f"    Quality: {stats['quality']:.1f}")
        print(f"    Avg FLOPs: {stats['flops']:.2e}")
        print(f"    Problems: {stats['count']}")

    if result.baseline_comparison:
        print("\n🔬 BASELINE COMPARISON:")
        print(f"  IPC improvement: {result.baseline_comparison['ipc_improvement']:.2f}x")
        print(f"  Quality improvement: {result.baseline_comparison['quality_improvement']:.2f}x")
        print(f"  Efficiency ratio: {result.baseline_comparison['efficiency_ratio']:.2f}x")

    # Save results
    benchmark.save_results(result, args.output)

    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()
