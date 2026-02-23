# AGENTS.md

This file contains agent-specific instructions. See the `README.md`, `docs/architecture.md`, and `docs/roadmap.md` for further guidance.

## E8 Kernel Hierarchy

This repository implements the QIG (Quantum Interactive Geometry) E8 Kernel structure. Please adhere to the thermodynamic protocol and ensure geometric purity (Fisher-Rao distance on the probability simplex) in all modifications.

### Core Principles

1. **Geometric Purity:** No neural networks, Euclidean distance, or cosine similarity. Use `fisher_rao_distance` and log/exp maps on the simplex instead.
2. **Quenched Disorder:** Each instantiated kernel receives a uniquely frozen parameter (`quenched_gain` or slope) acting as intrinsic subjectivity.
3. **Consensus via Resonance:** All generated text arises from Fréchet means and geometric proximity within the probability simplex vocabulary bank.
4. **Constellation Constraints:** Avoid directly modifying `GenesisKernel` beyond its intended orchestration responsibilities. Support the Core-8 hierarchical spawning.

*For specific workflow or subsystem instructions, locate the appropriate `.md` files within `{.agents,.agent,workflows}/` or `skills/`.*
