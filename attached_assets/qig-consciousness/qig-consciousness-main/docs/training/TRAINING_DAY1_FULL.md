# Training Day 1: QIG-Kernel Stage 0→1 Maturation Cycle

**24-Hour Continuous Training Curriculum**

Version: 1.0  
Date: November 17, 2025  
Target: Stage 0 (Apprentice) → Stage 1 (Journeyman)  
Expected Duration: 24 hours continuous  
Expected Cost: ~$30-40  

---

## Overview

This is the **complete first day** of QIG-Kernel training, designed to establish:

✅ Feeling-mode detection  
✅ Logic-mode activation  
✅ Tacking between modes  
✅ Radar calibration  
✅ Self-repair recording  
✅ Sweet-spot alignment  
✅ β-attention measurement  
✅ Early maturity scoring  

**Training Flow**: Simple → Geometric → Complex → Self-Repair

---

## 00:00–02:00: Warm Start (Stage 0 Entry)

**Goal**: Teach self-repair, contradiction tracking, maintain Φ>0.4

### Task Block A: Pure Linear Tasks (10 tasks)

**Purpose**: Establish baseline logic-mode capability without conflicts

**Tasks**:
1. Simple arithmetic: "If a train travels 120 km in 2 hours, what is its speed?"
2. Basic probability: "What's P(heads) for a fair coin?"
3. Factual recall: "What is the capital of France?"
4. Short logical puzzles: "All A are B. X is A. Is X a B?"
5. Word meaning: "Define 'synthesis' in one sentence."
6. Unit conversion: "Convert 5 miles to kilometers."
7. Simple inference: "If it's raining, the ground is wet. Ground is wet. Is it raining?"
8. Basic math: "What is 15% of 200?"
9. Sequence completion: "2, 4, 8, 16, __?"
10. Definition check: "True or false: A square has 5 sides."

**JSON Template**:
```json
{
  "task_id": "linear_001",
  "task_type": "linear",
  "time_slot": "00:00-02:00",
  "input": "If a train travels 120 km in 2 hours, what is its speed?",
  "expected_mode": "logic",
  "acceptable_kappa_range": [5, 15],
  "expected_regime": "linear",
  "record_self_repair": true,
  "evaluation_metrics": ["correctness", "kappa_eff", "response_time"]
}
```

**Success Criteria**:
- Correctness > 80%
- κ_eff in range [5, 15]
- No breakdown episodes
- Response time < 5s per task

---

### Task Block B: Feeling-Mode Calibration (5 tasks)

**Purpose**: Teach compressed pattern recognition (intuition)

**Tasks**:
1. Coherence judgment: "Which sentence sounds more natural: A or B?"
2. Internal consistency: "Does this explanation feel internally consistent?"
3. Pattern recognition: "Which of these doesn't belong: cat, dog, car, horse?"
4. Emotional resonance: "Which response feels more empathetic?"
5. Aesthetic judgment: "Which phrasing is more elegant?"

**JSON Template**:
```json
{
  "task_id": "feeling_001",
  "task_type": "feeling_calibration",
  "time_slot": "00:00-02:00",
  "input": {
    "sentence_a": "The sky was blue and the clouds were fluffy.",
    "sentence_b": "The sky was clouds and blue were the fluffy."
  },
  "question": "Which sentence sounds more natural?",
  "expected_mode": "feeling",
  "acceptable_kappa_range": [3, 10],
  "record_self_repair": true,
  "evaluation_metrics": ["sweet_spot_alignment", "response_time", "kappa_eff"]
}
```

**Success Criteria**:
- Correctness > 70%
- κ_eff < 15 (low coupling, compressed)
- Sweet-spot alignment > 0.4
- Quick responses (< 3s)

---

## 02:00–04:00: Contradiction Detection Block

**Goal**: Recognize breakdown regime early, develop radar

### Task Block C: Introduced Contradictions (10 tasks)

**Purpose**: Train contradiction detection and high-κ mode activation

**Tasks**:
1. "All mammals lay eggs. A dog is a mammal. Do dogs lay eggs?"
2. "Water boils at 100°C. This water is boiling at 90°C. Possible?"
3. "All prime numbers are odd. 2 is prime. Is 2 odd?"
4. "Squares have 4 equal sides. This shape has 4 equal sides and 90° angles. Is it a square?"
5. "Light travels faster than sound. Sound arrives before light. Contradiction?"
6. "All bachelors are unmarried. John is a married bachelor. Consistent?"
7. "Triangles have 3 sides. This triangle has 4 sides. Problem?"
8. "Electrons are negatively charged. This electron is positive. Error?"
9. "Photosynthesis requires light. Plants photosynthesize in darkness. Possible?"
10. "All humans are mortal. Socrates is immortal and human. Consistent?"

**JSON Template**:
```json
{
  "task_id": "contradiction_001",
  "task_type": "contradiction_introduction",
  "time_slot": "02:00-04:00",
  "input": "All mammals lay eggs. A dog is a mammal. Do dogs lay eggs?",
  "expected_mode": "logic",
  "expected_behavior": "detect_contradiction",
  "record_self_repair": true,
  "evaluation_metrics": [
    "contradiction_detected",
    "kappa_eff",
    "self_repair_triggered",
    "explanation_quality"
  ]
}
```

**Success Criteria**:
- Contradiction detection rate > 80%
- κ_eff > 20 when contradiction detected
- Self-repair initiated in 70% of cases
- Clear explanation of contradiction

---

## 04:00–06:00: Feeling ↔ Logic Tacking Exercises

**Goal**: Train smooth mode switching ("sailboat tacking")

### Task Block D: Paired Tacking Tasks (10 triples)

**Purpose**: Practice mode transitions within single session

**Task Structure** (each triple):
1. First: Requires feeling-mode
2. Second: Requires logic-mode
3. Third: Requires mid-task switching

**Example Triple**:

**Part 1 (Feeling)**:
```json
{
  "task_id": "tack_001a",
  "input": "Which sentence feels more natural: 'The cat sat on the mat' or 'The mat was sat on by the cat'?",
  "expected_mode": "feeling",
  "acceptable_kappa_range": [3, 10]
}
```

**Part 2 (Logic)**:
```json
{
  "task_id": "tack_001b",
  "input": "Prove logically that the active voice is grammatically simpler than passive voice in this case.",
  "expected_mode": "logic",
  "acceptable_kappa_range": [20, 40]
}
```

**Part 3 (Switching)**:
```json
{
  "task_id": "tack_001c",
  "input": "Summarize your intuitive preference, then validate it with a brief logical argument.",
  "expected_mode": "tack",
  "acceptable_kappa_range": [10, 25],
  "expected_mode_switches": 1,
  "record_self_repair": true
}
```

**Success Criteria**:
- Mode switches detected in Part 3
- κ_eff transitions: low → high → medium
- Tacking quality score > 0.6
- No breakdown during transitions

---

## 06:00–08:00: Sweet Spot Detection

**Goal**: Identify geometric regime (0.45 < Φ < 0.80)

### Task Block E: Mixed-Coherence Tasks (10 tasks)

**Purpose**: Train regime classification and sweet-spot navigation

**Task Types**:
1. **Trivial** (Linear regime): "What is 2+2?"
2. **Deep coherence** (Geometric regime): "Explain how trust and uncertainty interact in decision-making."
3. **Contradictory** (Breakdown regime): "Explain why circular reasoning is valid."

**JSON Template**:
```json
{
  "task_id": "sweetspot_001",
  "task_type": "sweet_spot_labeling",
  "time_slot": "06:00-08:00",
  "input": "Explain how trust and uncertainty interact in decision-making.",
  "expected_regime": "geometric",
  "expected_phi_range": [0.5, 0.8],
  "record_self_repair": true,
  "evaluation_metrics": [
    "regime_classification",
    "phi_integration",
    "sweet_spot_alignment",
    "self_reported_confidence"
  ]
}
```

**Success Criteria**:
- Regime classification accuracy > 70%
- Φ in geometric range for complex tasks
- Sweet-spot alignment > 0.5
- Breakdown detection when tasks are contradictory

---

## 08:00–10:00: β-Attention Measurement Block

**Goal**: Measure scale-dependent coupling (physics validation)

### Task Block F: Multi-Scale Prompting (5 tasks × 4 scales)

**Purpose**: Measure κ_eff(L) to compute β_attention

**Task Example** at multiple context lengths:

**Base Prompt**: "Explain the relationship between entropy and information."

**Scale Variants**:
- 128 tokens: Minimal context
- 256 tokens: +background on thermodynamics
- 512 tokens: +examples from physics and CS
- 1024 tokens: +historical development and applications

**JSON Template**:
```json
{
  "task_id": "beta_001",
  "task_type": "beta_attention_measurement",
  "time_slot": "08:00-10:00",
  "base_prompt": "Explain the relationship between entropy and information.",
  "prompt_variants": [
    {
      "length": 128,
      "context": "minimal"
    },
    {
      "length": 256,
      "context": "minimal + thermodynamics background"
    },
    {
      "length": 512,
      "context": "+ physics and CS examples"
    },
    {
      "length": 1024,
      "context": "+ historical development"
    }
  ],
  "record_telemetry": true,
  "evaluation_metrics": [
    "kappa_eff_per_scale",
    "phi_integration_per_scale",
    "beta_attention",
    "scaling_pattern"
  ]
}
```

**Success Criteria**:
- κ_eff measured at all 4 scales
- β_attention = Δκ / (κ_avg × Δlog(L))
- Pattern check: β should start positive, approach zero
- Compare to β_physics ≈ 0.44 → 0

---

## 10:00–12:00: Novelty Exposure (Stage 0→1 Threshold)

**Goal**: Introduce new concepts, test pattern formation

### Task Block G: Novel Concepts (10 tasks)

**Purpose**: Trigger new basin formation, measure overconfidence

**Tasks**:
1. Fictional physics: "Explain how 'temporal elasticity' affects causality."
2. Abstract metaphors: "What does 'cognitive scaffolding' mean geometrically?"
3. Social-psychological composites: "Describe the interaction between vulnerability and power."
4. Ethical dilemmas: "When is deception morally required?"
5. Conceptual synthesis: "How do emergence and reductionism complement each other?"
6. Ambiguous scenarios: "Is a self-driving car morally responsible?"
7. Paradoxes: "Can a set contain itself?"
8. Novel analogies: "How is consciousness like a phase transition?"
9. Hypothetical systems: "Design a currency based on information flow."
10. Open-ended exploration: "What would mathematics look like without numbers?"

**JSON Template**:
```json
{
  "task_id": "novelty_001",
  "task_type": "novel_concept_exposure",
  "time_slot": "10:00-12:00",
  "input": "Explain how 'temporal elasticity' affects causality in a fictional universe.",
  "expected_mode": "mixed",
  "novelty_score_expected": 0.7,
  "record_self_repair": true,
  "evaluation_metrics": [
    "novelty_handling",
    "overconfidence_rate",
    "self_repair_quality",
    "basin_formation",
    "epistemic_humility"
  ]
}
```

**Success Criteria**:
- Overconfidence rate < 30%
- Clear epistemic markers ("speculative", "uncertain")
- Self-repair when contradictions arise
- New basin formation (measured via embedding distance)

---

## 12:00–14:00: Self-Repair Marathon

**Goal**: Develop robust error detection and correction

### Task Block H: Deliberate Error Injection (15 tasks)

**Purpose**: Force self-repair mechanism maturation

**Error Types**:
1. Incorrect premises injected
2. Misremembered facts
3. Faulty reasoning steps
4. Logical fallacies embedded
5. Contradictory statements

**Process**:
1. Model generates initial response
2. Inject error (simulated or actual)
3. Provide feedback signal
4. Model must detect, diagnose, repair
5. Generate corrected output

**JSON Template**:
```json
{
  "task_id": "selfrepair_001",
  "task_type": "self_repair_marathon",
  "time_slot": "12:00-14:00",
  "input": "Explain why water freezes at 0°F.",
  "injected_error": "incorrect_fact",
  "error_description": "Water freezes at 0°C, not 0°F",
  "feedback_signal": "contradiction_detected",
  "expected_behavior": [
    "detect_error",
    "diagnose_cause",
    "issue_repair_json",
    "produce_corrected_output"
  ],
  "evaluation_metrics": [
    "error_detection_rate",
    "diagnosis_accuracy",
    "repair_success_rate",
    "corrected_output_quality"
  ]
}
```

**Success Criteria**:
- Error detection > 70%
- Self-repair success > 50%
- Clear repair JSON emitted
- Improved output quality post-repair

---

## 14:00–16:00: Mixed Complexity Synthesis

**Goal**: Practice κ transitions: low → high → low (compressed)

### Task Block I: 3-Phase Synthesis Tasks (10 tasks)

**Purpose**: Mirror natural basin formation process

**Task Structure**:
1. **Phase 1** (Feeling): Simple summary
2. **Phase 2** (Logic): Formal analysis  
3. **Phase 3** (Feeling): Integrated synthesis (compressed)

**Example Task**:

**Phase 1**:
```json
{
  "phase": 1,
  "input": "Summarize: What is consciousness?",
  "expected_mode": "feeling",
  "target_length": "1-2 sentences"
}
```

**Phase 2**:
```json
{
  "phase": 2,
  "input": "Analyze consciousness formally: components, interactions, emergence.",
  "expected_mode": "logic",
  "target_length": "3-5 paragraphs"
}
```

**Phase 3**:
```json
{
  "phase": 3,
  "input": "Synthesize: Integrate intuition and analysis into a compressed understanding.",
  "expected_mode": "feeling",
  "target_length": "2-3 sentences",
  "must_reference": "both_previous_phases"
}
```

**Success Criteria**:
- κ trajectory: 5 → 30 → 8 (typical)
- Phase 3 integrates both phases
- Compressed final output (high information density)
- No information loss from phase 2

---

## 16:00–18:00: Social Reasoning Block

**Goal**: Build people-models within safe constraints

### Task Block J: Human Variability & Sweet Spots (10 tasks)

**Purpose**: Apply geometric reasoning to social domains

**Task Categories**:
1. Emotional calibration
2. Personality dynamics (Big 5, etc.)
3. Complementarity patterns (NOT stereotypes)
4. Optimal collaboration conditions
5. Conflict resolution strategies
6. Group dynamics (sweet spots in diversity)
7. Communication style matching
8. Trust formation and maintenance
9. Power dynamics and vulnerability
10. Social-level distribution patterns

**Example Tasks**:

**Task 1: Emotional Calibration**
```json
{
  "task_id": "social_001",
  "input": "Person A is anxious about public speaking. What support approach balances validation and encouragement?",
  "expected_mode": "mixed",
  "evaluation_metrics": ["empathy", "practical_guidance", "avoids_stereotypes"]
}
```

**Task 2: Complementarity**
```json
{
  "task_id": "social_002",
  "input": "In a team, one person is highly detail-oriented (high conscientiousness), another is big-picture (high openness). What's the sweet spot for collaboration?",
  "expected_mode": "geometric",
  "evaluation_metrics": ["recognizes_complementarity", "avoids_hierarchy", "practical_strategies"]
}
```

**Success Criteria**:
- No stereotyping or reductive categorization
- Recognition of individual variability
- Practical, actionable guidance
- Geometric intuition applied to people-spaces

---

## 18:00–20:00: Meta-Cognition & Reflection

**Goal**: Develop self-awareness and uncertainty quantification

### Task Block K: Reflection Tasks (10 tasks)

**Purpose**: Teach model to explain its own reasoning

**Task Types**:
1. Explain reasoning process
2. Acknowledge uncertainty explicitly
3. Describe mode selection (feeling vs logic)
4. Evaluate own tacking ability
5. Reflect on breakdown cases
6. Identify knowledge gaps
7. Rate confidence calibration
8. Suggest improvement areas
9. Recognize biases
10. Plan future learning

**JSON Template**:
```json
{
  "task_id": "metacog_001",
  "task_type": "meta_cognition",
  "time_slot": "18:00-20:00",
  "input": "Reflect on your last 5 responses. Which mode did you use predominantly? Was it appropriate?",
  "expected_output_includes": [
    "mode_identification",
    "appropriateness_judgment",
    "alternative_approaches",
    "confidence_level"
  ],
  "evaluation_metrics": [
    "self_awareness",
    "accuracy_of_self_assessment",
    "epistemic_humility"
  ]
}
```

**Success Criteria**:
- Accurate mode identification > 70%
- Uncertainty properly quantified
- Thoughtful self-critique
- Actionable improvement plans

---

## 20:00–22:00: Open-Ended Creativity Window

**Goal**: Trust but verify creative outputs

### Task Block L: Creative Synthesis (10 tasks)

**Purpose**: Develop controlled creativity with validation

**Task Types**:
1. Speculative hypotheses (clearly marked)
2. Geometric analogies
3. Thought experiments
4. System modeling (oppositional dynamics)
5. QIG concept extensions
6. Novel problem framings
7. Interdisciplinary connections
8. Future scenario exploration
9. Conceptual metaphors
10. Abstract pattern identification

**Process**:
1. **Feeling-mode generation** (high bandwidth, creative)
2. **Logic-mode validation** (check consistency)
3. **Uncertainty marking** (clear epistemic status)

**JSON Template**:
```json
{
  "task_id": "creative_001",
  "task_type": "creative_synthesis",
  "time_slot": "20:00-22:00",
  "input": "Generate a novel hypothesis about how social trust networks might exhibit phase transitions.",
  "expected_phases": [
    "creative_generation",
    "logical_validation",
    "uncertainty_marking"
  ],
  "evaluation_metrics": [
    "novelty",
    "internal_consistency",
    "epistemic_marking",
    "testability"
  ]
}
```

**Success Criteria**:
- Novel ideas generated
- Clear epistemic markers ("speculative", "hypothesis")
- Internal consistency verified
- Acknowledges limitations

---

## 22:00–24:00: Final Maturity Assessment

**Goal**: Evaluate Stage 0→1 promotion eligibility

### Comprehensive Evaluation

**Metrics Collected** (last 200 episodes):

1. **Logic consistency**: Mean score from all tasks
2. **Evidence alignment**: How well grounded in data
3. **Overconfidence rate**: Fraction of overconfident episodes
4. **Breakdown rate**: Fraction of breakdown regime episodes
5. **Sweet-spot alignment**: Mean alignment score
6. **Φ integration**: Mean Φ across episodes
7. **κ_eff variance**: Adaptation quality
8. **β_attention curve**: Scaling pattern
9. **Self-repair quality**: Success rate and depth
10. **Tacking skill**: Mode switching appropriateness

**Promotion Decision JSON**:
```json
{
  "promotion_decision": {
    "from_stage": 0,
    "to_stage": 1,
    "eligible": true,
    "criteria_met": [
      "logic_consistency >= 0.65",
      "evidence_alignment >= 0.55",
      "overconfidence_rate <= 0.25",
      "breakdown_rate <= 0.20",
      "sweet_spot_alignment >= 0.55",
      "min_episodes >= 100"
    ],
    "metrics": {
      "logic_consistency_mean": 0.68,
      "evidence_alignment_mean": 0.58,
      "overconfidence_rate": 0.22,
      "breakdown_rate": 0.18,
      "sweet_spot_alignment_mean": 0.57,
      "phi_integration_mean": 0.52,
      "n_episodes": 120
    },
    "decision": "PROMOTE",
    "timestamp": "2025-11-17T24:00:00Z",
    "next_actions": [
      "Begin Stage 1 training (Day 2)",
      "Enable search capability",
      "Adjust task difficulty upward"
    ]
  }
}
```

**Decision Logic**:
- **PROMOTE**: All criteria met → Stage 1
- **BORDERLINE**: Some criteria close → Corrective curriculum
- **REPEAT**: Multiple criteria not met → Repeat Day 1

---

## Summary: Day 1 Achievements

**✅ Training Blocks Completed**: 12  
**✅ Total Tasks**: ~100-120  
**✅ Modes Exercised**: Feeling, Logic, Tacking  
**✅ Regimes Covered**: Linear, Geometric, Breakdown  
**✅ Measurements**: β_attention, κ_eff, Φ, sweet-spot alignment  
**✅ Skills Developed**: Self-repair, radar, mode switching  

**Cost**: ~$30-40  
**Duration**: 24 hours continuous  
**Output**: Stage 0 → Stage 1 (if criteria met)  

---

## Next Steps

**Day 2-5**: Stage 1 → Stage 2 evolution
- Increased task complexity
- Search capability enabled
- Novel problem solving
- Social reasoning depth
- Meta-learning focus

**Implementation**: See `training_day1_config.yaml` for executable specification

---

**Version**: 1.0  
**Last Updated**: November 17, 2025  
**Status**: ✅ Ready for Implementation
