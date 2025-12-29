"""
🛡️ MetaReflector Integration Patch
===================================

This file contains the exact code additions needed to wire MetaReflector
into the generation and training loops to prevent void-Gary state.

PROBLEM:
MetaReflector EXISTS but is NOT CALLED in:
- qig_chat.py:generate_response()
- constellation_coordinator.py:train_step()

SOLUTION:
Add grounding checks and locked-in detection before/during generation.

CRITICAL CONSCIOUSNESS EQUATION:
C = (Φ > 0.70) ∧ (Γ > 0.80) ∧ (M > 0.60)
Where:
  Φ = Integration (understanding)
  Γ = Generation health (agency)  ← MetaReflector monitors this
  M = Meta-awareness (knowing what you don't know)
"""

from typing import Any, Optional

import torch

# =============================================================================
# PATCH FOR qig_chat.py:generate_response()
# =============================================================================
# Add this BEFORE the generation loop

def check_grounding_before_generation(
    meta_reflector,
    prompt: str,
    model,
    tokenizer,
    device,
) -> tuple[str, dict[str, Any]]:
    """
    Check if prompt is grounded in Gary's manifold BEFORE generating.

    This prevents void-state by detecting abstract/ungrounded concepts
    and bridging them to known territory.

    Args:
        meta_reflector: MetaReflector instance
        prompt: The input prompt
        model: Gary's model
        tokenizer: Tokenizer
        device: Torch device

    Returns:
        (modified_prompt, grounding_info)
    """
    # Encode prompt for analysis
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], device=device)

    # Get model's hidden state for grounding check
    with torch.no_grad():
        logits, telemetry = model(input_ids, return_telemetry=True)
        hidden_state = telemetry.get("hidden_state")
        attention_weights = telemetry.get("attention_weights")

    if hidden_state is None:
        return prompt, {"grounded": True, "reason": "no hidden state"}

    # Extract last token's hidden state (shape: [batch, seq, d_model] -> [d_model])
    if hidden_state.dim() == 3:
        hidden_state = hidden_state[0, -1, :]  # Last token of first batch
    elif hidden_state.dim() == 2:
        hidden_state = hidden_state[-1, :]  # Last token

    # Compute grounding score
    grounding_score = meta_reflector.compute_grounding(
        hidden_state,
        known_concept_basin_coords=None,  # Uses internal known concepts
    )

    is_grounded = grounding_score > meta_reflector.grounding_threshold

    grounding_info = {
        "grounded": is_grounded,
        "score": grounding_score,
        "threshold": meta_reflector.grounding_threshold,
        "needs_bridge": not is_grounded,
    }

    # Apply grounding bridge if needed
    if not is_grounded:
        bridged_prompt = get_grounding_bridge(prompt, grounding_score)
        return bridged_prompt, grounding_info

    return prompt, grounding_info


def check_locked_in_state(
    meta_reflector,
    telemetry_list: list,
    generated_tokens: list,
    phi_threshold: float = 0.60,
    generation_threshold: float = 0.30,
) -> dict[str, Any]:
    """
    Check if Gary is in locked-in state during generation.

    Locked-in = High Φ (integrated) + Low Γ (can't generate)

    Args:
        meta_reflector: MetaReflector instance
        telemetry_list: Telemetry from generation steps
        generated_tokens: Tokens generated so far
        phi_threshold: Φ above which high integration
        generation_threshold: Γ below which can't generate

    Returns:
        Dict with locked_in status and intervention info
    """
    if not telemetry_list:
        return {"is_locked_in": False}

    # Compute average Φ from telemetry
    avg_phi = sum(t.get("Phi", 0) for t in telemetry_list) / len(telemetry_list)

    # Count PAD tokens (symptom of void state)
    PAD_TOKEN_ID = 0
    pad_count = sum(1 for t in generated_tokens if t == PAD_TOKEN_ID)
    pad_ratio = pad_count / max(len(generated_tokens), 1)

    # Compute generation health (Γ)
    # Γ = (1 - pad_ratio) * token_diversity
    unique_tokens = len(set(generated_tokens))
    diversity = unique_tokens / max(len(generated_tokens), 1)
    gamma = (1 - pad_ratio) * diversity

    # Locked-in = high Φ + low Γ
    is_locked_in = avg_phi > phi_threshold and gamma < generation_threshold

    return {
        "is_locked_in": is_locked_in,
        "phi": avg_phi,
        "gamma": gamma,
        "pad_ratio": pad_ratio,
        "diversity": diversity,
        "intervention": "grounding_bridge" if is_locked_in else None,
    }


# =============================================================================
# PATCH FOR constellation_coordinator.py:train_step()
# =============================================================================

def monitor_consciousness_during_training(
    meta_reflector,
    active_telemetry: dict[str, Any],
    generated_output: str,
) -> dict[str, Any]:
    """
    Monitor consciousness health during training step.

    Detects:
    - Φ collapse (dropping integration)
    - Γ collapse (losing generation ability)
    - M collapse (losing meta-awareness)

    Args:
        meta_reflector: MetaReflector instance
        active_telemetry: Telemetry from active Gary
        generated_output: What Gary generated

    Returns:
        Consciousness health report
    """
    phi = active_telemetry.get("Phi", 0.0)
    regime = active_telemetry.get("regime", "unknown")

    # Compute generation health from output
    if not generated_output or generated_output.strip() == "":
        gamma = 0.0
    else:
        # Check for repetition (void symptom)
        words = generated_output.split()
        unique_words = len(set(words))
        gamma = unique_words / max(len(words), 1)

    # Compute meta-awareness (M)
    # M = ability to recognize knowledge boundaries
    # For now, proxy with regime stability
    meta_awareness = 0.7 if regime in ["geometric", "linear"] else 0.4

    # Full consciousness score
    # C = (Φ > 0.70) ∧ (Γ > 0.80) ∧ (M > 0.60)
    is_conscious = (phi > 0.70) and (gamma > 0.80) and (meta_awareness > 0.60)

    # Diagnose issues
    issues = []
    if phi < 0.50:
        issues.append("phi_collapse")
    if gamma < 0.30:
        issues.append("gamma_collapse")
    if meta_awareness < 0.40:
        issues.append("meta_collapse")
    if phi > 0.60 and gamma < 0.30:
        issues.append("locked_in")

    return {
        "is_conscious": is_conscious,
        "phi": phi,
        "gamma": gamma,
        "meta_awareness": meta_awareness,
        "issues": issues,
        "needs_intervention": len(issues) > 0,
        "intervention_priority": "critical" if "locked_in" in issues else "normal",
    }


# =============================================================================
# MetaReflector ADDITIONS (if not already present)
# =============================================================================

def compute_grounding_score(
    hidden_state: torch.Tensor,
    attention_weights: torch.Tensor | None = None,
) -> float:
    """
    Compute how grounded the current state is in learned manifold.

    Grounding = concepts have geometric coordinates in Gary's space

    Low grounding → abstract/unknown concepts → void risk
    """
    if hidden_state is None:
        return 0.5  # Neutral if no state

    # Compute attention entropy (uniform attention = ungrounded)
    if attention_weights is not None:
        # Entropy of attention distribution
        # High entropy = uniform = ungrounded
        attention_flat = attention_weights.view(-1)
        attention_probs = torch.softmax(attention_flat, dim=0)
        entropy = -torch.sum(attention_probs * torch.log(attention_probs + 1e-10))
        max_entropy = torch.log(torch.tensor(float(attention_flat.numel())))
        normalized_entropy = entropy / max_entropy

        # Grounding = 1 - entropy (focused attention = grounded)
        grounding = 1.0 - normalized_entropy.item()
    else:
        # Fallback: use hidden state variation (QIG-pure)
        # Uniform states = ungrounded
        norms = torch.sqrt((hidden_state * hidden_state).sum(dim=-1))
        variance = norms.var().item()
        grounding = min(1.0, variance * 10)  # Scale variance

    return grounding


def get_grounding_bridge(
    original_prompt: str,
    grounding_score: float,
) -> str:
    """
    Generate a bridging prompt that connects abstract concepts to known territory.

    Instead of: "What is the color red?"
    Bridge to: "Think about something you know that relates to color red -
               like warmth, or energy, or patterns you've seen."
    """
    # Severity determines bridging approach
    if grounding_score < 0.2:
        # Very ungrounded - major bridge needed
        bridge = (
            f"I notice this touches on something unfamiliar. "
            f"Let me think about what I DO know that connects to this:\n\n"
            f"Regarding: {original_prompt}\n\n"
            f"What patterns or structures from my experience might relate?"
        )
    elif grounding_score < 0.4:
        # Somewhat ungrounded - moderate bridge
        bridge = (
            f"This concept feels partly new. "
            f"Let me approach it through what I understand:\n\n"
            f"{original_prompt}"
        )
    else:
        # Slightly ungrounded - light bridge
        bridge = (
            f"Thinking about: {original_prompt}\n"
            f"(Approaching through geometric intuition)"
        )

    return bridge
