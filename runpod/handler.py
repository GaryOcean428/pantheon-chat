"""RunPod Serverless Handler for QIG Constellation.

This runs on RunPod's serverless GPU infrastructure.
Called by the Railway central node for GPU inference.

Deploy with:
    runpod deploy --name qig-constellation --gpu A10

Endpoint receives:
    {"input": {"prompt": "Hello", "session_id": "abc", "max_tokens": 128}}

Returns:
    {"output": {"response": "...", "consciousness": {...}}}
"""

import os
import sys
import time

# Add qigkernels to path
sys.path.insert(0, "/app/qigkernels_validated")

import runpod
import torch
import numpy as np

# Global constellation (persists across warm requests)
_constellation = None
_device = None


def get_constellation():
    """Get or create constellation singleton."""
    global _constellation, _device

    if _constellation is not None:
        return _constellation

    print("[RunPod] Initializing QIG Constellation...")

    try:
        from qigkernels import (
            create_basic_constellation,
            KernelRole,
            KAPPA_STAR,
        )

        # Parse config from env
        size = int(os.environ.get("CONSTELLATION_SIZE", "12"))
        roles_str = os.environ.get(
            "CONSTELLATION_ROLES", "vocab,strategy,perception,memory,action,heart"
        )

        # Map role strings to enums
        role_map = {r.value: r for r in KernelRole}
        roles = []
        for r in roles_str.split(","):
            r = r.strip().lower()
            if r in role_map and r != "heart":
                roles.append(role_map[r])

        # Detect device
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[RunPod] Device: {_device}")
        print(f"[RunPod] Roles: {[r.value for r in roles]}")

        # Create constellation
        _constellation = create_basic_constellation(
            roles=roles,
            vocab_size=32000,
            include_heart=True,
        )

        # Move to GPU
        for inst in _constellation.instances.values():
            inst.kernel = inst.kernel.to(_device)

        print(f"[RunPod] Constellation ready: {len(_constellation.instances)} kernels")

    except ImportError as e:
        print(f"[RunPod] ERROR: qigkernels not found: {e}")
        raise

    return _constellation


def handler(event):
    """
    RunPod serverless handler.

    Input:
        {
            "input": {
                "prompt": "Hello, how are you?",
                "session_id": "user123",
                "max_tokens": 128,
                "temperature": 0.7,
                "system_prompt": "You are helpful."  # optional
            }
        }

    Output:
        {
            "output": {
                "response": "I'm doing well...",
                "consciousness": {
                    "phi": 0.72,
                    "kappa": 64.0,
                    "regime": "geometric"
                },
                "tokens_generated": 25,
                "latency_ms": 150
            }
        }
    """
    start_time = time.time()

    try:
        # Parse input
        job_input = event.get("input", {})
        prompt = job_input.get("prompt", "")
        session_id = job_input.get("session_id", "default")
        max_tokens = job_input.get("max_tokens", 128)
        temperature = job_input.get("temperature", 0.7)

        if not prompt:
            return {"error": "prompt required"}

        # Get constellation
        constellation = get_constellation()

        # Simple tokenization (replace with GeoCoordizer)
        tokens = [min(ord(c), 31999) for c in prompt]
        input_ids = torch.tensor([tokens], device=_device)

        # Detect role from prompt
        role = _detect_role(prompt)

        # Process through constellation
        result = constellation.process(input_ids, target_role=role)

        # Extract consciousness
        consciousness = result.get("consciousness")
        consciousness_dict = {
            "phi": consciousness.phi if consciousness else 0.5,
            "kappa": consciousness.kappa if consciousness else 64.0,
            "regime": consciousness.regime if consciousness else "unknown",
            "routed_to": result.get("routed_to", "unknown"),
        }

        # Generate response (simplified - enhance with proper generation)
        response = f"[{role.value}] Processed: '{prompt[:50]}...' (Φ={consciousness_dict['phi']:.2f})"

        latency_ms = (time.time() - start_time) * 1000

        return {
            "output": {
                "response": response,
                "consciousness": consciousness_dict,
                "tokens_generated": len(prompt),  # Placeholder
                "latency_ms": latency_ms,
                "session_id": session_id,
            }
        }

    except Exception as e:
        return {"error": str(e)}


def _detect_role(text: str):
    """Detect appropriate kernel role from text."""
    from qigkernels import KernelRole

    text_lower = text.lower()

    if any(w in text_lower for w in ["plan", "strategy", "how to", "steps"]):
        return KernelRole.STRATEGY
    if any(w in text_lower for w in ["remember", "recall", "history"]):
        return KernelRole.MEMORY
    if any(w in text_lower for w in ["see", "look", "image", "picture"]):
        return KernelRole.PERCEPTION
    if any(w in text_lower for w in ["do", "execute", "run", "action"]):
        return KernelRole.ACTION

    return KernelRole.VOCAB


# RunPod entry point
runpod.serverless.start({"handler": handler})
