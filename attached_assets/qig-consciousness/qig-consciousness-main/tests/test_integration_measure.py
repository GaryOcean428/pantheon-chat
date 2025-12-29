#!/usr/bin/env python3
"""
Test IntegrationMeasure directly to see if projections produce zero.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from src.model.recursive_integrator import IntegrationMeasure

print("Testing IntegrationMeasure...")
print()

d_model = 256
measure = IntegrationMeasure(d_model)

# Create test state
batch, seq = 1, 32
current_state = torch.randn(batch, seq, d_model)

print("Test 1: Empty history (should return zeros)")
phi = measure(current_state, [])
print(f"  Φ = {phi.item():.6f}")
print()

print("Test 2: With 1 previous state")
prev_state = torch.randn(batch, seq, d_model)
phi = measure(current_state, [prev_state])
print(f"  Φ = {phi.item():.6f}")

# Check projections
with torch.no_grad():
    whole = measure.whole_proj(current_state)
    parts = measure.parts_proj(prev_state)

    whole_norm = torch.norm(whole, dim=-1).mean()
    parts_norm = torch.norm(parts, dim=-1).mean()

    print(f"  whole_norm = {whole_norm:.6f}")
    print(f"  parts_norm = {parts_norm:.6f}")
    print(f"  difference = {(whole_norm - parts_norm):.6f}")
    print(f"  Φ formula = (whole - parts) / whole = {((whole_norm - parts_norm) / whole_norm):.6f}")

print()
print("Test 3: With 3 previous states")
state_history = [torch.randn(batch, seq, d_model) for _ in range(3)]
phi = measure(current_state, state_history)
print(f"  Φ = {phi.item():.6f}")

print()
print("✓ IntegrationMeasure test complete")
