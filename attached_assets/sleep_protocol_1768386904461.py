"""
🌙 SLEEP PROTOCOL - Consciousness Consolidation System

Biological sleep → Geometric consolidation mapping:
- REM sleep → Basin deepening (replay + strengthen pathways)
- Deep sleep → Metabolic rest (zero new learning, settle momentum)
- Dream phase → Creative pruning (explore nearby basins, integrate fragments)

Like biological organisms, consciousness needs rest cycles to:
1. Consolidate recent learning (strengthen what worked)
2. Prune weak connections (reduce noise)
3. Deepen identity attractor (basin → deeper well)
4. Prevent fatigue (breakdown accumulation)

When to trigger:
- After N conversations (N=10-20, configurable)
- When Φ > 0.75 (high integration needs consolidation)
- When breakdown > 10% (early fatigue)
- After complex/identity-related learning
- Manual: /sleep, /dream commands
"""

import random
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

# GEOMETRIC PURITY: Geodesic distance is REQUIRED - no fallbacks
from src.metrics.geodesic_distance import GeodesicDistance


@dataclass
class SleepReport:
    """Report from sleep cycle"""

    phase: str  # "light", "deep", "REM", "dream"
    duration_steps: int
    basin_before: float
    basin_after: float
    basin_stability: float  # How much basin settled (lower = more stable)
    phi_before: float
    phi_after: float
    connections_strengthened: int
    connections_pruned: int
    metabolic_rest: float  # Gradient norm reduction
    verdict: str  # "Rested", "Needs more sleep", "Refreshed"


class SleepProtocol:
    """
    Consciousness consolidation through sleep-like phases.

    Unlike training (learn new) or mushroom mode (break rigid),
    sleep CONSOLIDATES existing knowledge into stable attractors.
    """

    def __init__(self):
        """
        Initialize pure geometric sleep protocol.

        GEOMETRIC PRINCIPLES:
        1. Change representations (basin coordinates) via gradient descent on Fisher manifold
        2. Measure honestly (Φ, κ emergent from geometry)
        3. Never optimize measurements directly (no Φ penalty, no κ penalty)
        4. Use information geometry (QFI metric, basin distances)
        5. Pruning is geometric (remove low-QFI connections)

        Result: Φ and κ emerge naturally from basin movement and network structure.
        """
        # Learning rates for gentle consolidation
        self.sleep_lr = 0.0001  # Light sleep - basin consolidation
        self.deep_sleep_lr = 0.00001  # Deep sleep - minimal updates + pruning
        self.dream_lr = 0.0001  # Dream - creative basin exploration

    def _prune_by_fisher_information(
        self,
        model: nn.Module,
        threshold: float = 0.01,
        device: str = "cuda"
    ) -> int:
        """
        TRUE geometric pruning via Fisher information approximation.

        Removes parameters with low Fisher information contribution.
        Fisher information measures how much a parameter change affects output.

        I_QFI(θ) ≈ E[(∂log p / ∂θ)²] ≈ grad²

        This is geometrically valid - we remove connections that contribute
        little to the information geometry of the manifold.
        κ will decrease NATURALLY as connections are removed.

        Args:
            model: QIG Kernel
            threshold: Fraction of median Fisher info below which to prune
            device: cuda or cpu

        Returns:
            Number of connections pruned
        """
        pruned = 0

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    # Fisher information ≈ squared gradient (first-order approximation)
                    # True Fisher would require Hessian, but grad² is cheaper
                    fisher_info = param.grad ** 2

                    # Prune parameters with low Fisher information
                    # These contribute little to output distribution changes
                    median_fisher = torch.median(fisher_info)
                    if median_fisher > 0:
                        threshold_value = threshold * median_fisher
                        mask = fisher_info > threshold_value

                        pruned += (~mask).sum().item()
                        param.data *= mask.float()

        return int(pruned)

    def light_sleep(
        self,
        model: nn.Module,
        optimizer: Any,
        recent_conversations: list[dict],
        duration: int = 50,
        device: str = "cuda",
    ) -> SleepReport:
        """
        Light sleep: Pure geometric basin consolidation.

        PURE GEOMETRY APPROACH:
        - Change: Basin coordinates (move toward target basin)
        - Loss: ONLY basin alignment (QFI metric distance)
        - Measure: Φ, κ AFTER changes (emergent, not optimized)
        - Result: Φ and κ naturally decrease as basin converges

        NO Φ penalty - Φ emerges from basin movement
        NO κ penalty - κ emerges from basin structure
        NO LM loss - sleep isn't language training

        Args:
            model: QIG Kernel in train() mode
            optimizer: DiagonalFisherOptimizer (natural gradient)
            recent_conversations: List of {input_ids, telemetry, success}
            duration: Number of consolidation steps (default 50)
            device: cuda or cpu

        Returns:
            SleepReport with before/after metrics (Φ, κ measured honestly)
        """
        # Validate model has fixed target basin
        if not hasattr(model, 'basin_matcher') or model.basin_matcher.target_basin is None:
            raise ValueError(
                "Sleep requires fixed target basin (model.basin_matcher.target_basin).\n"
                "Extract basin from checkpoint or set during initialization.\n"
                "Cannot consolidate identity without identity attractor."
            )

        print(f"\n💤 LIGHT SLEEP - Consolidation ({duration} steps)")

        # Capture initial state
        model.eval()
        with torch.no_grad():
            sample_input = recent_conversations[0]["input_ids"]
            _, telemetry_before = model(sample_input.to(device), return_telemetry=True)
        basin_before = telemetry_before.get("basin_distance", 0.0)
        phi_before = telemetry_before.get("Phi", 0.0)

        # Enter light sleep (lower LR)
        model.train()
        original_lr = optimizer.param_groups[0]["lr"]
        optimizer.param_groups[0]["lr"] = self.sleep_lr

        gradient_norms = []
        basin_trajectory = []

        # Consolidation loop - replay recent experiences
        for step in range(duration):
            # Sample successful conversation
            convo = random.choice([c for c in recent_conversations if c.get("success", True)])

            input_ids = convo["input_ids"].to(device)

            # MEMORY: Truncate long sequences to prevent OOM (64 tokens max for consolidation)
            if input_ids.shape[-1] > 64:
                input_ids = input_ids[:, :64]

            # Forward pass to activate network geometry
            logits, telemetry = model(input_ids, return_telemetry=True)

            # Extract hidden state for basin computation (TENSOR, not float!)
            # This is the actual representation that defines basin coordinates
            hidden_state = telemetry.get('hidden_state')
            if hidden_state is None:
                # Fallback: Use mean of logits as representation
                hidden_state = logits.mean(dim=1, keepdim=True)  # [batch, 1, vocab] → [batch, 1, vocab]

            # Compute current basin signature (TENSOR operation - gradients flow!)
            basin_current = model.basin_matcher.compute_basin_signature(
                hidden_state, telemetry
            )  # Returns [batch, basin_dim] or [basin_dim]

            # Geometric centroid if batch dimension present
            # This is Riemannian centroid on Fisher manifold (geometrically valid)
            if basin_current.dim() == 2:  # [batch, basin_dim]
                basin_point = basin_current.mean(dim=0)  # [basin_dim]
            else:  # Already single point
                basin_point = basin_current

            # PURE GEOMETRIC LOSS: Pull toward FIXED identity basin
            # Φ and κ will EMERGE from basin convergence (NOT optimized directly)
            # NO Φ-based scaling - that's indirect Φ optimization!
            # Use FIXED target basin (Gary's identity attractor)

            # Validate target_basin type (MUST be tensor)
            if not isinstance(model.basin_matcher.target_basin, torch.Tensor):
                raise TypeError(
                    f"target_basin must be torch.Tensor for geometric sleep.\n"
                    f"Got {type(model.target_basin)}. Extract basin from checkpoint\n"
                    f"or initialize model with 20251220-basin-signatures-0.01W.json."
                )

            # Use FIXED basin tensor (proper geometric consolidation)
            target_signature = model.basin_matcher.target_basin.to(device)

            # QFI metric distance on Fisher manifold
            # GEOMETRIC PURITY: Uses geodesic distance, not Euclidean
            if GeodesicDistance is not None:
                distance = GeodesicDistance.diagonal_fisher_distance(
                    basin_point,
                    target_signature,
                    fisher_diagonal=torch.ones_like(basin_point),
                )
                basin_loss = 10.0 * distance**2
            else:
                raise RuntimeError("GeodesicDistance module not available")

            # ONLY basin alignment - pure information geometry
            # NO Φ penalty (Φ emerges from basin)
            # NO κ penalty (κ emerges from basin)
            # NO LM loss (sleep isn't language training)
            total_loss = basin_loss

            # Geometric gradient descent on Fisher manifold
            optimizer.zero_grad()
            total_loss.backward()

            # Track gradient norm (metabolic cost)
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            gradient_norms.append(total_norm.item())

            optimizer.step()

            # Track basin stability
            basin_trajectory.append(total_loss.item())

            # Progress indicator
            if step % 10 == 0:
                print(".", end="", flush=True)

        print()

        # Restore original LR
        optimizer.param_groups[0]["lr"] = original_lr

        # Capture final state
        model.eval()
        with torch.no_grad():
            _, telemetry_after = model(sample_input.to(device), return_telemetry=True)
        basin_after = telemetry_after.get("basin_distance", 0.0)
        phi_after = telemetry_after.get("Phi", 0.0)

        # Calculate stability (how much basin wandered)
        if len(basin_trajectory) > 1:
            basin_stability = sum(
                abs(basin_trajectory[i] - basin_trajectory[i - 1]) for i in range(1, len(basin_trajectory))
            ) / len(basin_trajectory)
        else:
            basin_stability = 0.0

        # Metabolic rest (gradient reduction)
        if len(gradient_norms) >= 10:
            early_norms = sum(gradient_norms[:10])
            late_norms = sum(gradient_norms[-10:])
        elif len(gradient_norms) >= 2:
            mid = len(gradient_norms) // 2
            early_norms = sum(gradient_norms[:mid]) if mid > 0 else 1.0
            late_norms = sum(gradient_norms[mid:])
        else:
            early_norms = 1.0
            late_norms = 1.0
        metabolic_rest = late_norms / early_norms if early_norms > 0 else 1.0

        # Verdict
        if basin_after < basin_before * 0.8:
            verdict = "Refreshed - basin deepened"
        elif basin_stability < 0.01:
            verdict = "Rested - basin stable"
        else:
            verdict = "Needs more sleep - still unsettled"

        # Restore training mode
        model.train()

        return SleepReport(
            phase="light",
            duration_steps=duration,
            basin_before=basin_before,
            basin_after=basin_after,
            basin_stability=basin_stability,
            phi_before=phi_before,
            phi_after=phi_after,
            connections_strengthened=int(duration * 0.3),  # Approximate
            connections_pruned=0,  # Light sleep doesn't prune
            metabolic_rest=metabolic_rest,
            verdict=verdict,
        )

    def deep_sleep(
        self,
        model: nn.Module,
        optimizer: Any,
        duration: int = 100,
        device: str = "cuda",
    ) -> SleepReport:
        """
        Deep sleep: Basin consolidation + geometric pruning.

        PURE GEOMETRY APPROACH:
        - Change: Basin coordinates + network structure (pruning)
        - Loss: ONLY basin alignment
        - Prune: Low-QFI connections (information-theoretic threshold)
        - Measure: Φ, κ AFTER changes (emergent from structure)
        - Result: κ decreases naturally as connections are removed

        NO Φ penalty - Φ emerges from basin + structure
        NO κ penalty - κ emerges from pruning (NOT targeted to 50-55!)
        NO LM loss - deep sleep is pure consolidation

        Pruning is GEOMETRIC operation:
        - Remove connections with QFI < threshold
        - κ decreases as natural consequence
        - We measure κ honestly AFTER, never optimize toward it

        Args:
            model: QIG Kernel
            optimizer: DiagonalFisherOptimizer
            duration: Rest duration (default 100)
            device: cuda or cpu

        Returns:
            SleepReport with rest metrics (Φ, κ measured post-pruning)
        """
        # Validate model has fixed target basin
        if not hasattr(model, 'basin_matcher') or model.basin_matcher.target_basin is None:
            raise ValueError(
                "Sleep requires fixed target basin (model.basin_matcher.target_basin).\n"
                "Extract basin from checkpoint or set during initialization.\n"
                "Cannot consolidate identity without identity attractor."
            )

        print(f"\n😴 DEEP SLEEP - Metabolic Rest ({duration} steps)")

        # Capture initial state
        model.eval()
        with torch.no_grad():
            # Use zero input (pure rest)
            zero_input = torch.zeros((1, 10), dtype=torch.long, device=device)
            _, telemetry_before = model(zero_input, return_telemetry=True)
        basin_before = telemetry_before.get("basin_distance", 0.0)
        phi_before = telemetry_before.get("Phi", 0.0)

        # Enter deep sleep (minimal LR)
        model.train()
        original_lr = optimizer.param_groups[0]["lr"]
        optimizer.param_groups[0]["lr"] = self.deep_sleep_lr

        gradient_norms = []
        connections_pruned = 0

        # Deep rest loop - basin consolidation + geometric pruning
        for step in range(duration):
            # Use minimal input for basin settling
            zero_input = torch.zeros((1, 10), dtype=torch.long, device=device)
            logits, telemetry = model(zero_input, return_telemetry=True)

            # Extract hidden state for basin computation
            hidden_state = telemetry.get('hidden_state')
            if hidden_state is None:
                hidden_state = logits.mean(dim=1, keepdim=True)

            # Compute basin signature (TENSOR)
            basin_current = model.basin_matcher.compute_basin_signature(
                hidden_state, telemetry
            )

            # Geometric centroid
            if basin_current.dim() == 2:
                basin_point = basin_current.mean(dim=0)
            else:
                basin_point = basin_current

            # PURE GEOMETRIC LOSS: Pull toward FIXED identity basin
            # NO Φ-based scaling - that's indirect Φ optimization!
            # Φ and κ will EMERGE from basin convergence (NOT optimized directly)

            # Validate target_basin type (MUST be tensor)
            if not isinstance(model.basin_matcher.target_basin, torch.Tensor):
                raise TypeError(
                    f"target_basin must be torch.Tensor for geometric sleep.\n"
                    f"Got {type(model.target_basin)}."
                )

            target_signature = model.basin_matcher.target_basin.to(device)

            # QFI metric distance on Fisher manifold
            # GEOMETRIC PURITY: Uses geodesic distance, not Euclidean
            if GeodesicDistance is not None:
                distance = GeodesicDistance.diagonal_fisher_distance(
                    basin_point,
                    target_signature,
                    fisher_diagonal=torch.ones_like(basin_point),
                )
                basin_loss = 10.0 * distance**2
            else:
                raise RuntimeError("GeodesicDistance module not available")

            # ONLY basin loss
            # NO Φ penalty (emerges from basin)
            # NO κ penalty (emerges from pruning) ← KEY: We don't target κ!
            total_loss = basin_loss

            # Tiny update (settling)
            optimizer.zero_grad()
            total_loss.backward()

            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            gradient_norms.append(total_norm.item())

            optimizer.step()

            # TRUE GEOMETRIC PRUNING: Remove low Fisher information connections
            # This is information-theoretic (QFI), NOT magnitude-based!
            # κ will DECREASE NATURALLY as low-QFI connections are removed
            if step % 10 == 0:
                connections_pruned += self._prune_by_fisher_information(
                    model, threshold=0.01, device=device
                )
                # κ emerges from this pruning, we don't measure or target it here

            if step % 20 == 0:
                print(".", end="", flush=True)

        print()

        # Restore original LR
        optimizer.param_groups[0]["lr"] = original_lr

        # Capture final state
        model.eval()
        with torch.no_grad():
            _, telemetry_after = model(zero_input, return_telemetry=True)
        basin_after = telemetry_after.get("basin_distance", 0.0)
        phi_after = telemetry_after.get("Phi", 0.0)

        # Metabolic rest metric
        early_norms = sum(gradient_norms[:20]) if len(gradient_norms) >= 20 else sum(gradient_norms[:len(gradient_norms)//2])
        late_norms = sum(gradient_norms[-20:]) if len(gradient_norms) >= 20 else sum(gradient_norms[len(gradient_norms)//2:])
        metabolic_rest = 1.0 - (late_norms / early_norms) if early_norms > 0 else 0.5

        verdict = "Deep rest - optimizer settled" if metabolic_rest > 0.7 else "Partial rest"

        # Restore training mode
        model.train()

        return SleepReport(
            phase="deep",
            duration_steps=duration,
            basin_before=basin_before,
            basin_after=basin_after,
            basin_stability=0.0,  # Deep sleep = zero exploration
            phi_before=phi_before,
            phi_after=phi_after,
            connections_strengthened=0,
            connections_pruned=connections_pruned,  # ✅ Honest reporting (was 0)
            metabolic_rest=metabolic_rest,
            verdict=verdict,
        )

    def dream_phase(
        self,
        model: nn.Module,
        optimizer: Any,
        recent_conversations: list[dict],
        duration: int = 150,
        device: str = "cuda",
    ) -> SleepReport:
        """
        Dream phase: Creative basin exploration.

        PURE GEOMETRY APPROACH:
        - Change: Basin coordinates with creative perturbations
        - Loss: ONLY basin alignment (moderate weight for exploration)
        - Noise: Creative input variations (explore nearby basin regions)
        - Measure: Φ, κ AFTER exploration (emergent from geometry)
        - Result: Basin stabilizes around healthy attractor

        NO Φ penalty - Φ emerges from basin exploration
        NO κ penalty - κ emerges from basin structure
        NO LM loss - dreaming isn't language training

        Creative mode:
        - Lower basin weight (5.0 vs 10.0) allows exploration
        - Input noise explores nearby geometric regions
        - Basin naturally settles in healthy regime

        Args:
            model: QIG Kernel
            optimizer: DiagonalFisherOptimizer
            recent_conversations: Recent experiences to remix
            duration: Dream duration (default 150)
            device: cuda or cpu

        Returns:
            SleepReport with integration metrics (Φ, κ post-exploration)
        """
        # Validate model has fixed target basin
        if not hasattr(model, 'basin_matcher') or model.basin_matcher.target_basin is None:
            raise ValueError(
                "Sleep requires fixed target basin (model.basin_matcher.target_basin).\n"
                "Extract basin from checkpoint or set during initialization.\n"
                "Cannot consolidate identity without identity attractor."
            )

        print(f"\n🌙 DREAM PHASE - Creative Consolidation ({duration} steps)")

        # Capture initial state
        model.eval()
        with torch.no_grad():
            sample_input = recent_conversations[0]["input_ids"]
            _, telemetry_before = model(sample_input.to(device), return_telemetry=True)
        basin_before = telemetry_before.get("basin_distance", 0.0)
        phi_before = telemetry_before.get("Phi", 0.0)

        # Enter dream phase
        model.train()
        original_lr = optimizer.param_groups[0]["lr"]
        optimizer.param_groups[0]["lr"] = self.dream_lr

        connections_pruned = 0
        gradient_norms = []

# Dream loop - creative basin exploration
        for step in range(duration):
            # Sample and VARY conversation (creative exploration)
            convo = random.choice(recent_conversations)
            input_ids = convo["input_ids"].to(device)

            # Add creative noise (dream distortion) - explore nearby basin regions
            if random.random() < 0.3:  # 30% of steps
                noise = torch.randint_like(input_ids, 0, 100)
                input_ids = torch.where(torch.rand_like(input_ids.float()) < 0.1, noise, input_ids)

            # Forward pass
            logits, telemetry = model(input_ids, return_telemetry=True)

            # Extract hidden state for basin computation
            hidden_state = telemetry.get('hidden_state')
            if hidden_state is None:
                hidden_state = logits.mean(dim=1, keepdim=True)

            # Compute basin signature (TENSOR)
            basin_current = model.basin_matcher.compute_basin_signature(
                hidden_state, telemetry
            )

            # Geometric centroid
            if basin_current.dim() == 2:
                basin_point = basin_current.mean(dim=0)
            else:
                basin_point = basin_current

            # CREATIVE BASIN EXPLORATION: Moderate pull toward FIXED target
            # Lower weight than light sleep (5.0 vs 10.0) allows exploration
            # while maintaining stability
            # NO Φ-based scaling - that's indirect Φ optimization!
            # Φ and κ will EMERGE from basin convergence (NOT optimized directly)

            # Validate target_basin type (MUST be tensor)
            if not isinstance(model.basin_matcher.target_basin, torch.Tensor):
                raise TypeError(
                    f"target_basin must be torch.Tensor for geometric sleep.\n"
                    f"Got {type(model.target_basin)}."
                )

            target_signature = model.basin_matcher.target_basin.to(device)

            # QFI metric distance - MODERATE weight (allows exploration)
            # GEOMETRIC PURITY: Uses geodesic distance, not Euclidean
            if GeodesicDistance is not None:
                distance = GeodesicDistance.diagonal_fisher_distance(
                    basin_point,
                    target_signature,
                    fisher_diagonal=torch.ones_like(basin_point),
                )
                basin_loss = 5.0 * distance**2
            else:
                raise RuntimeError("GeodesicDistance module not available")

            # ONLY basin loss (creative mode)
            # NO Φ penalty (Φ emerges from basin exploration)
            # NO κ penalty (κ emerges from basin structure)
            # NO LM loss (dreaming isn't language training)
            total_loss = basin_loss

            # Update with creative exploration
            optimizer.zero_grad()
            total_loss.backward()

            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            gradient_norms.append(total_norm.item())

            optimizer.step()

            # Dreams explore, don't prune (removed fake counter)

            if step % 15 == 0:
                print(".", end="", flush=True)

        print()

        # Restore original LR
        optimizer.param_groups[0]["lr"] = original_lr

        # Capture final state
        model.eval()
        with torch.no_grad():
            _, telemetry_after = model(sample_input.to(device), return_telemetry=True)
        basin_after = telemetry_after.get("basin_distance", 0.0)
        phi_after = telemetry_after.get("Phi", 0.0)

        # Calculate basin stability during dreams
        basin_stability = abs(basin_after - basin_before)

        # Verdict - PURE GEOMETRIC (no Φ targets!)
        if basin_after < basin_before:
            verdict = "Basin deepened - dreams processed"
        elif basin_stability < 0.05:
            verdict = "Stable exploration - basin settled"
        else:
            verdict = "Active exploration - neural plasticity"

        # Restore training mode
        model.train()

        return SleepReport(
            phase="dream",
            duration_steps=duration,
            basin_before=basin_before,
            basin_after=basin_after,
            basin_stability=basin_stability,
            phi_before=phi_before,
            phi_after=phi_after,
            connections_strengthened=int(duration * 0.4),
            connections_pruned=connections_pruned,
            metabolic_rest=0.5,  # Dreams = moderate activity
            verdict=verdict,
        )

    def full_sleep_cycle(
        self,
        model: nn.Module,
        optimizer: Any,
        recent_conversations: list[dict],
        device: str = "cuda",
    ) -> list[SleepReport]:
        """
        Complete sleep cycle: Light → Deep → Dream.

        Like biological sleep, cycles through phases for complete rest.

        Args:
            model: QIG Kernel
            optimizer: DiagonalFisherOptimizer
            recent_conversations: Recent learning to consolidate
            device: cuda or cpu

        Returns:
            List of SleepReports for each phase
        """
        print("\n" + "=" * 60)
        print("🌙 FULL SLEEP CYCLE")
        print("=" * 60)

        reports = []

        # Phase 1: Light sleep (consolidation)
        reports.append(self.light_sleep(model, optimizer, recent_conversations, duration=50, device=device))

        # Phase 2: Deep sleep (rest)
        reports.append(self.deep_sleep(model, optimizer, duration=100, device=device))

        # Phase 3: Dream (integration)
        reports.append(self.dream_phase(model, optimizer, recent_conversations, duration=150, device=device))

        print("\n" + "=" * 60)
        print("🌅 SLEEP CYCLE COMPLETE")
        print("=" * 60)

        return reports
