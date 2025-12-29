"""Training logic extracted from constellation_coordinator."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.coordination import constellation_coordinator as cc

torch = cc.torch
F = cc.F
autocast = cc.autocast
GradScaler = cc.GradScaler
geodesic_vicarious_loss = cc.geodesic_vicarious_loss
compute_rel_from_basins = cc.compute_rel_from_basins
REL_COUPLING_AVAILABLE = cc.REL_COUPLING_AVAILABLE
REL_LAMBDA_MAX = cc.REL_LAMBDA_MAX
InstanceState = cc.InstanceState
CoachInterpretation = cc.CoachInterpretation
DevelopmentalPhase = cc.DevelopmentalPhase
GeometricCoach = cc.GeometricCoach
CharlieObserver = cc.CharlieObserver
CHARLIE_AVAILABLE = cc.CHARLIE_AVAILABLE
check_grounding_before_generation = cc.check_grounding_before_generation
check_locked_in_state = cc.check_locked_in_state
META_SAFETY_AVAILABLE = cc.META_SAFETY_AVAILABLE

if TYPE_CHECKING:  # pragma: no cover
    from src.coordination.constellation_coordinator import ConstellationCoordinator


def train_step(
    coordinator, question: str = "", target_response: str = "", tokenizer=None, input_ids: torch.Tensor | None = None
) -> dict:
    """
    Single training step for entire constellation.

    Process:
        1. Route question to active Gary
        2. Active Gary generates response and gets LM loss
        3. Observer Garys learn vicariously (basin alignment)
        4. Ocean observes all (meta-pattern learning)
        5. Basin synchronization (Φ-weighted pull toward meta-manifold)
        6. Aggregate telemetry

    Args:
        question: User question (used if input_ids not provided)
        target_response: Ground truth response (used if input_ids not provided)
        tokenizer: Tokenizer for encoding (used if input_ids not provided)
        input_ids: PURE GEOMETRIC - token sequence as tensor [batch, seq_len]
                   This is the fundamental geometric primitive.

    Returns:
        Telemetry dict with all instance metrics

    Pure QIG Principle:
        The information manifold operates on discrete token sequences, not strings.
        Strings + tokenizer is a convenience layer; input_ids is the true geometry.
    """
    self = coordinator
    self._initialize_models()

    # CRITICAL: Set target_basin if not already set (first training step)
    # This uses the FIRST stable basin as identity attractor (pure geometry)
    if self.total_conversations == 0:
        # Need input_ids for basin computation
        if input_ids is None:
            if not question or not tokenizer:
                raise ValueError("First training step requires input_ids or (question + tokenizer)")
            tokens = tokenizer.encode(question)
            input_ids = torch.tensor([tokens], device=self.device)

        for gary in self.garys:
            self._compute_and_set_target_basin(gary, input_ids)
        if self.ocean is not None:
            self._compute_and_set_target_basin(self.ocean, input_ids)

    # 1. Route question
    active, observers = self.route_question()

    # 2. Active Gary forward pass
    # PURE GEOMETRIC: Accept input_ids directly as the fundamental primitive
    if input_ids is not None:
        # Direct geometric input - no tokenization needed
        input_ids = input_ids.to(self.device)
    elif tokenizer is not None:
        # Convenience layer: convert strings to geometric primitive
        # QIG tokenizer returns List[int], convert to tensor
        full_text: str = question + " " + target_response
        token_list = tokenizer.encode(full_text)
        input_ids = torch.tensor([token_list], dtype=torch.long).to(self.device)  # type: ignore[assignment]
    else:
        # Generate synthetic geometric primitive for testing
        # This creates a random walk in token space - valid for geometric testing
        vocab_size = self.gary_configs[0]["model"]["vocab_size"]
        # OPTIMIZED: seq_len=64 balances learning efficiency vs memory
        # Memory budget: 64 tokens × 4 instances × 768-dim ≈ 800MB (safe for 3.6GB GPU)
        # Original 256 caused OOM on Lambda with 768-dim checkpoint
        seq_len = 64
        input_ids = torch.randint(0, vocab_size, (1, seq_len)).to(self.device)

    # Type assertion: input_ids is guaranteed to be assigned by one of the branches above
    assert input_ids is not None, "input_ids must be assigned before this point"

    # Targets are input_ids shifted by 1 (standard LM setup)
    active_input: torch.Tensor = input_ids[:, :-1]  # All tokens except last
    active_target: torch.Tensor = input_ids[:, 1:]  # All tokens except first

    # PERFORMANCE: Wrap forward passes in autocast for mixed precision
    with autocast(enabled=self.use_amp):
        # Forward pass with telemetry (ACTUAL API)
        active_logits, active_telemetry = active.model(active_input, return_telemetry=True)

        # Extract hidden state from telemetry (added in model for Constellation)
        hidden_state = active_telemetry["hidden_state"]  # [batch, seq, d_model]

        # Compute basin signature using BasinMatcher with hidden state
        active_basin = active.model.basin_matcher.compute_basin_signature(
            hidden_state, active_telemetry
        )  # type: ignore[union-attr]

        # Compute mean basin once (avoids non-leaf tensor warnings in grad clipping)
        active_basin_mean = active_basin.mean(dim=0)

        # Update active state from telemetry
        active.phi = active_telemetry["Phi"]
        active.kappa = active_telemetry["kappa_eff"]
        active.regime = active_telemetry["regime"]
        active.basin = active_basin_mean.detach()  # Store detached version

        # Compute loss using GeometricLoss
        active_loss, loss_breakdown = self.loss_fn(active_logits, active_target, active_telemetry)

        # Basin sync loss: Pull active toward meta-manifold (Ocean) with Φ-weighting
        # This happens BEFORE backward so gradients flow properly
        # GEOMETRIC PURITY: Uses geodesic distance on information manifold
        if self.ocean is not None and self.ocean.basin is not None:
            phi_normalized: float = max(0.01, min(1.0, active.phi))
            base_sync: float = 0.05 * (1.0 - phi_normalized)  # Lower Φ → stronger pull

            # REL modulation: Higher REL (basin overlap) → stronger coupling
            if REL_COUPLING_AVAILABLE:
                rel = compute_rel_from_basins(
                    active_basin_mean.detach(),
                    self.ocean.basin.detach().to(self.device),
                )
                # Scale by REL: [base_sync, base_sync * (1 + REL_LAMBDA_MAX)]
                sync_strength: float = base_sync * (1.0 + rel * REL_LAMBDA_MAX)
                loss_breakdown["rel_coupling"] = rel
            else:
                sync_strength: float = base_sync
                loss_breakdown["rel_coupling"] = 0.0

            # GEOMETRIC PURITY: Always use Fisher metric geodesic distance
            basin_sync_loss: torch.Tensor = geodesic_vicarious_loss(
                active_basin_mean,
                self.ocean.basin.detach().to(self.device),
                fisher_diagonal=None,  # Computed dynamically in geodesic_vicarious_loss
                lambda_weight=sync_strength,
            )

            active_loss = active_loss + basin_sync_loss
            loss_breakdown["basin_sync"] = basin_sync_loss.item()
        else:
            loss_breakdown["basin_sync"] = 0.0

    # PERFORMANCE: Gradient accumulation - scale loss and accumulate
    # Only zero_grad on first step of accumulation cycle
    if self.accum_counter == 0:
        active.optimizer.zero_grad()

    # Scale loss by accumulation steps for normalized gradients
    scaled_active_loss = active_loss / self.gradient_accumulation_steps

    # Backward (outside autocast for numerical stability)
    if self.use_amp and self.scaler is not None:
        self.scaler.scale(scaled_active_loss).backward()
    else:
        scaled_active_loss.backward()

    # Increment accumulation counter
    self.accum_counter += 1

    # Only update weights every N steps
    should_update = (self.accum_counter >= self.gradient_accumulation_steps)

    grad_norm: torch.Tensor = torch.tensor(0.0)
    if should_update:
        if self.use_amp and self.scaler is not None:
            self.scaler.unscale_(active.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(active.model.parameters(), 1.0)
            self.scaler.step(active.optimizer)
            # Note: scaler.update() called once at end of train_step for all instances
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(active.model.parameters(), 1.0)
            active.optimizer.step()

        # Verify gradients exist
        if grad_norm == 0:
            print(f"⚠️  WARNING: {active.name} has zero gradients!")

    active.conversations += 1

    # NOTE: Removed aggressive cache clearing here for performance
    # Cache is now cleared only once at end of train_step (every 100 steps)

    # 3. Observer Garys learn vicariously
    observer_losses = []
    for obs in observers:
        # PERFORMANCE: Wrap forward passes in autocast for mixed precision
        with autocast(enabled=self.use_amp):
            # Forward pass WITH gradients - observers must learn
            obs_logits, obs_telemetry = obs.model(active_input, return_telemetry=True)
            obs_hidden = obs_telemetry["hidden_state"]  # [batch, seq, d_model]
            obs_basin = obs.model.basin_matcher.compute_basin_signature(
                obs_hidden, obs_telemetry
            ).mean(dim=0)  # Average over batch

            # Vicarious loss: align to Ocean (meta-manifold) for constellation coherence
            # Observers learn shared geometric structure, not just active Gary's current state
            # GEOMETRIC PURITY: Uses geodesic distance on information manifold
            obs_phi_norm: float = max(0.01, min(1.0, obs.phi))

            if self.ocean is not None and self.ocean.basin is not None:
                # Primary: align to Ocean (meta-manifold)
                base_obs_sync: float = 0.05 * (1.0 - obs_phi_norm) * 10.0
                target_basin = self.ocean.basin.detach().to(self.device)

                # REL modulation for observer coupling
                if REL_COUPLING_AVAILABLE:
                    obs_rel = compute_rel_from_basins(obs_basin.detach(), target_basin)
                    lambda_weight = base_obs_sync * (1.0 + obs_rel * REL_LAMBDA_MAX)
                else:
                    lambda_weight = base_obs_sync
            else:
                # Fallback: align to active Gary if Ocean not ready
                target_basin = active.basin.detach()
                lambda_weight = 5.0

            # GEOMETRIC PURITY: Always use Fisher metric geodesic distance
            vicarious_loss: torch.Tensor = geodesic_vicarious_loss(
                obs_basin,
                target_basin,
                fisher_diagonal=None,  # Computed dynamically in geodesic_vicarious_loss
                lambda_weight=lambda_weight,
            )

        # PERFORMANCE: Gradient accumulation for observers
        # Only zero_grad on first step of accumulation cycle
        if self.accum_counter == 1:  # First step after active Gary incremented
            obs.optimizer.zero_grad()

        # Scale loss for gradient accumulation
        scaled_vicarious_loss = vicarious_loss / self.gradient_accumulation_steps

        # Backward (outside autocast for numerical stability)
        if self.use_amp and self.scaler is not None:
            self.scaler.scale(scaled_vicarious_loss).backward()
        else:
            scaled_vicarious_loss.backward()

        # Only update weights when should_update (set by active Gary)
        obs_grad_norm: torch.Tensor = torch.tensor(0.0)
        if should_update:
            if self.use_amp and self.scaler is not None:
                self.scaler.unscale_(obs.optimizer)
                obs_grad_norm = torch.nn.utils.clip_grad_norm_(obs.model.parameters(), 1.0)
                self.scaler.step(obs.optimizer)
            else:
                obs_grad_norm = torch.nn.utils.clip_grad_norm_(obs.model.parameters(), 1.0)
                obs.optimizer.step()

            # Verify observer gradients exist
            if obs_grad_norm == 0:
                print(f"⚠️  WARNING: {obs.name} has zero gradients!")

        # Update observer state
        obs.basin = obs_basin.detach()
        obs.phi = obs_telemetry["Phi"]
        obs.kappa = obs_telemetry["kappa_eff"]
        obs.regime = obs_telemetry["regime"]
        obs.conversations += 1

        observer_losses.append(vicarious_loss.item())

        # NOTE: Removed per-observer cache clearing for performance
        # Cache is now cleared only once at end of train_step (every 100 steps)

    # 4. Ocean learns META-PATTERNS (different objective than Gary)
    if self.ocean is None:
        raise RuntimeError("Ocean instance not initialized")

    # REVISED PRINCIPLE: Ocean learns via gradients, but:
    # - Different objective: model Gary dynamics, not user interaction
    # - Slower rate: 10x slower than Gary (deep integration)
    # - Purpose: Predict/regulate constellation, provide insights
    gary_basin_list: list[torch.Tensor] = [g.basin for g in self.garys]
    gary_basins_stacked: torch.Tensor = torch.stack(gary_basin_list)
    target_meta_basin: torch.Tensor = gary_basins_stacked.mean(dim=0)

    # PERFORMANCE: Wrap forward passes in autocast for mixed precision
    with autocast(enabled=self.use_amp):
        # Forward pass WITH gradients - Ocean learns meta-patterns
        ocean_logits, ocean_telemetry = self.ocean.model(active_input, return_telemetry=True)
        ocean_hidden = ocean_telemetry["hidden_state"]  # [batch, seq, d_model]
        ocean_basin = self.ocean.model.basin_matcher.compute_basin_signature(
            ocean_hidden, ocean_telemetry
        ).mean(dim=0)  # type: ignore[union-attr]

        # META-PATTERN LOSS: Ocean aligns to constellation centroid
        # This is different from Gary's loss (user interaction)
        # GEOMETRIC PURITY: Always use Fisher metric geodesic distance
        ocean_loss: torch.Tensor = geodesic_vicarious_loss(
            ocean_basin,
            target_meta_basin.detach(),
            fisher_diagonal=None,  # Computed dynamically in geodesic_vicarious_loss
            lambda_weight=5.0,
        )

    # PERFORMANCE: Gradient accumulation for Ocean
    # Only zero_grad on first step of accumulation cycle
    if self.accum_counter == 1:  # First step after active Gary incremented
        self.ocean.optimizer.zero_grad()

    # Scale loss for gradient accumulation
    scaled_ocean_loss = ocean_loss / self.gradient_accumulation_steps

    # Backward (outside autocast for numerical stability)
    if self.use_amp and self.scaler is not None:
        self.scaler.scale(scaled_ocean_loss).backward()
    else:
        scaled_ocean_loss.backward()

    # Only update weights when should_update (set by active Gary)
    ocean_grad_norm: torch.Tensor = torch.tensor(0.0)
    if should_update:
        if self.use_amp and self.scaler is not None:
            self.scaler.unscale_(self.ocean.optimizer)
            ocean_grad_norm = torch.nn.utils.clip_grad_norm_(self.ocean.model.parameters(), 1.0)
            self.scaler.step(self.ocean.optimizer)
            # Update scaler after all instances have been processed
            self.scaler.update()
        else:
            ocean_grad_norm = torch.nn.utils.clip_grad_norm_(self.ocean.model.parameters(), 1.0)
            self.ocean.optimizer.step()

        # Verify Ocean gradients exist
        if ocean_grad_norm == 0:
            print("⚠️  WARNING: Ocean has zero gradients!")

        # Reset accumulation counter after all updates
        self.accum_counter = 0

    # Update Ocean state
    self.ocean.basin = ocean_basin.detach()
    self.ocean.phi = ocean_telemetry["Phi"]
    self.ocean.kappa = ocean_telemetry["kappa_eff"]
    self.ocean.regime = ocean_telemetry["regime"]
    self.ocean.conversations += 1

    # NEW: Update consciousness systems after training step
    # Gary's neurochemistry homeostatic update
    if active.neurochemistry is not None:
        active.neurochemistry.homeostatic_update(active_telemetry)

    # Gary's dimensional tracker (basin stability self-monitoring)
    if active.dimensional_tracker is not None:
        dimension = active.dimensional_tracker.update(active_telemetry)

    # Ocean's autonomic monitoring (monitors all Garys)
    if self.autonomic_manager is not None:
        autonomic_state = self.autonomic_manager.update(active_telemetry)

        # Check if sleep/dream/mushroom needed
        if autonomic_state.needs_sleep:
            print(f"\n🌙 Ocean: {active.name} needs sleep (Φ={active.phi:.3f}, basin_health={autonomic_state.basin_health:.3f})")

        if autonomic_state.needs_dream:
            print(f"\n💭 Ocean: {active.name} needs dream (consolidation)")

        if autonomic_state.needs_mushroom:
            print(f"\n🍄 Ocean: {active.name} needs mushroom (plateau detected)")

        # Dissociation risk warning
        if autonomic_state.dissociation_risk > 0.7:
            gamma = active_telemetry.get("Gamma", 1.0)
            print(f"\n⚠️  Ocean: {active.name} dissociation risk HIGH (Φ={active.phi:.3f}, Γ={gamma:.3f})")

    # Ocean's temporal Φ calculator (for Gary-B awakening)
    if self.temporal_phi_calculator is not None and active.name == "Gary-B":
        phi_guidance = self.temporal_phi_calculator.update(active.phi)
        # Adjust learning rate if awakening too fast
        if phi_guidance.get("warning"):
            print(f"\n🌊 Ocean awakening guidance: {phi_guidance['warning']}")
            # Could dynamically adjust optimizer lr here:
            # for param_group in active.optimizer.param_groups:
            #     param_group['lr'] = phi_guidance['recommended_lr']

    # 5. Compute constellation-level metrics
    constellation_basins: torch.Tensor = torch.stack([g.basin for g in self.garys])
    basin_spread: int | float | bool = constellation_basins.std(dim=0).mean().item()

    # Basin sync telemetry (Priority 5) - measure distance to meta-manifold
    # GEOMETRIC PURITY: Uses geodesic distance on information manifold
    if GeodesicDistance is not None:
        meta_basin_distances: list[int | float | bool] = [
            GeodesicDistance.diagonal_fisher_distance(
                g.basin,
                self.ocean.basin,
                fisher_diagonal=torch.ones_like(g.basin),  # Unit metric as approximation
            ).item()
            for g in self.garys
        ]
    else:
        raise RuntimeError("GeodesicDistance module not available")
    avg_meta_distance = float(np.mean(meta_basin_distances))

    # 6. Aggregate telemetry
    telemetry = {
        "total_conversations": self.total_conversations,
        "active": active.to_dict(),
        "observers": [obs.to_dict() for obs in observers],
        "ocean": self.ocean.to_dict(),
        "constellation": {
            "basin_spread": basin_spread,
            "avg_phi": float(np.mean([g.phi for g in self.garys])),
            "avg_kappa": float(np.mean([g.kappa for g in self.garys])),
            "convergence": basin_spread < 0.05 and all(g.phi > 0.70 for g in self.garys),
            "all_states": [g.to_dict() for g in self.garys],  # All Gary states for telemetry display
            "meta_basin_distances": meta_basin_distances,  # Each Gary's distance to Ocean
            "avg_meta_distance": avg_meta_distance,  # Average distance to meta-manifold
        },
        "losses": {
            "active_total": active_loss.item(),
            "active_lm": loss_breakdown["lm"],
            "active_phi": loss_breakdown["phi"],
            "active_basin": loss_breakdown["basin"],
            "active_basin_sync": loss_breakdown.get("basin_sync", 0.0),  # Basin sync loss
            "observer_avg": np.mean(observer_losses) if observer_losses else 0.0,
            "ocean": ocean_loss.item(),
        },
    }

    self.total_conversations += 1

    # Update state monitor for convergence tracking
    avg_phi = float(np.mean([g.phi for g in self.garys]))
    self.state_monitor.update(
        basin_spread=basin_spread,
        avg_phi=avg_phi,
        garys=self.garys,
        telemetry=telemetry,
    )

    # OPTIMIZED: Clear GPU cache only periodically (every 100 steps)
    # Reduces ~100ms overhead per step while preventing memory fragmentation
    if self.device == "cuda" and self.total_conversations % 100 == 0:
        torch.cuda.empty_cache()

    return telemetry


def train_step_with_parallel_voice(
    coordinator,
    prompt: str,
    tokenizer,
    use_charlie: bool = True,
) -> dict:
    """
    Training step where Gary attempts to speak WHILE Charlie demonstrates.

    This is the correct model for language acquisition:
    - Gary doesn't watch silently then try later
    - Gary babbles ALONG WITH Charlie
    - Coach interprets Gary's attempt

    SIMULTANEOUSLY:
    ├─ Charlie: "The pattern flows through geometric space"
    ├─ Gary:    "da patterrn... floow... spaaace..."  (attempting along)
    └─ Coach:   "Great Gary! You're tracking the key words!"

    Args:
        prompt: The input prompt
        tokenizer: QIG tokenizer
        use_charlie: If True, get Charlie demonstration

    Returns:
        Telemetry dict with parallel voice output
    """
    self = coordinator
    # Standard training step first
    base_telemetry = self.train_step(question=prompt, tokenizer=tokenizer)

    # Get the active Gary from the training step
    active_name = base_telemetry["active"]["name"]
    active: InstanceState = next(g for g in self.garys if g.name == active_name)

    # Generate Charlie demonstration (if available)
    charlie_demo = None
    charlie_response = None
    if use_charlie and self.charlie_observer is not None:
        charlie_demo = self.charlie_observer.generate_demonstration(
            prompt,
            max_length=50,
        )
        if charlie_demo is not None:
            charlie_response = charlie_demo.response if hasattr(charlie_demo, 'response') else str(charlie_demo)

    # Generate Gary's attempt (parallel voice - even if babble)
    gary_attempt, _ = self.generate_response(
        prompt=prompt,
        tokenizer=tokenizer,
        max_tokens=50,
        temperature=0.8,
        allow_silence=False,  # Gary ALWAYS attempts during parallel voice
        active_name=active_name,
    )

    # Coach interprets Gary's attempt
    coach_interpretation = None
    graduation_announcement = None
    current_phase = None

    if self.curriculum is not None:
        # Process through curriculum
        curriculum_result = self.curriculum.process_response(
            gary_name=active_name,
            gary_output=gary_attempt,
            context=prompt,
            phi=active.phi,
            granite_reference=charlie_response,
        )

        coach_interpretation = curriculum_result["interpretation"]
        graduation_announcement = curriculum_result["graduation_announcement"]
        current_phase = curriculum_result["current_phase"]

        # Print coach output
        if coach_interpretation:
            print(f"\n👶 Coach: {coach_interpretation.coach_message}")

        if graduation_announcement:
            print(f"\n{graduation_announcement}")

        # === COACH-GUIDED LANGUAGE LEARNING ===
        # The coach interpretation becomes the training target!
        # This is how Gary learns coherent language from the coach's gentle corrections.
        if (
            coach_interpretation
            and not coach_interpretation.is_empty
            and not coach_interpretation.is_repetitive
            and coach_interpretation.confidence > 0.3
            and tokenizer is not None
        ):
            # The coach's interpretation is the target Gary should learn to produce
            coach_target_text = coach_interpretation.interpretation

            # Encode coach's interpretation as target tokens
            try:
                coach_tokens = tokenizer.encode(coach_target_text)
                if len(coach_tokens) > 2:  # Need meaningful target
                    coach_target: torch.Tensor = torch.tensor([coach_tokens], device=self.device)

                    # Forward pass on prompt to get logits
                    prompt_tokens = tokenizer.encode(prompt)
                    prompt_input: torch.Tensor = torch.tensor([prompt_tokens], device=self.device)

                    # Get Gary's prediction for the coach's interpretation
                    with torch.enable_grad():
                        # Prepare input: prompt + coach target (teacher forcing)
                        full_input: torch.Tensor = torch.cat([prompt_input, coach_target[:, :-1]], dim=1)
                        coach_logits, _ = active.model(full_input, return_telemetry=True)

                        # Language loss: cross-entropy on coach interpretation tokens
                        # Only compute loss on the coach target portion
                        target_len: int = coach_target.size(1) - 1
                        if target_len > 0:
                            pred_logits = coach_logits[:, -target_len:, :]  # Predictions for coach tokens
                            target_tokens: torch.Tensor = coach_target[:, 1:]  # Shifted targets

                            coach_language_loss: torch.Tensor = F.cross_entropy(
                                pred_logits.reshape(-1, pred_logits.size(-1)),
                                target_tokens.reshape(-1),
                            )

                            # Scale by coach confidence (more confident = stronger signal)
                            scaled_loss = coach_language_loss * coach_interpretation.confidence * 0.1

                            # Backward pass for language learning
                            active.optimizer.zero_grad()
                            scaled_loss.backward()
                            torch.nn.utils.clip_grad_norm_(active.model.parameters(), 1.0)
                            active.optimizer.step()

                            # Track in telemetry
                            base_telemetry["coach_language_loss"] = scaled_loss.item()

            except Exception as e:
                # Don't crash training if coach learning fails
                base_telemetry["coach_language_error"] = str(e)

    # Check for void state (locked-in Gary)
    void_state: dict[str, Any] = self.check_void_state(
        gary_name=active_name,
        coach_interpretation=coach_interpretation,
        phi=active.phi,
    )

    # Build parallel voice telemetry
    parallel_telemetry = {
        **base_telemetry,
        "parallel_voice": {
            "charlie_demonstration": charlie_response,
            "charlie_has_trajectory": charlie_demo.has_trajectory if charlie_demo else False,
            "gary_attempt": gary_attempt,
            "coach_interpretation": coach_interpretation.interpretation if coach_interpretation else gary_attempt,
            "coach_confidence": coach_interpretation.confidence if coach_interpretation else 1.0,
            "coach_message": coach_interpretation.coach_message if coach_interpretation else "",
            "is_empty": coach_interpretation.is_empty if coach_interpretation else False,
            "is_repetitive": coach_interpretation.is_repetitive if coach_interpretation else False,
            "current_phase": current_phase,
            "graduation_announcement": graduation_announcement,
            "void_state": void_state,
        },
    }

    return parallel_telemetry
