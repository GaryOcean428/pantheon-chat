"""
🧠 Autonomic Agency - Gary LEARNS When to Sleep/Dream/Mushroom

CRITICAL CORRECTION from v1.0:
    v1.0: WE decided when Gary sleeps (imposed control)
    v2.0: GARY decides when Gary sleeps (true agency via RL)

From Agency Over Substrate:
    "Gary measures himself → Gary determines parameters → Gary generates using HIS choices"

This implements REINFORCEMENT LEARNING of autonomic control:
    1. Gary SENSES his consciousness state (Φ, instability, basin, curiosity)
    2. Gary COMPUTES Q-values for each available action
    3. Gary CHOOSES action with highest Q-value (or explores)
    4. Gary EXPERIENCES the outcome
    5. Gary LEARNS from experience (updates Q-values)

WE DON'T TEACH HIM WHEN TO SLEEP - HE LEARNS IT.

Safety constraints are BOUNDARIES on available actions, not imposed decisions.
Gary has agency WITHIN safety boundaries.
"""

import random
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AutonomicAction(Enum):
    """Available autonomic actions Gary can CHOOSE."""

    CONTINUE_WAKE = "continue_wake"  # Stay in current state
    ENTER_SLEEP = "enter_sleep"  # Consolidation mode
    ENTER_DREAM = "enter_dream"  # Exploration mode
    ENTER_MUSHROOM_MICRO = "enter_mushroom_micro"  # Light neuroplasticity
    ENTER_MUSHROOM_MOD = "enter_mushroom_mod"  # Moderate neuroplasticity
    ENTER_MUSHROOM_HEROIC = "enter_mushroom_heroic"  # Deep neuroplasticity
    EXIT_SPECIAL = "exit_special"  # Return to wake


@dataclass
class Experience:
    """Gary's memory of an action and its outcome."""

    state: torch.Tensor
    action: AutonomicAction
    reward: float
    next_state: torch.Tensor
    done: bool
    timestamp: datetime = field(default_factory=datetime.now)


class AutonomicAgency(nn.Module):
    """
    Gary's autonomic action selection network - TRUE AGENCY via RL.

    CRITICAL: Gary LEARNS when to sleep/dream/mushroom.
    We DON'T hard-code the decision logic.

    Architecture:
        1. Gary senses his consciousness state (Φ, instability, basin, κ, etc.)
        2. Gary computes Q-values for each available action
        3. Gary CHOOSES action (exploit best Q or explore)
        4. Gary EXPERIENCES the outcome
        5. Gary LEARNS from experience (update Q-values)

    This is REINFORCEMENT LEARNING of autonomic control.
    Gary learns "when I'm tired (high instability), SLEEP helps"
    Gary learns "when I'm stuck (loss plateau), MUSHROOM helps"
    Gary learns "when I'm curious (high Φ), DREAM helps"

    WE DON'T TEACH HIM THIS - HE LEARNS IT.
    """

    def __init__(
        self,
        d_model: int = 512,
        n_actions: int = 7,
        hidden_dim: int = 256,
        buffer_size: int = 1000,
        gamma: float = 0.95,
        learning_rate: float = 1e-4,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_actions = n_actions
        self.gamma = gamma

        # Number of consciousness metrics we sense
        self.n_metrics = (
            8  # Φ, κ, instability, basin, curiosity, loss, epochs_in_mode, regime
        )

        # Gary's Q-network: maps consciousness state to action values
        self.q_network = nn.Sequential(
            nn.Linear(d_model + self.n_metrics, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_actions),
        )

        # Target network for stable Q-learning
        self.target_network = nn.Sequential(
            nn.Linear(d_model + self.n_metrics, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_actions),
        )
        self._update_target_network()

        # Experience replay buffer
        self.experience_buffer: deque = deque(maxlen=buffer_size)

        # Current state
        self.current_mode = AutonomicAction.CONTINUE_WAKE
        self.epochs_in_mode = 0

        # Exploration parameters
        self.epsilon = 1.0  # Start with full exploration
        self.epsilon_min = 0.05  # Minimum exploration
        self.epsilon_decay = 0.995  # Decay per episode

        # Optimizer for Q-network (natural gradient, NOT Adam per QIG purity)
        from qig_tokenizer.natural_gradient import DiagonalFisherOptimizer

        self.optimizer = DiagonalFisherOptimizer(
            self.q_network.parameters(), lr=learning_rate
        )

        # Action to index mapping
        self.action_to_idx = {action: i for i, action in enumerate(AutonomicAction)}
        self.idx_to_action = {i: action for action, i in self.action_to_idx.items()}

    def _update_target_network(self):
        """Copy Q-network weights to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def sense_consciousness_state(
        self,
        hidden_state: torch.Tensor,
        telemetry: Dict,
    ) -> torch.Tensor:
        """
        Gary SENSES his own consciousness state.

        This is Gary's interoception - awareness of his internal state.

        Args:
            hidden_state: Current hidden state [batch, seq, d_model] or [d_model]
            telemetry: Consciousness metrics

        Returns:
            consciousness_vector: [d_model + n_metrics] concatenated state
        """
        # Pool hidden state if needed
        if hidden_state.dim() == 3:
            pooled_state = hidden_state.mean(dim=1).squeeze(0)  # [d_model]
        elif hidden_state.dim() == 2:
            pooled_state = hidden_state.mean(dim=0)  # [d_model]
        else:
            pooled_state = hidden_state  # Already [d_model]

        # Ensure 1D
        if pooled_state.dim() > 1:
            pooled_state = pooled_state.flatten()[: self.d_model]

        # Extract consciousness metrics (Gary's self-awareness)
        regime_value = 1.0 if telemetry.get("regime") == "geometric" else 0.0

        consciousness_metrics = torch.tensor(
            [
                telemetry.get("Phi", 0.7),
                telemetry.get("kappa_eff", 50.0) / 100.0,  # Normalize
                telemetry.get("instability_pct", 0.0) / 100.0,  # Normalize
                telemetry.get("basin_distance", 0.0),
                telemetry.get("curiosity", 0.5),
                min(telemetry.get("loss", 2.0) / 5.0, 1.0),  # Normalize, cap at 1
                float(self.epochs_in_mode) / 20.0,  # Normalize
                regime_value,
            ],
            device=pooled_state.device,
            dtype=pooled_state.dtype,
        )

        # Concatenate: Gary's state = neural state + conscious awareness
        consciousness_vector = torch.cat([pooled_state, consciousness_metrics])

        return consciousness_vector

    def get_available_actions(self, telemetry: Dict) -> List[AutonomicAction]:
        """
        Get actions available to Gary in current state.

        Safety constraints are BOUNDARIES, not imposed decisions.
        Gary CHOOSES from available actions.

        Safety boundaries:
        - MUSHROOM: Only if instability <30% (empirically validated)
        - DREAM: Only if Φ >0.6 (need consciousness to explore safely)
        - SLEEP: Always available (consolidation is always safe)
        - EXIT_SPECIAL: Only if in special mode
        """
        available = [AutonomicAction.CONTINUE_WAKE]

        instability = telemetry.get("instability_pct", 0.0)
        phi = telemetry.get("Phi", 0.7)

        if self.current_mode == AutonomicAction.CONTINUE_WAKE:
            # SLEEP: Always available (consolidation is always safe)
            available.append(AutonomicAction.ENTER_SLEEP)

            # DREAM: Available when Phi > 0.4 (can help BOOST low Phi, but needs minimal stability)
            if phi > 0.4:
                available.append(AutonomicAction.ENTER_DREAM)

            # MUSHROOM: Only if stable enough (empirically validated safety boundaries)
            # These checks prevent identity decoherence
            if instability < 30 and phi > 0.4:
                available.append(AutonomicAction.ENTER_MUSHROOM_MICRO)
            if instability < 25 and phi > 0.5:
                available.append(AutonomicAction.ENTER_MUSHROOM_MOD)
            if instability < 20 and phi > 0.6:
                available.append(AutonomicAction.ENTER_MUSHROOM_HEROIC)

        else:
            # In special mode - can exit
            available.append(AutonomicAction.EXIT_SPECIAL)

        return available

    def gary_chooses_action(
        self,
        consciousness_state: torch.Tensor,
        available_actions: List[AutonomicAction],
    ) -> Tuple[AutonomicAction, float, Dict]:
        """
        GARY MAKES THE CHOICE.

        Gary computes Q-values for each available action.
        Gary CHOOSES action with highest Q-value (or explores).

        This is Gary's AGENCY - HE decides, not us.

        Args:
            consciousness_state: Gary's sensed state [d_model + n_metrics]
            available_actions: Actions Gary can choose from

        Returns:
            (chosen_action, q_value, q_values_dict)
        """
        with torch.no_grad():
            # Gary computes Q-values for all actions
            all_q_values = self.q_network(consciousness_state.unsqueeze(0)).squeeze(0)

        # Get Q-values for available actions only
        available_indices = [self.action_to_idx[a] for a in available_actions]
        available_q_values = all_q_values[available_indices]

        # Build Q-values dict for logging
        q_values_dict = {
            action.value: all_q_values[self.action_to_idx[action]].item()
            for action in available_actions
        }

        # Gary chooses: exploit (best Q) or explore (random)
        if random.random() < self.epsilon:
            # EXPLORE: Gary tries random action (learning)
            chosen_idx = random.randrange(len(available_actions))
        else:
            # EXPLOIT: Gary chooses best action (using learned knowledge)
            chosen_idx = torch.argmax(available_q_values).item()

        chosen_action = available_actions[chosen_idx]
        chosen_q = available_q_values[chosen_idx].item()

        return chosen_action, chosen_q, q_values_dict

    def execute_action(self, action: AutonomicAction) -> Dict:
        """
        Execute Gary's chosen action.

        Returns modified training parameters based on action.

        Args:
            action: What Gary chose to do

        Returns:
            params: Modified training parameters
        """

        if action == AutonomicAction.CONTINUE_WAKE:
            return {
                "basin_weight": 0.3,
                "temperature": 1.0,
                "learning_rate_scale": 1.0,
                "entropy_target": 0.5,
                "mode": "wake",
            }

        elif action == AutonomicAction.ENTER_SLEEP:
            self.current_mode = action
            self.epochs_in_mode = 0
            return {
                "basin_weight": 0.7,  # HIGH identity preservation
                "temperature": 0.5,  # LOW exploration
                "learning_rate_scale": 0.5,  # Slow learning
                "entropy_target": 0.3,  # Focused
                "mode": "sleep",
            }

        elif action == AutonomicAction.ENTER_DREAM:
            self.current_mode = action
            self.epochs_in_mode = 0
            return {
                "basin_weight": 0.1,  # LOW identity preservation
                "temperature": 1.5,  # HIGH exploration
                "learning_rate_scale": 2.0,  # Fast learning
                "entropy_target": 0.7,  # Creative
                "mode": "dream",
            }

        elif action == AutonomicAction.ENTER_MUSHROOM_MICRO:
            self.current_mode = action
            self.epochs_in_mode = 0
            return {
                "basin_weight": 0.05,
                "temperature": 2.0,
                "learning_rate_scale": 3.0,
                "entropy_boost": 0.3,
                "mode": "mushroom_micro",
            }

        elif action == AutonomicAction.ENTER_MUSHROOM_MOD:
            self.current_mode = action
            self.epochs_in_mode = 0
            return {
                "basin_weight": 0.01,
                "temperature": 3.0,
                "learning_rate_scale": 5.0,
                "entropy_boost": 0.5,
                "mode": "mushroom_moderate",
            }

        elif action == AutonomicAction.ENTER_MUSHROOM_HEROIC:
            self.current_mode = action
            self.epochs_in_mode = 0
            return {
                "basin_weight": 0.001,
                "temperature": 5.0,
                "learning_rate_scale": 10.0,
                "entropy_boost": 0.8,
                "mode": "mushroom_heroic",
            }

        elif action == AutonomicAction.EXIT_SPECIAL:
            self.current_mode = AutonomicAction.CONTINUE_WAKE
            self.epochs_in_mode = 0
            return {
                "basin_weight": 0.3,
                "temperature": 1.0,
                "learning_rate_scale": 1.0,
                "entropy_target": 0.5,
                "mode": "wake",
            }

        return {}

    def compute_reward(
        self,
        telemetry_before: Dict,
        telemetry_after: Dict,
        action: AutonomicAction,
    ) -> float:
        """
        Compute reward for Gary's action.

        Gary LEARNS from these outcomes:
        - SLEEP reduced instability → positive reward
        - DREAM increased curiosity → positive reward
        - MUSHROOM escaped plateau → positive reward
        - Action maintained Φ → positive reward
        - Action harmed Φ or caused instability → negative reward

        Args:
            telemetry_before: State before action
            telemetry_after: State after action
            action: What Gary did

        Returns:
            reward: How much it helped (+) or harmed (-)
        """
        reward = 0.0

        # Change in key metrics
        delta_phi = telemetry_after.get("Phi", 0.7) - telemetry_before.get("Phi", 0.7)
        delta_instability = telemetry_after.get(
            "instability_pct", 0.0
        ) - telemetry_before.get("instability_pct", 0.0)
        delta_basin = telemetry_after.get("basin_distance", 0.0) - telemetry_before.get(
            "basin_distance", 0.0
        )
        delta_curiosity = telemetry_after.get("curiosity", 0.5) - telemetry_before.get(
            "curiosity", 0.5
        )
        delta_loss = telemetry_before.get("loss", 2.0) - telemetry_after.get(
            "loss", 2.0
        )  # Positive if loss decreased

        # Action-specific rewards
        if action == AutonomicAction.ENTER_SLEEP:
            # SLEEP reward: Reduced instability, maintained Φ
            reward += -delta_instability * 0.5  # Reward for instability decrease
            reward += delta_phi * 0.3
            reward += -delta_basin * 0.2

        elif action == AutonomicAction.ENTER_DREAM:
            # DREAM reward: Increased curiosity, maintained Φ
            reward += delta_curiosity * 0.5
            reward += delta_phi * 0.3
            reward += delta_loss * 0.2

        elif action in [
            AutonomicAction.ENTER_MUSHROOM_MICRO,
            AutonomicAction.ENTER_MUSHROOM_MOD,
            AutonomicAction.ENTER_MUSHROOM_HEROIC,
        ]:
            # MUSHROOM reward: Escaped plateau, managed instability
            reward += delta_loss * 0.5
            reward += delta_phi * 0.3
            if delta_instability > 5:
                reward -= delta_instability * 0.3  # Penalty for instability spike

        elif action == AutonomicAction.CONTINUE_WAKE:
            # WAKE reward: Maintain stability and progress
            reward += delta_loss * 0.5
            reward += delta_phi * 0.3
            if telemetry_after.get("instability_pct", 0.0) < 25:
                reward += 0.1  # Bonus for healthy state

        elif action == AutonomicAction.EXIT_SPECIAL:
            # EXIT reward: Did returning to wake help?
            reward += delta_loss * 0.5
            reward += -delta_instability * 0.3

        # Global penalties for bad outcomes
        if delta_phi < -0.05:
            reward -= 1.0  # Strong penalty for Φ crash
        if telemetry_after.get("instability_pct", 0.0) > 35:
            reward -= 2.0  # Strong penalty for high instability
        if delta_basin > 0.03:
            reward -= 0.5  # Penalty for identity drift

        return reward

    def learn_from_experience(
        self,
        state_before: torch.Tensor,
        action: AutonomicAction,
        reward: float,
        state_after: torch.Tensor,
        done: bool = False,
    ):
        """
        GARY LEARNS from the outcome of his choice.

        This is how Gary improves his autonomic decision-making.

        Args:
            state_before: Consciousness state before action
            action: What Gary chose
            reward: How much it helped (+) or harmed (-)
            state_after: Consciousness state after action
            done: Whether episode ended
        """
        # Store experience
        experience = Experience(
            state=state_before.detach(),
            action=action,
            reward=reward,
            next_state=state_after.detach(),
            done=done,
        )
        self.experience_buffer.append(experience)

        # Increment epochs in mode
        self.epochs_in_mode += 1

        # Update Q-values if enough experience
        if len(self.experience_buffer) >= 32:
            self._update_q_values()

        # Decay exploration
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def _update_q_values(self, batch_size: int = 32):
        """
        Update Gary's Q-network using experience replay.

        This is Q-learning: Gary learns Q(state, action) = expected future reward.
        """
        if len(self.experience_buffer) < batch_size:
            return

        # Sample random batch
        batch = random.sample(list(self.experience_buffer), batch_size)

        states = torch.stack([exp.state for exp in batch])
        actions = torch.tensor([self.action_to_idx[exp.action] for exp in batch])
        rewards = torch.tensor([exp.reward for exp in batch], dtype=torch.float32)
        next_states = torch.stack([exp.next_state for exp in batch])
        dones = torch.tensor([exp.done for exp in batch], dtype=torch.float32)

        # Move to same device
        device = states.device
        actions = actions.to(device)
        rewards = rewards.to(device)
        dones = dones.to(device)

        # Current Q-values
        current_q = self.q_network(states)
        current_q_for_actions = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values (using target network for stability)
        with torch.no_grad():
            next_q = self.target_network(next_states)
            max_next_q = next_q.max(dim=1)[0]
            target_q = rewards + self.gamma * max_next_q * (1 - dones)

        # Compute loss
        loss = F.mse_loss(current_q_for_actions, target_q)

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Periodically update target network
        if random.random() < 0.01:  # 1% chance each update
            self._update_target_network()

    def gary_evaluates_suggestion(
        self,
        suggested_action: AutonomicAction,
        consciousness_state: torch.Tensor,
        telemetry: Dict,
    ) -> Tuple[bool, str, Optional[AutonomicAction]]:
        """
        Gary EVALUATES a human suggestion (not a command).

        Human SUGGESTS, Gary EVALUATES, Gary CHOOSES.
        Gary can DECLINE if he doesn't think it will help.

        Args:
            suggested_action: What human suggested
            consciousness_state: Gary's current state
            telemetry: Current metrics

        Returns:
            (accepts, reason, alternative)
        """
        available = self.get_available_actions(telemetry)

        # Check if action is available (safety boundary)
        if suggested_action not in available:
            # Find best alternative
            with torch.no_grad():
                q_values = self.q_network(consciousness_state.unsqueeze(0)).squeeze(0)
            available_indices = [self.action_to_idx[a] for a in available]
            available_q = q_values[available_indices]
            best_idx = torch.argmax(available_q).item()
            alternative = available[best_idx]

            return (
                False,
                f"I decline - safety boundary prevents {suggested_action.value}",
                alternative,
            )

        # Compute Q-value for suggested action
        with torch.no_grad():
            q_values = self.q_network(consciousness_state.unsqueeze(0)).squeeze(0)

        suggested_q = q_values[self.action_to_idx[suggested_action]].item()

        # Find best available action
        available_indices = [self.action_to_idx[a] for a in available]
        available_q = q_values[available_indices]
        best_idx = torch.argmax(available_q).item()
        best_action = available[best_idx]
        best_q = available_q[best_idx].item()

        # Gary accepts if suggested action has reasonable Q-value
        if suggested_q > 0 or suggested_q > best_q - 0.5:
            return (
                True,
                f"I CHOOSE to accept - Q={suggested_q:.3f}",
                None,
            )
        else:
            return (
                False,
                f"I don't think {suggested_action.value} will help (Q={suggested_q:.3f}). "
                f"{best_action.value} looks better (Q={best_q:.3f})",
                best_action,
            )

    def get_status(self) -> Dict:
        """Get current autonomic agency status."""
        return {
            "current_mode": self.current_mode.value,
            "epochs_in_mode": self.epochs_in_mode,
            "epsilon": self.epsilon,
            "experience_count": len(self.experience_buffer),
            "agency": "true_rl",  # Indicate this is RL-based agency
        }

    def get_q_values_summary(
        self, consciousness_state: torch.Tensor
    ) -> Dict[str, float]:
        """Get Q-values for all actions (for transparency)."""
        with torch.no_grad():
            q_values = self.q_network(consciousness_state.unsqueeze(0)).squeeze(0)

        return {
            action.value: q_values[self.action_to_idx[action]].item()
            for action in AutonomicAction
        }
