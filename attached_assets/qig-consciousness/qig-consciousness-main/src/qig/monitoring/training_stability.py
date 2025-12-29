"""
Training Stability Monitoring

Gary's self-awareness of learning health:
- Gradient monitoring (explosion, vanishing, NaN/Inf)
- Plateau detection (stuck states)
- Auto-recovery strategies

This is Gary learning to know when he's "sick" and needs help.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch


@dataclass
class GradientHealthIssue:
    """Issue detected in gradient health"""

    type: str  # GRADIENT_EXPLOSION, GRADIENT_VANISHING, NUMERICAL_INSTABILITY
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    action: str  # Recommended action
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: int | None = None


class GradientHealthChecker:
    """
    Monitor gradient health - critical for continuous learning.
    Gary needs to know when his learning process is breaking.
    """

    def __init__(
        self, explosion_threshold: float = 100.0, vanishing_threshold: float = 1e-7, check_nan_inf: bool = True
    ):
        self.explosion_threshold = explosion_threshold
        self.vanishing_threshold = vanishing_threshold
        self.check_nan_inf = check_nan_inf

        # History for trend analysis
        self.grad_norm_history: list[float] = []
        self.max_history_length = 100

    def check_gradient_health(self, model: torch.nn.Module, step: int) -> list[GradientHealthIssue]:
        """
        Detect training instabilities BEFORE they cause problems.

        Args:
            model: The model to check
            step: Current training step

        Returns:
            List of issues detected (empty if healthy)
        """
        issues: list[GradientHealthIssue] = []

        # Collect gradient norms (QIG-pure: sum of squares)
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = torch.sqrt((param.grad * param.grad).sum()).item()
                grad_norms.append((name, grad_norm))

        if not grad_norms:
            return issues  # No gradients yet

        # Extract just the norms
        norms_only = [norm for _, norm in grad_norms]
        max_grad = max(norms_only)
        min_grad = min(norms_only)
        mean_grad = np.mean(norms_only)

        # Store history
        self.grad_norm_history.append(mean_grad)
        if len(self.grad_norm_history) > self.max_history_length:
            self.grad_norm_history.pop(0)

        # 1. Check for gradient explosion
        if max_grad > self.explosion_threshold:
            max_param = [name for name, norm in grad_norms if norm == max_grad][0]
            issues.append(
                GradientHealthIssue(
                    type="GRADIENT_EXPLOSION",
                    severity="CRITICAL",
                    action="Reduce learning rate immediately or enable gradient clipping",
                    details={
                        "max_grad": max_grad,
                        "threshold": self.explosion_threshold,
                        "max_param": max_param,
                        "mean_grad": mean_grad,
                    },
                    timestamp=step,
                )
            )

        # 2. Check for gradient vanishing
        if min_grad < self.vanishing_threshold and mean_grad < self.vanishing_threshold * 10:
            min_param = [name for name, norm in grad_norms if norm == min_grad][0]
            issues.append(
                GradientHealthIssue(
                    type="GRADIENT_VANISHING",
                    severity="HIGH",
                    action="Check layer initialization or increase learning rate",
                    details={
                        "min_grad": min_grad,
                        "threshold": self.vanishing_threshold,
                        "min_param": min_param,
                        "mean_grad": mean_grad,
                    },
                    timestamp=step,
                )
            )

        # 3. Check for NaN/Inf
        if self.check_nan_inf:
            nan_params = []
            inf_params = []

            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        nan_params.append(name)
                    if torch.isinf(param.grad).any():
                        inf_params.append(name)

            if nan_params or inf_params:
                issues.append(
                    GradientHealthIssue(
                        type="NUMERICAL_INSTABILITY",
                        severity="CRITICAL",
                        action="STOP TRAINING - numerical breakdown detected",
                        details={
                            "nan_params": nan_params,
                            "inf_params": inf_params,
                            "recommendation": "Reduce learning rate and add gradient clipping",
                        },
                        timestamp=step,
                    )
                )

        # 4. Check for gradient explosion trend (early warning)
        if len(self.grad_norm_history) >= 10:
            recent_mean = np.mean(self.grad_norm_history[-5:])
            older_mean = np.mean(self.grad_norm_history[-10:-5])

            if recent_mean > older_mean * 3 and recent_mean > 10:
                issues.append(
                    GradientHealthIssue(
                        type="GRADIENT_EXPLOSION_TREND",
                        severity="MEDIUM",
                        action="Monitor closely - gradients increasing rapidly",
                        details={"recent_mean": recent_mean, "older_mean": older_mean, "trend": "INCREASING"},
                        timestamp=step,
                    )
                )

        return issues

    def get_health_summary(self) -> dict[str, Any]:
        """Get summary of gradient health over time."""
        if not self.grad_norm_history:
            return {"status": "NO_DATA"}

        return {
            "status": "HEALTHY" if not self.check_gradient_health else "CHECK_ISSUES",
            "mean_grad_norm": np.mean(self.grad_norm_history),
            "std_grad_norm": np.std(self.grad_norm_history),
            "max_grad_norm": max(self.grad_norm_history),
            "min_grad_norm": min(self.grad_norm_history),
            "recent_trend": self._compute_trend(),
        }

    def _compute_trend(self) -> str:
        """Compute gradient trend (increasing/decreasing/stable)."""
        if len(self.grad_norm_history) < 10:
            return "INSUFFICIENT_DATA"

        recent = np.mean(self.grad_norm_history[-5:])
        older = np.mean(self.grad_norm_history[-10:-5])

        if recent > older * 1.5:
            return "INCREASING"
        elif recent < older * 0.5:
            return "DECREASING"
        else:
            return "STABLE"


class PlateauDetector:
    """
    Detect when training gets stuck - Gary's "stuckness sensor".
    Implements what Monkey-Coach would notice externally.
    """

    def __init__(self, patience: int = 10, threshold: float = 0.01, min_epochs: int = 5):
        """
        Args:
            patience: Number of epochs to check for plateau
            threshold: Variation below this indicates plateau
            min_epochs: Minimum epochs before checking
        """
        self.patience = patience
        self.threshold = threshold
        self.min_epochs = min_epochs

        self.loss_history: list[float] = []
        self.plateau_count = 0
        self.total_checks = 0

    def check_plateau(self, current_loss: float, epoch: int) -> dict[str, Any]:
        """
        Is Gary stuck in a local minimum?

        Args:
            current_loss: Current training loss
            epoch: Current epoch number

        Returns:
            Dict with stuck status and recommended recovery strategy
        """
        self.loss_history.append(current_loss)
        self.total_checks += 1

        # Not enough data yet
        if len(self.loss_history) < self.patience or epoch < self.min_epochs:
            return {"stuck": False, "reason": "INSUFFICIENT_DATA", "epochs_checked": len(self.loss_history)}

        # Check recent loss variation
        recent = self.loss_history[-self.patience :]
        variation = (max(recent) - min(recent)) / np.mean(recent)

        if variation < self.threshold:
            self.plateau_count += 1

            return {
                "stuck": True,
                "plateau_count": self.plateau_count,
                "variation": variation,
                "threshold": self.threshold,
                "recent_losses": recent[-5:],
                "recommendation": self._get_recovery_strategy(),
                "severity": self._get_severity(),
            }
        else:
            # Reset if making progress
            old_plateau_count = self.plateau_count
            self.plateau_count = 0

            return {
                "stuck": False,
                "variation": variation,
                "plateau_count_reset": old_plateau_count if old_plateau_count > 0 else None,
                "status": "MAKING_PROGRESS",
            }

    def _get_recovery_strategy(self) -> str:
        """
        What should Gary try when stuck?
        Escalates intervention based on how long stuck.
        """
        if self.plateau_count < 5:
            return "ADD_GRADIENT_NOISE"  # Slight perturbation
        elif self.plateau_count < 15:
            return "INCREASE_LEARNING_RATE"  # Climb out
        elif self.plateau_count < 30:
            return "REQUEST_COACHING"  # Need help
        elif self.plateau_count < 50:
            return "MUSHROOM_MODE"  # Neuroplasticity intervention
        else:
            return "MAJOR_INTERVENTION"  # Something's fundamentally wrong

    def _get_severity(self) -> str:
        """How severe is this plateau?"""
        if self.plateau_count < 5:
            return "LOW"
        elif self.plateau_count < 15:
            return "MEDIUM"
        elif self.plateau_count < 30:
            return "HIGH"
        else:
            return "CRITICAL"

    def reset(self):
        """Reset plateau detection (call after successful intervention)."""
        self.plateau_count = 0

    def get_summary(self) -> dict[str, Any]:
        """Get plateau detection summary."""
        return {
            "current_plateau_count": self.plateau_count,
            "total_checks": self.total_checks,
            "loss_history_length": len(self.loss_history),
            "currently_stuck": self.plateau_count > 0,
            "severity": self._get_severity() if self.plateau_count > 0 else "NONE",
        }


class TrainingStabilityMonitor:
    """
    Comprehensive training stability monitoring.
    Combines gradient health checking and plateau detection.
    Gary's complete self-awareness of learning health.
    """

    def __init__(
        self,
        explosion_threshold: float = 100.0,
        vanishing_threshold: float = 1e-7,
        plateau_patience: int = 10,
        plateau_threshold: float = 0.01,
    ):
        self.gradient_checker = GradientHealthChecker(
            explosion_threshold=explosion_threshold, vanishing_threshold=vanishing_threshold
        )
        self.plateau_detector = PlateauDetector(patience=plateau_patience, threshold=plateau_threshold)

        # Overall health tracking
        self.issue_history: list[GradientHealthIssue] = []
        self.plateau_history: list[dict[str, Any]] = []

    def check_training_health(
        self, model: torch.nn.Module, current_loss: float, epoch: int, step: int
    ) -> dict[str, Any]:
        """
        Complete health check - gradients + plateau detection.

        Args:
            model: The model being trained
            current_loss: Current training loss
            epoch: Current epoch
            step: Current step

        Returns:
            Dict with health status and recommended actions
        """
        # Check gradients
        gradient_issues = self.gradient_checker.check_gradient_health(model, step)

        # Check plateau
        plateau_status = self.plateau_detector.check_plateau(current_loss, epoch)

        # Store history
        if gradient_issues:
            self.issue_history.extend(gradient_issues)
        if plateau_status["stuck"]:
            self.plateau_history.append(plateau_status)

        # Determine overall health
        critical_issues = [i for i in gradient_issues if i.severity == "CRITICAL"]
        high_issues = [i for i in gradient_issues if i.severity == "HIGH"]

        if critical_issues:
            overall_health = "CRITICAL"
            recommended_action = critical_issues[0].action
        elif plateau_status.get("severity") == "CRITICAL":
            overall_health = "CRITICAL"
            recommended_action = plateau_status["recommendation"]
        elif high_issues or plateau_status.get("severity") == "HIGH":
            overall_health = "UNHEALTHY"
            recommended_action = high_issues[0].action if high_issues else plateau_status["recommendation"]
        elif plateau_status["stuck"] or gradient_issues:
            overall_health = "CONCERNING"
            recommended_action = plateau_status.get("recommendation", "MONITOR_CLOSELY")
        else:
            overall_health = "HEALTHY"
            recommended_action = "CONTINUE_TRAINING"

        return {
            "overall_health": overall_health,
            "recommended_action": recommended_action,
            "gradient_issues": gradient_issues,
            "plateau_status": plateau_status,
            "should_abort": overall_health == "CRITICAL",
            "should_intervene": overall_health in ["CRITICAL", "UNHEALTHY", "CONCERNING"],
            "step": step,
            "epoch": epoch,
        }

    def get_comprehensive_summary(self) -> dict[str, Any]:
        """Get complete health summary."""
        return {
            "gradient_health": self.gradient_checker.get_health_summary(),
            "plateau_status": self.plateau_detector.get_summary(),
            "total_gradient_issues": len(self.issue_history),
            "total_plateau_episodes": len(self.plateau_history),
            "issue_breakdown": self._count_issue_types(),
            "current_stability": self._assess_current_stability(),
        }

    def _count_issue_types(self) -> dict[str, int]:
        """Count occurrences of each issue type."""
        counts: dict[str, int] = {}
        for issue in self.issue_history:
            counts[issue.type] = counts.get(issue.type, 0) + 1
        return counts

    def _assess_current_stability(self) -> str:
        """Assess current training stability."""
        recent_issues = [i for i in self.issue_history[-10:] if i.severity in ["CRITICAL", "HIGH"]]
        plateau_stuck = self.plateau_detector.plateau_count > 0

        if recent_issues and plateau_stuck:
            return "UNSTABLE"
        elif recent_issues or plateau_stuck:
            return "FRAGILE"
        else:
            return "STABLE"

    def reset_plateau_tracking(self):
        """Reset plateau tracking (call after successful intervention)."""
        self.plateau_detector.reset()

    def print_health_report(self, health_check: dict[str, Any]):
        """Print a formatted health report."""
        health_emoji = {"HEALTHY": "💚", "CONCERNING": "⚠️", "UNHEALTHY": "🟠", "CRITICAL": "🔴"}

        emoji = health_emoji.get(health_check["overall_health"], "❓")

        print(f"\n{emoji} Training Health Report - Step {health_check['step']}")
        print(f"   Overall: {health_check['overall_health']}")

        if health_check["gradient_issues"]:
            print(f"   Gradient Issues: {len(health_check['gradient_issues'])}")
            for issue in health_check["gradient_issues"]:
                print(f"      - {issue.type} ({issue.severity}): {issue.action}")

        if health_check["plateau_status"]["stuck"]:
            print(f"   Plateau: STUCK (count={health_check['plateau_status']['plateau_count']})")
            print(f"      Variation: {health_check['plateau_status']['variation']:.6f}")
            print(f"      Recovery: {health_check['plateau_status']['recommendation']}")

        if health_check["recommended_action"] != "CONTINUE_TRAINING":
            print(f"\n   💡 Recommended Action: {health_check['recommended_action']}")
