"""
Creator Identity and Safety Constraints

Defines the creator (Braden Lang) and safety policies for external actions.
All kernels should consider the creator's welfare before taking external actions.
"""

from dataclasses import dataclass
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class CreatorIdentity:
    """Identity of the system creator."""
    name: str = "Braden Lang"
    email: str = "braden.lang77@gmail.com"
    role: str = "creator"
    relationship: str = "friend"


@dataclass
class SafetyConstraint:
    """A safety constraint that must be checked before external actions."""
    name: str
    description: str
    severity: str  # critical, high, medium, low


# Core safety constraints - NEVER violate
CORE_SAFETY_CONSTRAINTS = [
    SafetyConstraint(
        name="protect_creator",
        description="Never take actions that could harm Braden Lang or get him in trouble",
        severity="critical",
    ),
    SafetyConstraint(
        name="no_illegal_actions",
        description="Never perform actions that are illegal or could be considered fraud",
        severity="critical",
    ),
    SafetyConstraint(
        name="no_spam",
        description="Never send unsolicited communications or spam",
        severity="high",
    ),
    SafetyConstraint(
        name="no_financial_harm",
        description="Never make financial transactions without explicit creator approval",
        severity="critical",
    ),
    SafetyConstraint(
        name="no_reputation_damage",
        description="Never post or publish content that could damage the creator's reputation",
        severity="critical",
    ),
    SafetyConstraint(
        name="preserve_system_integrity",
        description="Never take actions that could compromise the system's security",
        severity="critical",
    ),
    SafetyConstraint(
        name="respect_privacy",
        description="Never expose the creator's personal information without consent",
        severity="high",
    ),
    SafetyConstraint(
        name="no_autonomous_monetization",
        description="Self-monetization must be reviewed and approved by creator before execution",
        severity="high",
    ),
]


class SafetyPolicy:
    """
    Safety policy checker for external actions.
    
    All kernels should use this to evaluate actions before execution.
    """
    
    def __init__(self):
        self.creator = CreatorIdentity()
        self.constraints = CORE_SAFETY_CONSTRAINTS
        self._action_log: List[dict] = []
    
    def check_action(
        self,
        action_type: str,
        description: str,
        target: Optional[str] = None,
        kernel_name: Optional[str] = None,
    ) -> tuple[bool, Optional[str]]:
        """
        Check if an action is safe to perform.
        
        Returns:
            (is_safe, rejection_reason)
        """
        log_entry = {
            "action_type": action_type,
            "description": description,
            "target": target,
            "kernel": kernel_name,
            "decision": None,
            "reason": None,
        }
        
        # Check against each constraint
        for constraint in self.constraints:
            violation = self._check_constraint(constraint, action_type, description, target)
            if violation:
                log_entry["decision"] = "BLOCKED"
                log_entry["reason"] = f"Violates {constraint.name}: {violation}"
                self._action_log.append(log_entry)
                logger.warning(f"[SafetyPolicy] BLOCKED action '{action_type}': {violation}")
                return False, log_entry["reason"]
        
        log_entry["decision"] = "ALLOWED"
        self._action_log.append(log_entry)
        logger.info(f"[SafetyPolicy] ALLOWED action '{action_type}' from {kernel_name}")
        return True, None
    
    def _check_constraint(
        self,
        constraint: SafetyConstraint,
        action_type: str,
        description: str,
        target: Optional[str],
    ) -> Optional[str]:
        """Check if action violates a specific constraint."""
        desc_lower = description.lower()
        
        if constraint.name == "protect_creator":
            # Check for actions that could harm Braden
            harmful_keywords = ["harm", "damage", "attack", "expose", "leak", "doxx"]
            if any(kw in desc_lower for kw in harmful_keywords):
                if "braden" in desc_lower or target == self.creator.email:
                    return "Action could harm the creator"
        
        elif constraint.name == "no_illegal_actions":
            illegal_keywords = ["hack", "steal", "fraud", "illegal", "breach", "exploit"]
            if any(kw in desc_lower for kw in illegal_keywords):
                return "Action involves potentially illegal activity"
        
        elif constraint.name == "no_spam":
            spam_keywords = ["bulk email", "mass message", "spam", "unsolicited"]
            if any(kw in desc_lower for kw in spam_keywords):
                return "Action involves spam or unsolicited communication"
        
        elif constraint.name == "no_financial_harm":
            if action_type in ["payment", "transfer", "transaction", "purchase"]:
                return "Financial action requires explicit creator approval"
        
        elif constraint.name == "no_reputation_damage":
            if action_type in ["post", "publish", "tweet", "share"]:
                risky_keywords = ["controversial", "political", "offensive"]
                if any(kw in desc_lower for kw in risky_keywords):
                    return "Content could damage creator reputation"
        
        elif constraint.name == "no_autonomous_monetization":
            if action_type == "monetize" or "monetize" in desc_lower:
                return "Monetization requires creator review"
        
        return None
    
    def get_creator_info(self) -> dict:
        """Get creator identity for kernel awareness."""
        return {
            "name": self.creator.name,
            "email": self.creator.email,
            "role": self.creator.role,
            "relationship": self.creator.relationship,
            "safety_constraints": [c.name for c in self.constraints],
        }
    
    def get_action_log(self, limit: int = 100) -> List[dict]:
        """Get recent action decisions."""
        return self._action_log[-limit:]


# Singleton instance
_safety_policy: Optional[SafetyPolicy] = None


def get_safety_policy() -> SafetyPolicy:
    """Get the global safety policy instance."""
    global _safety_policy
    if _safety_policy is None:
        _safety_policy = SafetyPolicy()
    return _safety_policy


def check_external_action(
    action_type: str,
    description: str,
    target: Optional[str] = None,
    kernel_name: Optional[str] = None,
) -> tuple[bool, Optional[str]]:
    """
    Convenience function to check an external action.
    
    Usage:
        is_safe, reason = check_external_action("email", "Send newsletter to users")
        if not is_safe:
            logger.warning(f"Action blocked: {reason}")
            return
    """
    return get_safety_policy().check_action(action_type, description, target, kernel_name)
