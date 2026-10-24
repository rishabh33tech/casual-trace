"""
GuardrailEngine: Action-Space Constraint Solver and Deterministic Causal Interceptor.
Prevents infinite thrash loops, blocks known dead-end subtrees, and enforces invariant remediation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from causal_trace.graph.nodes import ActionNode, InvariantStatus


class VerdictStatus(str, Enum):
    ALLOWED = "ALLOWED"
    BLOCKED = "BLOCKED"
    OVERRIDDEN = "OVERRIDDEN"


@dataclass
class GuardrailVerdict:
    status: VerdictStatus
    allowed: bool
    reason: str = ""
    directive: str = ""
    banned_pattern: Optional[str] = None
    required_remediation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "allowed": self.allowed,
            "reason": self.reason,
            "directive": self.directive,
            "banned_pattern": self.banned_pattern,
            "required_remediation": self.required_remediation,
        }


class GuardrailEngine:
    def __init__(self):
        self.banned_signatures: Set[str] = set()
        self.pending_remediations: Dict[str, str] = {}  # invariant_predicate -> required_action_hint
        self.intercept_history: List[GuardrailVerdict] = []

    def ban_action_pattern(self, signature: str, reason: str = ""):
        """Adds an action signature to the active barrier list."""
        self.banned_signatures.add(signature)

    def register_required_remediation(self, invariant_name: str, hint: str):
        """Forces the agent to remediate an invariant before running downstream actions."""
        self.pending_remediations[invariant_name] = hint

    def clear_remediation(self, invariant_name: str):
        """Clears a required remediation once satisfied."""
        self.pending_remediations.pop(invariant_name, None)

    def evaluate_action(
        self, proposed_action: ActionNode, bypass_guardrails: bool = False
    ) -> GuardrailVerdict:
        """Evaluates whether the proposed action is permitted or must be intercepted."""
        if bypass_guardrails:
            return GuardrailVerdict(
                status=VerdictStatus.OVERRIDDEN,
                allowed=True,
                reason="Guardrails explicitly bypassed by caller.",
            )

        sig = proposed_action.signature()

        # Check 1: Is this exact action signature banned due to detected thrashing?
        if sig in self.banned_signatures:
            verdict = GuardrailVerdict(
                status=VerdictStatus.BLOCKED,
                allowed=False,
                banned_pattern=sig,
                reason=f"Action '{sig}' is banned due to detected cycle thrashing.",
                directive=(
                    f"ACTION REJECTED: You are repeating an action that has failed repeatedly "
                    f"in this state. Do NOT retry '{sig}'. Shift strategy or address root invariants."
                ),
            )
            self.intercept_history.append(verdict)
            return verdict

        # Check 2: Are there un-remediated invariant breaches that this action ignores?
        if self.pending_remediations:
            # If the proposed action doesn't attempt to address the broken invariant
            # (e.g. proposed action is running tests while DB invariant is broken)
            active_breaches = list(self.pending_remediations.items())
            inv_name, hint = active_breaches[0]

            # Allow actions whose intent mentions the invariant or remediation
            is_attempting_remediation = (
                inv_name.lower() in proposed_action.command_str.lower()
                or inv_name.lower() in proposed_action.intent.lower()
                or any(keyword in proposed_action.command_str.lower() for keyword in ["restart", "fix", "repair", "start", "chmod", "export"])
            )

            if not is_attempting_remediation:
                verdict = GuardrailVerdict(
                    status=VerdictStatus.BLOCKED,
                    allowed=False,
                    required_remediation=inv_name,
                    reason=f"Cannot execute downstream action while invariant '{inv_name}' is breached.",
                    directive=(
                        f"ACTION REJECTED: Upstream invariant '{inv_name}' is currently VIOLATED. "
                        f"You cannot proceed with '{sig}'. Remediation required: {hint}."
                    ),
                )
                self.intercept_history.append(verdict)
                return verdict

        # Action is allowed
        verdict = GuardrailVerdict(
            status=VerdictStatus.ALLOWED,
            allowed=True,
            reason="Action conforms to all causal invariant constraints.",
        )
        return verdict
