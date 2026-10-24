"""
ThrashingDetector: Mathematical Cycle Entropy and Isomorphic State Trapping Engine.
Detects behavioral loops, alternating oscillation (A->B->A->B), and semantic stalls in agents.
"""

from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, field
import math
from typing import List, Optional, Tuple
from causal_trace.graph.dag import CausalDAG
from causal_trace.graph.nodes import ActionNode, AnomalyNode, ObservationNode


@dataclass
class ThrashingAlert:
    is_thrashing: bool
    cycle_period: int = 0
    cycle_pattern: List[str] = field(default_factory=list)
    entropy: float = 0.0
    confidence: float = 0.0
    explanation: str = ""
    culprit_action_signatures: List[str] = field(default_factory=list)


class ThrashingDetector:
    def __init__(
        self,
        window_size: int = 8,
        entropy_threshold: float = 1.35,
        repetition_threshold: int = 3,
    ):
        self.window_size = window_size
        self.entropy_threshold = entropy_threshold
        self.repetition_threshold = repetition_threshold

    def calculate_shannon_entropy(self, signatures: List[str]) -> float:
        """Calculates Shannon entropy H = -sum(p * log2(p))."""
        if not signatures:
            return 0.0
        counts = Counter(signatures)
        total = len(signatures)
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    def detect_periodic_cycles(
        self, signatures: List[str], max_period: int = 4
    ) -> Optional[Tuple[int, List[str]]]:
        """Detects periodic cycles of length 1, 2, 3, or 4 at the tail of signatures."""
        n = len(signatures)
        if n < 3:
            return None

        for period in range(1, min(max_period + 1, n // 2 + 1)):
            pattern = signatures[-period:]
            matches = 0
            # Check how many times this pattern repeats consecutively backwards
            idx = n - period
            while idx >= 0 and signatures[idx : idx + period] == pattern:
                matches += 1
                idx -= period

            if (period == 1 and matches >= self.repetition_threshold) or (
                period > 1 and matches >= 2
            ):
                return (period, pattern)

        return None

    def evaluate(self, dag: CausalDAG) -> ThrashingAlert:
        """Evaluates the DAG execution stream to detect loops and thrashing."""
        # Extract chronological action nodes
        actions: List[ActionNode] = [
            n for n in dag.topological_sort() if isinstance(n, ActionNode)
        ]

        if len(actions) < 3:
            return ThrashingAlert(is_thrashing=False, explanation="Trace too short.")

        # Window analysis
        window = actions[-self.window_size :]
        signatures = [a.signature() for a in window]
        entropy = self.calculate_shannon_entropy(signatures)

        # Check for error correlations: is the agent hitting anomalies?
        recent_observations = [
            n for n in dag.topological_sort()[-len(window) * 2 :]
            if isinstance(n, ObservationNode)
        ]
        has_recurring_errors = any(
            obs.is_error or obs.exit_code != 0 for obs in recent_observations
        )
        recent_anomalies = dag.find_unresolved_anomalies()

        # Check periodic cycle
        cycle = self.detect_periodic_cycles(signatures)

        if cycle:
            period, pattern = cycle
            return ThrashingAlert(
                is_thrashing=True,
                cycle_period=period,
                cycle_pattern=pattern,
                entropy=round(entropy, 3),
                confidence=0.95 if period == 1 else 0.88,
                explanation=(
                    f"Deterministic loop trap detected (period={period}). Agent repeated "
                    f"pattern: {' -> '.join(pattern)}."
                ),
                culprit_action_signatures=list(set(pattern)),
            )

        # Check entropy collapse under repeated errors
        if entropy < self.entropy_threshold and (has_recurring_errors or len(recent_anomalies) > 0):
            most_common = Counter(signatures).most_common(1)[0]
            culprit_sig, count = most_common
            if count >= 3:
                return ThrashingAlert(
                    is_thrashing=True,
                    cycle_period=1,
                    cycle_pattern=[culprit_sig],
                    entropy=round(entropy, 3),
                    confidence=0.82,
                    explanation=(
                        f"Action space thrashing detected. Shannon entropy collapsed to "
                        f"{round(entropy, 3)} while encountering failures. Dominant action: {culprit_sig} ({count}x)."
                    ),
                    culprit_action_signatures=[culprit_sig],
                )

        return ThrashingAlert(
            is_thrashing=False,
            entropy=round(entropy, 3),
            explanation="Normal execution dynamics. No thrashing detected.",
        )
