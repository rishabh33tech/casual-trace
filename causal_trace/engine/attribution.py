"""
CounterfactualEngine: Structural Root Cause Attribution for Agent Failures.
Pinpoints the First Invariant Breach (FIB) and isolates root causes from downstream symptoms.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from causal_trace.graph.dag import CausalDAG
from causal_trace.graph.edges import EdgeType
from causal_trace.graph.nodes import (
    ActionNode,
    AnomalyNode,
    InvariantNode,
    InvariantStatus,
    NodeType,
)


@dataclass
class CausalCandidate:
    node_id: str
    node_type: NodeType
    action_signature: str
    step_index: int
    score: float
    reason: str
    associated_invariant: Optional[str] = None


@dataclass
class RootCauseReport:
    anomaly_id: str
    symptom_summary: str
    primary_culprit_id: Optional[str]
    primary_culprit_signature: str
    confidence: float
    broken_invariant: Optional[str]
    causal_chain: List[str]
    remediation_directive: str
    candidates: List[CausalCandidate] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "anomaly_id": self.anomaly_id,
            "symptom_summary": self.symptom_summary,
            "primary_culprit_id": self.primary_culprit_id,
            "primary_culprit_signature": self.primary_culprit_signature,
            "confidence": round(self.confidence, 3),
            "broken_invariant": self.broken_invariant,
            "causal_chain": self.causal_chain,
            "remediation_directive": self.remediation_directive,
            "candidates": [
                {
                    "node_id": c.node_id,
                    "action": c.action_signature,
                    "step": c.step_index,
                    "score": round(c.score, 3),
                    "reason": c.reason,
                    "invariant": c.associated_invariant,
                }
                for c in self.candidates
            ],
        }


class CounterfactualEngine:
    def __init__(
        self,
        invariant_weight: float = 12.0,
        direct_dependency_weight: float = 5.0,
        decay_gamma: float = 0.95,
    ):
        self.invariant_weight = invariant_weight
        self.direct_dependency_weight = direct_dependency_weight
        self.decay_gamma = decay_gamma

    def diagnose_anomaly(
        self, dag: CausalDAG, anomaly_id: str
    ) -> Optional[RootCauseReport]:
        """Performs backward structural attribution to isolate the primary culprit."""
        anomaly = dag.get_node(anomaly_id)
        if not isinstance(anomaly, AnomalyNode):
            return None

        # 1. Gather all upstream ancestors
        ancestor_ids = dag.ancestors(anomaly_id)
        if not ancestor_ids:
            # Fallback to direct parents or preceding actions if graph edges were sparse
            ancestor_ids = {
                n.id
                for n in dag.nodes.values()
                if n.step_index < anomaly.step_index and isinstance(n, ActionNode)
            }

        candidates: List[CausalCandidate] = []

        # 2. Inspect broken invariants
        violated_invariants = dag.find_violated_invariants()

        # Build mapping of violated invariant -> causing action(s)
        inv_breakers: Dict[str, List[ActionNode]] = {}
        for inv in violated_invariants:
            # Find actions with incoming INVALIDATES edge to this invariant or actions directly upstream
            breakers = []
            for edge in dag.get_in_edges(inv.id, EdgeType.INVALIDATES):
                node = dag.get_node(edge.source_id)
                if isinstance(node, ActionNode):
                    breakers.append(node)
            if not breakers:
                # Chronologically closest action before invariant violation
                preceding_actions = [
                    n for n in dag.nodes.values()
                    if isinstance(n, ActionNode) and n.step_index <= inv.step_index
                ]
                if preceding_actions:
                    breakers.append(sorted(preceding_actions, key=lambda a: a.step_index)[-1])
            inv_breakers[inv.id] = breakers

        # 3. Score every ancestor action
        action_nodes = [
            dag.get_node(nid)
            for nid in ancestor_ids
            if isinstance(dag.get_node(nid), ActionNode)
        ]

        for act in action_nodes:
            score = 0.0
            reasons = []
            assoc_inv = None

            # Temporal distance
            step_distance = max(1, anomaly.step_index - act.step_index)
            time_decay = self.decay_gamma ** step_distance

            # Check if this action broke any violated invariant
            for inv_id, breakers in inv_breakers.items():
                if any(b.id == act.id for b in breakers):
                    inv_node = dag.get_node(inv_id)
                    predicate = getattr(inv_node, "predicate", inv_id)
                    score += self.invariant_weight
                    reasons.append(f"Breached invariant '{predicate}' at step {act.step_index}")
                    assoc_inv = predicate

            # Check if this action has explicit DEPENDS_ON or INVALIDATES edge to anomaly path
            paths_to_anomaly = dag.find_paths(act.id, anomaly_id)
            if paths_to_anomaly:
                score += self.direct_dependency_weight * len(paths_to_anomaly)
                reasons.append(f"Direct causal path to anomaly ({len(paths_to_anomaly)} route(s))")

            # Edge type bonus
            out_edges = dag.get_out_edges(act.id)
            for e in out_edges:
                if e.edge_type == EdgeType.INVALIDATES:
                    score += 6.0
                    reasons.append("Contains explicit INVALIDATES edge")

            # Final score modulated by decay
            final_score = score * time_decay

            if score > 0:
                candidates.append(
                    CausalCandidate(
                        node_id=act.id,
                        node_type=act.node_type,
                        action_signature=act.signature(),
                        step_index=act.step_index,
                        score=final_score,
                        reason="; ".join(reasons),
                        associated_invariant=assoc_inv,
                    )
                )

        if not candidates:
            # If no scored candidates, blame the immediate trigger action
            immediate_parents = dag.get_parents(anomaly_id)
            trigger_action = next(
                (p for p in immediate_parents if isinstance(p, ActionNode)), None
            )
            if trigger_action:
                return RootCauseReport(
                    anomaly_id=anomaly_id,
                    symptom_summary=anomaly.symptom_summary or anomaly.content,
                    primary_culprit_id=trigger_action.id,
                    primary_culprit_signature=trigger_action.signature(),
                    confidence=0.5,
                    broken_invariant=None,
                    causal_chain=[trigger_action.id, anomaly_id],
                    remediation_directive=(
                        f"Action '{trigger_action.signature()}' failed. Check immediate arguments and environment."
                    ),
                )
            return None

        # Sort candidates by score descending
        candidates.sort(key=lambda c: c.score, reverse=True)
        top = candidates[0]

        # Extract shortest causal chain
        paths = dag.find_paths(top.node_id, anomaly_id)
        causal_chain = paths[0] if paths else [top.node_id, anomaly_id]

        confidence = min(0.98, max(0.65, top.score / (self.invariant_weight + 5.0)))

        # Synthesize remediation directive
        if top.associated_invariant:
            directive = (
                f"ROOT CAUSE IDENTIFIED: Step {top.step_index} ('{top.action_signature}') "
                f"invalidated invariant '{top.associated_invariant}'. The failure at Step "
                f"{anomaly.step_index} is a downstream consequence. You must restore "
                f"'{top.associated_invariant}' before re-attempting this operation."
            )
        else:
            directive = (
                f"ROOT CAUSE IDENTIFIED: Step {top.step_index} ('{top.action_signature}') "
                f"corrupted prerequisite state. Revert or patch this action's side effects."
            )

        return RootCauseReport(
            anomaly_id=anomaly_id,
            symptom_summary=anomaly.symptom_summary or anomaly.content,
            primary_culprit_id=top.node_id,
            primary_culprit_signature=top.action_signature,
            confidence=confidence,
            broken_invariant=top.associated_invariant,
            causal_chain=causal_chain,
            remediation_directive=directive,
            candidates=candidates,
        )
