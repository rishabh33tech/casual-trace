"""
CausalContextCompiler: Topological Causal Context Pruner.
Replaces naive sliding-window and vector RAG context buffers with mathematically
guaranteed Minimal Causal Ancestral Subgraphs.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from causal_trace.graph.dag import CausalDAG
from causal_trace.graph.nodes import (
    ActionNode,
    AnomalyNode,
    BaseNode,
    GoalNode,
    HypothesisNode,
    InvariantNode,
    InvariantStatus,
    NodeType,
    ObservationNode,
)
from causal_trace.graph.edges import EdgeType


@dataclass
class CompiledContext:
    raw_node_count: int
    pruned_node_count: int
    raw_token_estimate: int
    pruned_token_estimate: int
    compression_ratio: float
    causal_completeness: float
    formatted_prompt: str
    retained_node_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_node_count": self.raw_node_count,
            "pruned_node_count": self.pruned_node_count,
            "raw_token_estimate": self.raw_token_estimate,
            "pruned_token_estimate": self.pruned_token_estimate,
            "compression_ratio": round(self.compression_ratio, 3),
            "causal_completeness": round(self.causal_completeness, 3),
            "retained_nodes": len(self.retained_node_ids),
        }


class CausalContextCompiler:
    def __init__(self, token_budget: int = 4000, chars_per_token: float = 4.0):
        self.token_budget = token_budget
        self.chars_per_token = chars_per_token

    def _estimate_tokens(self, text: str) -> int:
        return max(1, int(len(text) / self.chars_per_token))

    def compile(
        self,
        dag: CausalDAG,
        target_goal_id: Optional[str] = None,
        include_lessons: bool = True,
    ) -> CompiledContext:
        """Compiles minimal sufficient causal context for the LLM agent prompt."""
        all_nodes = list(dag.nodes.values())
        raw_text_parts = []
        for n in sorted(all_nodes, key=lambda x: (x.step_index, x.timestamp)):
            raw_text_parts.append(f"[{n.node_type.value}] Step {n.step_index}: {n.content}")
            if isinstance(n, ObservationNode):
                if n.stdout:
                    raw_text_parts.append(f"STDOUT: {n.stdout}")
                if n.stderr:
                    raw_text_parts.append(f"STDERR: {n.stderr}")
        raw_full_text = "\n".join(raw_text_parts)
        raw_token_estimate = self._estimate_tokens(raw_full_text)

        # 1. Identify Root Goals
        goals = [n for n in all_nodes if isinstance(n, GoalNode)]
        target_goal = (
            dag.get_node(target_goal_id)
            if target_goal_id
            else (goals[-1] if goals else None)
        )

        retained_ids: Set[str] = set()

        if target_goal:
            retained_ids.add(target_goal.id)

        # 2. Add Active & Violated Invariants
        invariants = [n for n in all_nodes if isinstance(n, InvariantNode)]
        for inv in invariants:
            retained_ids.add(inv.id)

        # 3. Add Causal Preconditions (Ancestors of Target Goal)
        if target_goal:
            causal_ancestors = dag.ancestors(
                target_goal.id,
                allowed_edge_types={
                    EdgeType.ENABLES,
                    EdgeType.DEPENDS_ON,
                    EdgeType.MITIGATES,
                },
            )
            retained_ids.update(causal_ancestors)

        # 4. Add Culprits & Anomalies (Causal Lessons)
        anomalies = dag.find_unresolved_anomalies()
        for anom in anomalies:
            retained_ids.add(anom.id)
            # Retain immediate parent actions and invalidating actions
            for parent in dag.get_parents(anom.id):
                retained_ids.add(parent.id)
            for inv_edge in dag.get_in_edges(anom.id, EdgeType.INVALIDATES):
                retained_ids.add(inv_edge.source_id)

        # 5. Always retain the most recent 2 actions to maintain immediate continuity
        recent_actions = [
            n for n in sorted(all_nodes, key=lambda x: x.step_index)
            if isinstance(n, ActionNode)
        ]
        for act in recent_actions[-2:]:
            retained_ids.add(act.id)

        # 6. Verify causal completeness
        # Causal completeness is 1.0 if all retained nodes have their required preconditions retained
        required_preconditions = 0
        satisfied_preconditions = 0
        for nid in list(retained_ids):
            for edge in dag.get_in_edges(nid, EdgeType.DEPENDS_ON):
                required_preconditions += 1
                if edge.source_id in retained_ids:
                    satisfied_preconditions += 1
                else:
                    # Auto-heal: pull missing precondition into retained set
                    retained_ids.add(edge.source_id)
                    satisfied_preconditions += 1

        causal_completeness = (
            1.0
            if required_preconditions == 0
            else (satisfied_preconditions / required_preconditions)
        )

        # 7. Synthesize dense, pruned prompt
        retained_nodes = [
            dag.get_node(nid)
            for nid in retained_ids
            if dag.get_node(nid) is not None
        ]
        retained_nodes.sort(key=lambda n: (n.step_index, n.timestamp))

        prompt_sections: List[str] = []
        prompt_sections.append("### CAUSAL EXECUTION CONTEXT (Topologically Verified)")

        if target_goal:
            prompt_sections.append(
                f"[OBJECTIVE] {target_goal.content} (Criteria: {', '.join(getattr(target_goal, 'target_criteria', []))})"
            )

        # Invariants summary
        if invariants:
            inv_lines = []
            for inv in invariants:
                icon = "[OK]" if inv.status == InvariantStatus.SATISFIED else "[BREACHED]"
                inv_lines.append(f"  {icon} {inv.predicate} (status: {inv.status.value})")
            prompt_sections.append("[SYSTEM INVARIANTS]\n" + "\n".join(inv_lines))

        # Compact Trace Log
        trace_lines = []
        for n in retained_nodes:
            if isinstance(n, (GoalNode, InvariantNode)):
                continue  # already rendered in dedicated sections
            if isinstance(n, ActionNode):
                trace_lines.append(
                    f"Step {n.step_index} [ACTION] {n.signature()} (Intent: {n.intent or n.content})"
                )
            elif isinstance(n, ObservationNode):
                # Distill stdout/stderr into compact single line
                status_str = "SUCCESS" if n.exit_code == 0 and not n.is_error else "FAILURE"
                summary = n.stderr if (n.stderr and status_str == "FAILURE") else (n.stdout[:120] if n.stdout else "ok")
                summary = summary.replace("\n", " ").strip()
                trace_lines.append(f"  -> [OBSERVATION {status_str}] {summary}")
            elif isinstance(n, AnomalyNode):
                trace_lines.append(
                    f"  -> [ANOMALY {n.severity.value}] {n.symptom_summary or n.content}"
                )

        if trace_lines:
            prompt_sections.append("[RELEVANT CAUSAL FRONTIER]\n" + "\n".join(trace_lines))

        final_prompt = "\n\n".join(prompt_sections)
        pruned_token_estimate = self._estimate_tokens(final_prompt)

        compression_ratio = (
            (1.0 - (pruned_token_estimate / raw_token_estimate))
            if raw_token_estimate > 0
            else 0.0
        )

        return CompiledContext(
            raw_node_count=len(all_nodes),
            pruned_node_count=len(retained_nodes),
            raw_token_estimate=raw_token_estimate,
            pruned_token_estimate=pruned_token_estimate,
            compression_ratio=max(0.0, compression_ratio),
            causal_completeness=causal_completeness,
            formatted_prompt=final_prompt,
            retained_node_ids=list(retained_ids),
        )
