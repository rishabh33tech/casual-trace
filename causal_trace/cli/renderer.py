"""
AsciiRenderer: Terminal visualization of Causal DAGs, root-cause attribution, and thrashing metrics.
"""

from __future__ import annotations
import sys
from typing import Any, Dict, List, Optional
from causal_trace.graph.dag import CausalDAG
from causal_trace.graph.nodes import (
    ActionNode,
    AnomalyNode,
    GoalNode,
    InvariantNode,
    InvariantStatus,
    NodeType,
    ObservationNode,
)
from causal_trace.engine.attribution import RootCauseReport
from causal_trace.engine.thrashing import ThrashingAlert
from causal_trace.engine.pruner import CompiledContext


# ANSI Colors (gracefully fallback if unsupported)
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


class AsciiRenderer:
    @staticmethod
    def render_header(title: str):
        width = 72
        print(f"\n{BOLD}{CYAN}{'=' * width}{RESET}")
        print(f"{BOLD}{CYAN}  {title.center(width - 4)}  {RESET}")
        print(f"{BOLD}{CYAN}{'=' * width}{RESET}\n")

    @staticmethod
    def render_dag_stream(dag: CausalDAG):
        print(f"{BOLD}### Execution Causal Flow DAG (Topologically Ordered):{RESET}")
        nodes = dag.topological_sort()

        for idx, node in enumerate(nodes):
            indent = "  "
            if isinstance(node, GoalNode):
                prefix = f"{CYAN}[GOAL]{RESET}"
                details = f"{node.content}"
            elif isinstance(node, ActionNode):
                prefix = f"{BOLD}[ACTION Step {node.step_index}]{RESET}"
                details = f"{node.signature()} ({node.intent})"
            elif isinstance(node, ObservationNode):
                status_color = RED if node.is_error or node.exit_code != 0 else GREEN
                status = "FAILED" if node.is_error or node.exit_code != 0 else "OK"
                prefix = f"{status_color}[OBSERVATION {status}]{RESET}"
                details = (node.stderr if status == "FAILED" else (node.stdout[:80] if node.stdout else "Success"))
            elif isinstance(node, InvariantNode):
                color = GREEN if node.status == InvariantStatus.SATISFIED else RED
                prefix = f"{color}[INVARIANT {node.status.value}]{RESET}"
                details = f"{node.predicate}"
            elif isinstance(node, AnomalyNode):
                prefix = f"{BOLD}{RED}[ANOMALY {node.severity.value}]{RESET}"
                details = f"{node.symptom_summary}"
            else:
                prefix = f"[{node.node_type.value}]"
                details = node.content

            # Show outgoing causal edges
            out_edges = dag.get_out_edges(node.id)
            edge_tags = []
            for e in out_edges:
                target_node = dag.get_node(e.target_id)
                target_label = target_node.node_type.value if target_node else e.target_id[:6]
                edge_tags.append(f"--({e.edge_type.value})--> {target_label}")

            edge_str = f" {DIM}{'; '.join(edge_tags)}{RESET}" if edge_tags else ""
            connector = "|--" if idx < len(nodes) - 1 else "+--"
            print(f"{indent}{connector} {prefix} {details}{edge_str}")
        print()

    @staticmethod
    def render_attribution_report(report: RootCauseReport):
        print(f"{BOLD}{MAGENTA}### Root Cause Attribution Report (Counterfactual Engine):{RESET}")
        print(f"  {BOLD}Anomaly:{RESET} {report.symptom_summary}")
        print(f"  {BOLD}Primary Culprit:{RESET} {RED}{report.primary_culprit_signature}{RESET}")
        print(f"  {BOLD}Confidence:{RESET} {GREEN}{report.confidence * 100:.1f}%{RESET}")
        if report.broken_invariant:
            print(f"  {BOLD}Breached Invariant:{RESET} {YELLOW}{report.broken_invariant}{RESET}")
        print(f"  {BOLD}Remediation Directive:{RESET}\n    -> {CYAN}{report.remediation_directive}{RESET}")

        if report.candidates:
            print(f"\n  {BOLD}Top Candidate Predecessors Evaluated:{RESET}")
            for c in report.candidates[:3]:
                bar = "#" * int(c.score * 1.5)
                print(f"    - Step {c.step_index:02d} [{bar:<15}] (CAS: {c.score:.2f}) {c.action_signature} | {c.reason}")
        print()

    @staticmethod
    def render_thrashing_alert(alert: ThrashingAlert):
        if not alert.is_thrashing:
            print(f"  {GREEN}[Thrashing Monitor]{RESET} Normal dynamics (Entropy: {alert.entropy:.2f}). No loops detected.")
            return

        print(f"{BOLD}{RED}### [ALERT] Thrashing & Action-Space Loop Detected!{RESET}")
        print(f"  {BOLD}Pattern Period:{RESET} {alert.cycle_period}")
        print(f"  {BOLD}Shannon Entropy:{RESET} {alert.entropy:.2f}")
        print(f"  {BOLD}Explanation:{RESET} {alert.explanation}")
        print(f"  {BOLD}Action Barrier Applied:{RESET} {', '.join(alert.culprit_action_signatures)}")
        print()

    @staticmethod
    def render_compression_metrics(context: CompiledContext):
        print(f"{BOLD}{CYAN}### Topological Context Compression:{RESET}")
        print(f"  Raw Trace Tokens (Sliding Context):       {context.raw_token_estimate:,} tokens")
        print(f"  Topologically Pruned Context:             {context.pruned_token_estimate:,} tokens")
        print(f"  Token Savings:                            {BOLD}{GREEN}{context.compression_ratio * 100:.1f}% reduction{RESET}")
        print(f"  Causal Completeness Guarantee:            {BOLD}{GREEN}{context.causal_completeness * 100:.1f}%{RESET}")
        print()
