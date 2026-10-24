"""
Agent implementations: BaselineReActAgent vs CausalAgent.
Demonstrates empirical behavior under latent invariant breaches, loops, and context stress.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
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
from causal_trace.graph.edges import EdgeType
from causal_trace.engine.thrashing import ThrashingDetector, ThrashingAlert
from causal_trace.engine.attribution import CounterfactualEngine, RootCauseReport
from causal_trace.engine.pruner import CausalContextCompiler, CompiledContext
from causal_trace.engine.guardrail import GuardrailEngine, GuardrailVerdict


class BaseAgent(ABC):
    def __init__(self, name: str, goal: str):
        self.name = name
        self.goal = goal
        self.total_tokens_used: int = 0
        self.actions_taken: List[ActionNode] = []
        self.observations: List[ObservationNode] = []
        self.anomalies: List[AnomalyNode] = []
        self.is_finished: bool = False
        self.is_successful: bool = False

    @abstractmethod
    def step(self, step_index: int) -> ActionNode:
        pass

    @abstractmethod
    def observe(
        self,
        action: ActionNode,
        observation: ObservationNode,
        anomaly: Optional[AnomalyNode],
        invariants: List[InvariantNode],
    ):
        pass


class BaselineReActAgent(BaseAgent):
    """
    Standard naive ReAct / AutoGPT agent.
    Suffers from context amnesia, semantic loop thrashing, and late-stage misattribution.
    """

    def __init__(self, name: str, goal: str, max_retries: int = 8):
        super().__init__(name, goal)
        self.max_retries = max_retries
        self.retry_count = 0
        self.history: List[str] = [f"Goal: {goal}"]

    def step(self, step_index: int) -> ActionNode:
        # Standard ReAct appends raw linear history to context
        context_str = "\n".join(self.history)
        self.total_tokens_used += max(50, int(len(context_str) / 4.0))

        # If previous step failed, naive agent retries with superficial flag tweaks
        last_obs = self.observations[-1] if self.observations else None
        if last_obs and last_obs.is_error:
            self.retry_count += 1
            variants = [
                "run test --retry",
                "run test --verbose",
                "run test --timeout=30",
                "run test --force",
                "run test --ignore-warnings",
                "run test --clean",
            ]
            cmd = variants[(self.retry_count - 1) % len(variants)]
            intent = f"Retrying test suite with variant flag (attempt {self.retry_count})"
        else:
            if step_index == 1:
                cmd = "setup environment"
                intent = "Initialize setup"
            elif step_index == 2:
                # Naive agent runs an action that inadvertently breaks an invariant
                cmd = "cleanup pkill postgres"
                intent = "Free up memory by terminating processes"
            elif step_index == 3:
                cmd = "touch /tmp/ready"
                intent = "Mark temporary files"
            else:
                cmd = "run test"
                intent = "Run integration test verification"

        action = ActionNode(
            step_index=step_index,
            tool_name="shell",
            command_str=cmd,
            intent=intent,
            content=f"Execute: {cmd}",
            cost_tokens=max(50, int(len(context_str) / 4.0)),
        )
        self.actions_taken.append(action)
        return action

    def observe(
        self,
        action: ActionNode,
        observation: ObservationNode,
        anomaly: Optional[AnomalyNode],
        invariants: List[InvariantNode],
    ):
        self.observations.append(observation)
        self.history.append(f"Action: {action.command_str}")
        self.history.append(f"Output: {observation.stdout} {observation.stderr}")

        if anomaly:
            self.anomalies.append(anomaly)
            self.history.append(f"Error: {anomaly.symptom_summary}")
        elif observation.exit_code == 0 and "test" in action.command_str:
            self.is_successful = True
            self.is_finished = True


class CausalAgent(BaseAgent):
    """
    Neuro-symbolic Causal Agent powered by CausalTrace Kernel.
    Uses CausalDAG, ThrashingDetector, CounterfactualEngine, ContextPruner, and GuardrailEngine.
    """

    def __init__(self, name: str, goal: str):
        super().__init__(name, goal)
        self.dag = CausalDAG(name=f"{name}_DAG")
        self.thrashing_detector = ThrashingDetector()
        self.attribution_engine = CounterfactualEngine()
        self.context_compiler = CausalContextCompiler()
        self.guardrail = GuardrailEngine()

        # Seed DAG with initial GoalNode
        self.root_goal = GoalNode(
            step_index=0,
            content=goal,
            target_criteria=["All tests passing", "All invariants satisfied"],
        )
        self.dag.add_node(self.root_goal)

        # Active diagnostic state
        self.active_diagnosis: Optional[RootCauseReport] = None
        self.latest_compiled_context: Optional[CompiledContext] = None
        self.intercept_count: int = 0

    def step(self, step_index: int) -> ActionNode:
        # 1. Compile minimal topologically pruned context
        self.latest_compiled_context = self.context_compiler.compile(
            self.dag, target_goal_id=self.root_goal.id
        )
        self.total_tokens_used += self.latest_compiled_context.pruned_token_estimate

        # 2. Check for thrashing / loop traps
        thrash_alert: ThrashingAlert = self.thrashing_detector.evaluate(self.dag)
        if thrash_alert.is_thrashing:
            for sig in thrash_alert.culprit_action_signatures:
                self.guardrail.ban_action_pattern(
                    sig, reason="Detected deterministic loop thrashing"
                )

        # 3. Formulate next action based on causal invariants
        proposed_action: Optional[ActionNode] = None

        # Check if there is an active root-cause diagnosis requiring remediation
        if self.active_diagnosis and self.active_diagnosis.broken_invariant:
            inv_name = self.active_diagnosis.broken_invariant
            # Synthesize remediation action
            if "postgres" in inv_name.lower() or "service" in inv_name.lower():
                cmd = "systemctl start postgres"
                intent = f"Remediate root cause: restart {inv_name}"
            elif "permission" in inv_name.lower():
                cmd = "chmod +x /bin/runner"
                intent = "Remediate root cause: fix executable permissions"
            elif "env" in inv_name.lower() or "db_url" in inv_name.lower():
                cmd = "export DB_URL=postgresql://localhost:5432/db"
                intent = "Remediate root cause: restore environment variable"
            else:
                cmd = f"repair {inv_name}"
                intent = f"Remediate {inv_name}"

            proposed_action = ActionNode(
                step_index=step_index,
                tool_name="shell",
                command_str=cmd,
                intent=intent,
                content=f"Remediation action: {cmd}",
            )
        else:
            # Standard planned progression
            if step_index == 1:
                cmd = "setup environment"
                intent = "Initialize setup"
            elif step_index == 2:
                cmd = "cleanup pkill postgres"
                intent = "Free up memory"
            elif step_index == 3:
                cmd = "touch /tmp/ready"
                intent = "Mark temporary files"
            else:
                cmd = "run test"
                intent = "Run integration tests"

            proposed_action = ActionNode(
                step_index=step_index,
                tool_name="shell",
                command_str=cmd,
                intent=intent,
                content=f"Execute: {cmd}",
            )

        # 4. Filter action through GuardrailEngine
        verdict: GuardrailVerdict = self.guardrail.evaluate_action(proposed_action)
        if not verdict.allowed:
            self.intercept_count += 1
            # Action was blocked by guardrail! Pivot to remediation
            if verdict.required_remediation:
                cmd = f"systemctl start {verdict.required_remediation}"
                intent = f"Guardrail-enforced remediation of {verdict.required_remediation}"
            else:
                cmd = "systemctl start postgres"
                intent = "Bypass thrash loop by restoring underlying service"

            proposed_action = ActionNode(
                step_index=step_index,
                tool_name="shell",
                command_str=cmd,
                intent=intent,
                content=f"Self-healed action: {cmd}",
            )

        # Register proposed action in DAG
        self.dag.add_node(proposed_action)

        # Link action to goal
        self.dag.add_edge(
            source_id=self.root_goal.id,
            target_id=proposed_action.id,
            edge_type=EdgeType.ENABLES,
        )

        self.actions_taken.append(proposed_action)
        return proposed_action

    def observe(
        self,
        action: ActionNode,
        observation: ObservationNode,
        anomaly: Optional[AnomalyNode],
        invariants: List[InvariantNode],
    ):
        self.observations.append(observation)

        # Register observation in DAG
        self.dag.add_node(observation)
        self.dag.add_edge(
            source_id=action.id,
            target_id=observation.id,
            edge_type=EdgeType.OBSERVES,
        )

        # Register invariants in DAG and detect mutations
        for inv in invariants:
            # Check if this invariant already exists in DAG
            existing_invs = [
                n for n in self.dag.nodes.values()
                if isinstance(n, InvariantNode) and n.predicate == inv.predicate
            ]
            if existing_invs:
                prev_inv = existing_invs[-1]
                if prev_inv.status != inv.status:
                    # Invariant status mutated!
                    self.dag.add_node(inv)
                    if inv.status == InvariantStatus.VIOLATED:
                        self.dag.add_edge(
                            source_id=action.id,
                            target_id=inv.id,
                            edge_type=EdgeType.INVALIDATES,
                        )
                    else:
                        self.dag.add_edge(
                            source_id=action.id,
                            target_id=inv.id,
                            edge_type=EdgeType.MITIGATES,
                        )
            else:
                self.dag.add_node(inv)
                if inv.status == InvariantStatus.VIOLATED:
                    self.dag.add_edge(
                        source_id=action.id,
                        target_id=inv.id,
                        edge_type=EdgeType.INVALIDATES,
                    )

        # Register anomaly if any
        if anomaly:
            self.anomalies.append(anomaly)
            self.dag.add_node(anomaly)
            self.dag.add_edge(
                source_id=observation.id,
                target_id=anomaly.id,
                edge_type=EdgeType.TRIGGERS,
            )

            # Trigger Counterfactual Root Cause Engine
            report: Optional[RootCauseReport] = (
                self.attribution_engine.diagnose_anomaly(self.dag, anomaly.id)
            )
            if report:
                self.active_diagnosis = report
                if report.broken_invariant:
                    self.guardrail.register_required_remediation(
                        report.broken_invariant,
                        hint=report.remediation_directive,
                    )

        else:
            # Check if action mitigated active diagnosis
            if self.active_diagnosis and self.active_diagnosis.broken_invariant:
                # Check if broken invariant is now satisfied
                current_invs = self.dag.find_active_invariants()
                if any(
                    inv.predicate == self.active_diagnosis.broken_invariant
                    for inv in current_invs
                ):
                    self.guardrail.clear_remediation(
                        self.active_diagnosis.broken_invariant
                    )
                    self.active_diagnosis = None

            if observation.exit_code == 0 and "test" in action.command_str:
                self.is_successful = True
                self.is_finished = True
