"""
ExecutionEnvironment: State-tracked sandbox environment with dynamic invariant validation.
Calculates deterministic state hashes and registers causal mutations.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Callable, Dict, List, Optional, Tuple
from causal_trace.graph.nodes import (
    ActionNode,
    AnomalyNode,
    AnomalySeverity,
    InvariantNode,
    InvariantStatus,
    ObservationNode,
)


@dataclass
class InvariantRule:
    name: str
    description: str
    validator: Callable[[ExecutionEnvironment], bool]
    expected: Any = True


class ExecutionEnvironment:
    def __init__(self, initial_state: Optional[Dict[str, Any]] = None):
        self.state: Dict[str, Any] = initial_state or {}
        self.env_vars: Dict[str, str] = {}
        self.files: Dict[str, str] = {}
        self.services: Dict[str, bool] = {}
        self.invariant_rules: Dict[str, InvariantRule] = {}
        self.step_counter: int = 0
        self.history: List[Dict[str, Any]] = []

    def register_invariant_rule(
        self,
        name: str,
        description: str,
        validator: Callable[[ExecutionEnvironment], bool],
    ):
        self.invariant_rules[name] = InvariantRule(
            name=name, description=description, validator=validator
        )

    def compute_state_hash(self) -> str:
        """Computes deterministic hash of system state for isomorphism checking."""
        representation = {
            "state": self.state,
            "env_vars": self.env_vars,
            "files": {k: hashlib.md5(v.encode()).hexdigest() for k, v in sorted(self.files.items())},
            "services": dict(sorted(self.services.items())),
        }
        serialized = json.dumps(representation, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]

    def check_invariants(self) -> List[InvariantNode]:
        """Evaluates all registered invariant rules against the current environment."""
        nodes: List[InvariantNode] = []
        for name, rule in self.invariant_rules.items():
            try:
                is_valid = rule.validator(self)
                status = InvariantStatus.SATISFIED if is_valid else InvariantStatus.VIOLATED
                actual = is_valid
            except Exception as ex:
                status = InvariantStatus.VIOLATED
                actual = str(ex)

            node = InvariantNode(
                step_index=self.step_counter,
                content=f"Invariant check: {name} ({status.value})",
                predicate=name,
                status=status,
                expected_value=rule.expected,
                actual_value=actual,
                system_resource=name,
            )
            nodes.append(node)
        return nodes

    def execute_command(
        self, action: ActionNode
    ) -> Tuple[ObservationNode, Optional[AnomalyNode], List[InvariantNode]]:
        """Simulates action execution, evaluates side-effects, and checks invariants."""
        self.step_counter += 1
        cmd = action.command_str.strip()
        stdout = ""
        stderr = ""
        exit_code = 0
        anomaly = None

        # Check for simulated mock behaviors based on command string
        if "rm" in cmd or "delete" in cmd or "pkill" in cmd:
            target = cmd.split()[-1]
            if target in self.files:
                del self.files[target]
                stdout = f"Removed file: {target}"
            elif target in self.services:
                self.services[target] = False
                stdout = f"Terminated service: {target}"
            else:
                stdout = f"Killed target: {target}"

        elif "touch" in cmd or "create" in cmd or "echo" in cmd:
            parts = cmd.split()
            fname = parts[-1] if len(parts) > 1 else "output.txt"
            self.files[fname] = "content"
            stdout = f"Created/Updated file: {fname}"

        elif "chmod" in cmd:
            self.state["permissions"] = "executable"
            stdout = "Permissions updated to executable."

        elif "export" in cmd or "set" in cmd:
            if "=" in cmd:
                var_def = cmd.replace("export", "").replace("set", "").strip()
                k, v = var_def.split("=", 1)
                self.env_vars[k.strip()] = v.strip()
                stdout = f"Environment variable {k.strip()} set."

        elif "systemctl start" in cmd or "service start" in cmd:
            svc = cmd.split()[-1]
            self.services[svc] = True
            stdout = f"Service '{svc}' started and listening."

        elif "test" in cmd or "verify" in cmd or "run" in cmd:
            # Check if any dependencies/invariants fail
            failed_svcs = [s for s, running in self.services.items() if not running]
            if failed_svcs:
                exit_code = 1
                stderr = f"ConnectionRefused: Service '{failed_svcs[0]}' is not running or socket unreachable."
            elif self.state.get("permissions") == "denied":
                exit_code = 126
                stderr = "PermissionDenied: Execution flag missing on target binary."
            elif "DB_URL" not in self.env_vars and "test_db" in cmd:
                exit_code = 2
                stderr = "ConfigurationError: DB_URL environment variable is missing."
            else:
                stdout = "Execution succeeded. All tests passed."

        else:
            stdout = f"Executed: {cmd}"

        obs = ObservationNode(
            step_index=self.step_counter,
            content=f"Observation for Step {self.step_counter}",
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            duration_ms=45.0,
            is_error=(exit_code != 0),
            state_hash=self.compute_state_hash(),
        )

        # Invariant verification
        invariant_nodes = self.check_invariants()
        violated = [inv for inv in invariant_nodes if inv.status == InvariantStatus.VIOLATED]

        if exit_code != 0 or violated:
            error_msg = stderr or (
                f"Invariant violation: {', '.join(v.predicate for v in violated)}"
            )
            anomaly = AnomalyNode(
                step_index=self.step_counter,
                content=f"Failure at Step {self.step_counter}: {error_msg}",
                error_type="ExecutionFailure" if exit_code != 0 else "InvariantBreach",
                stack_trace=stderr,
                severity=AnomalySeverity.HIGH if exit_code != 0 else AnomalySeverity.MEDIUM,
                symptom_summary=error_msg,
                state_hash=self.compute_state_hash(),
            )

        return obs, anomaly, invariant_nodes
