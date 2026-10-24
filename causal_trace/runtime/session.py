"""
SessionRunner: Orchestrates end-to-end agent trial runs and collects comparative telemetry.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional
from causal_trace.graph.nodes import InvariantStatus
from causal_trace.runtime.agent import BaseAgent, CausalAgent
from causal_trace.runtime.environment import ExecutionEnvironment


@dataclass
class SessionMetrics:
    agent_name: str
    goal: str
    success: bool
    total_steps: int
    total_tokens: int
    anomalies_count: int
    loops_intercepted: int
    elapsed_time_ms: float
    context_compression_ratio: float = 0.0
    step_history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "goal": self.goal,
            "success": self.success,
            "total_steps": self.total_steps,
            "total_tokens": self.total_tokens,
            "anomalies_count": self.anomalies_count,
            "loops_intercepted": self.loops_intercepted,
            "elapsed_time_ms": round(self.elapsed_time_ms, 2),
            "context_compression_ratio": round(self.context_compression_ratio, 3),
            "step_history": self.step_history,
        }


class SessionRunner:
    def __init__(self, agent: BaseAgent, env: ExecutionEnvironment, max_steps: int = 12):
        self.agent = agent
        self.env = env
        self.max_steps = max_steps

    def run(self) -> SessionMetrics:
        start_time = time.time()
        step_logs: List[Dict[str, Any]] = []

        for step in range(1, self.max_steps + 1):
            if self.agent.is_finished:
                break

            action = self.agent.step(step)
            obs, anomaly, invariants = self.env.execute_command(action)
            self.agent.observe(action, obs, anomaly, invariants)

            step_record = {
                "step": step,
                "action": action.command_str,
                "intent": action.intent,
                "exit_code": obs.exit_code,
                "stdout": obs.stdout,
                "stderr": obs.stderr,
                "has_anomaly": anomaly is not None,
                "invariants_violated": [
                    inv.predicate for inv in invariants if inv.status == InvariantStatus.VIOLATED
                ],
            }
            step_logs.append(step_record)

        elapsed = (time.time() - start_time) * 1000.0

        compression = 0.0
        loops_intercepted = 0
        if isinstance(self.agent, CausalAgent):
            loops_intercepted = self.agent.intercept_count
            if self.agent.latest_compiled_context:
                compression = self.agent.latest_compiled_context.compression_ratio

        return SessionMetrics(
            agent_name=self.agent.name,
            goal=self.agent.goal,
            success=self.agent.is_successful,
            total_steps=len(step_logs),
            total_tokens=self.agent.total_tokens_used,
            anomalies_count=len(self.agent.anomalies),
            loops_intercepted=loops_intercepted,
            elapsed_time_ms=elapsed,
            context_compression_ratio=compression,
            step_history=step_logs,
        )
