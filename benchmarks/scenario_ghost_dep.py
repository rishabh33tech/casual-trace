"""
Benchmark Scenario 1: The Ghost Dependency Trap.
Tests isolation of upstream invariant destruction (First Invariant Breach)
separated by irrelevant intermediate steps.
"""

from causal_trace.runtime.environment import ExecutionEnvironment
from causal_trace.runtime.agent import BaselineReActAgent, CausalAgent
from causal_trace.runtime.session import SessionRunner


def run_ghost_dependency_benchmark():
    def create_env():
        env = ExecutionEnvironment()
        env.services["postgres"] = True
        env.register_invariant_rule(
            name="PostgresOnline",
            description="Database service must be active",
            validator=lambda e: e.services.get("postgres", False),
        )
        return env

    # Run Baseline
    env_b = create_env()
    agent_b = BaselineReActAgent("Baseline-ReAct", "Run test suite")
    m_baseline = SessionRunner(agent_b, env_b, max_steps=6).run()

    # Run CausalTrace
    env_c = create_env()
    agent_c = CausalAgent("CausalTrace", "Run test suite")
    m_causal = SessionRunner(agent_c, env_c, max_steps=6).run()

    return m_baseline, m_causal
