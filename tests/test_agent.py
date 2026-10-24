import pytest
from causal_trace.runtime.environment import ExecutionEnvironment
from causal_trace.runtime.agent import BaselineReActAgent, CausalAgent
from causal_trace.runtime.session import SessionRunner


def setup_test_env():
    env = ExecutionEnvironment()
    env.services["postgres"] = True

    def check_postgres(e: ExecutionEnvironment) -> bool:
        return e.services.get("postgres", False)

    env.register_invariant_rule(
        name="PostgresService",
        description="Database must be online",
        validator=check_postgres,
    )
    return env


def test_baseline_react_agent_fails_and_loops():
    env = setup_test_env()
    agent = BaselineReActAgent(name="Baseline", goal="Execute test suite")
    runner = SessionRunner(agent, env, max_steps=5)
    metrics = runner.run()

    # Baseline fails because it does not understand why tests failed after pkill
    assert not metrics.success
    assert metrics.total_steps == 5
    assert metrics.anomalies_count > 0


def test_causal_agent_self_heals_and_succeeds():
    env = setup_test_env()
    agent = CausalAgent(name="CausalTraceAgent", goal="Execute test suite")
    runner = SessionRunner(agent, env, max_steps=8)
    metrics = runner.run()

    # CausalAgent diagnoses pkill postgres, restarts it, and succeeds!
    assert metrics.success is True
    assert metrics.loops_intercepted >= 0
    assert metrics.total_tokens < 3000
