import pytest
from causal_trace.graph.dag import CausalDAG
from causal_trace.graph.nodes import (
    ActionNode,
    AnomalyNode,
    GoalNode,
    InvariantNode,
    InvariantStatus,
    ObservationNode,
)
from causal_trace.graph.edges import EdgeType
from causal_trace.engine.attribution import CounterfactualEngine


def test_root_cause_attribution_pinpoints_fib():
    dag = CausalDAG()
    engine = CounterfactualEngine()

    # Step 1: Initialize
    g = dag.add_node(GoalNode(content="Run Pipeline"))
    a1 = dag.add_node(ActionNode(command_str="setup", step_index=1))
    dag.add_edge(g.id, a1.id, EdgeType.ENABLES)

    # Step 2: Culprit action invalidates database invariant
    a2 = dag.add_node(ActionNode(command_str="pkill postgres", step_index=2))
    inv = dag.add_node(
        InvariantNode(
            predicate="PostgresRunning",
            status=InvariantStatus.VIOLATED,
            step_index=2,
        )
    )
    dag.add_edge(a1.id, a2.id, EdgeType.ENABLES)
    dag.add_edge(a2.id, inv.id, EdgeType.INVALIDATES)

    # Step 3: Irrelevant action
    a3 = dag.add_node(ActionNode(command_str="touch /tmp/marker", step_index=3))
    dag.add_edge(a2.id, a3.id, EdgeType.ENABLES)

    # Step 4: Symptom action that crashes
    a4 = dag.add_node(ActionNode(command_str="run integration_tests", step_index=4))
    obs4 = dag.add_node(
        ObservationNode(
            step_index=4,
            exit_code=1,
            is_error=True,
            stderr="ConnectionRefused: port 5432 closed",
        )
    )
    anom4 = dag.add_node(
        AnomalyNode(
            step_index=4,
            symptom_summary="ConnectionRefused: port 5432 closed",
        )
    )
    dag.add_edge(a3.id, a4.id, EdgeType.ENABLES)
    dag.add_edge(a4.id, obs4.id, EdgeType.OBSERVES)
    dag.add_edge(obs4.id, anom4.id, EdgeType.TRIGGERS)

    # Diagnose anomaly at Step 4
    report = engine.diagnose_anomaly(dag, anom4.id)

    assert report is not None
    assert report.primary_culprit_id == a2.id
    assert "pkill postgres" in report.primary_culprit_signature
    assert report.broken_invariant == "PostgresRunning"
    assert report.confidence > 0.8
    assert "restore" in report.remediation_directive.lower()
