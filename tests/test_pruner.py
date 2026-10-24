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
from causal_trace.engine.pruner import CausalContextCompiler


def test_topological_context_compiler():
    dag = CausalDAG()
    compiler = CausalContextCompiler()

    g = dag.add_node(GoalNode(content="Deploy Service", target_criteria=["Live"]))
    a1 = dag.add_node(ActionNode(command_str="build binary", step_index=1))
    dag.add_edge(g.id, a1.id, EdgeType.ENABLES)

    # Add 15 noisy exploratory intermediate steps with large outputs
    for i in range(2, 17):
        act = dag.add_node(
            ActionNode(command_str=f"debug_probe --flag={i}", step_index=i)
        )
        obs = dag.add_node(
            ObservationNode(
                step_index=i,
                stdout="LOG LINE " * 100,  # noisy log buffer
                stderr="",
            )
        )
        dag.add_edge(act.id, obs.id, EdgeType.OBSERVES)

    inv = dag.add_node(
        InvariantNode(predicate="BinaryIntegrity", status=InvariantStatus.SATISFIED)
    )

    compiled = compiler.compile(dag, target_goal_id=g.id)

    assert compiled.causal_completeness == 1.0
    assert compiled.compression_ratio > 0.50
    assert g.id in compiled.retained_node_ids
    assert inv.id in compiled.retained_node_ids
