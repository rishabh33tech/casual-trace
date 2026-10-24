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


def test_dag_initialization_and_acyclic():
    dag = CausalDAG()
    g = dag.add_node(GoalNode(content="Master Goal"))
    a1 = dag.add_node(ActionNode(command_str="build", step_index=1))
    a2 = dag.add_node(ActionNode(command_str="test", step_index=2))

    dag.add_edge(g.id, a1.id, EdgeType.ENABLES)
    dag.add_edge(a1.id, a2.id, EdgeType.ENABLES)

    assert dag.is_acyclic()
    order = dag.topological_sort()
    assert len(order) == 3
    assert order[0].id == g.id
    assert order[1].id == a1.id
    assert order[2].id == a2.id


def test_dag_ancestor_and_descendant_traversal():
    dag = CausalDAG()
    n1 = dag.add_node(ActionNode(command_str="step1", step_index=1))
    n2 = dag.add_node(ActionNode(command_str="step2", step_index=2))
    n3 = dag.add_node(ActionNode(command_str="step3", step_index=3))
    n4 = dag.add_node(ActionNode(command_str="step4", step_index=4))

    dag.add_edge(n1.id, n2.id, EdgeType.ENABLES)
    dag.add_edge(n2.id, n3.id, EdgeType.ENABLES)
    dag.add_edge(n3.id, n4.id, EdgeType.ENABLES)

    ancestors = dag.ancestors(n4.id)
    assert ancestors == {n1.id, n2.id, n3.id}

    descendants = dag.descendants(n1.id)
    assert descendants == {n2.id, n3.id, n4.id}


def test_dag_invariant_querying():
    dag = CausalDAG()
    inv_ok = dag.add_node(
        InvariantNode(predicate="PortOpen", status=InvariantStatus.SATISFIED)
    )
    inv_bad = dag.add_node(
        InvariantNode(predicate="AuthValid", status=InvariantStatus.VIOLATED)
    )

    assert len(dag.find_active_invariants()) == 1
    assert dag.find_active_invariants()[0].id == inv_ok.id

    assert len(dag.find_violated_invariants()) == 1
    assert dag.find_violated_invariants()[0].id == inv_bad.id


def test_dag_path_finding():
    dag = CausalDAG()
    a = dag.add_node(ActionNode(command_str="A"))
    b = dag.add_node(ActionNode(command_str="B"))
    c = dag.add_node(ActionNode(command_str="C"))
    d = dag.add_node(ActionNode(command_str="D"))

    dag.add_edge(a.id, b.id, EdgeType.ENABLES)
    dag.add_edge(b.id, d.id, EdgeType.ENABLES)
    dag.add_edge(a.id, c.id, EdgeType.ENABLES)
    dag.add_edge(c.id, d.id, EdgeType.ENABLES)

    paths = dag.find_paths(a.id, d.id)
    assert len(paths) == 2
    assert [a.id, b.id, d.id] in paths
    assert [a.id, c.id, d.id] in paths
