import pytest
from causal_trace.graph.dag import CausalDAG
from causal_trace.graph.nodes import ActionNode, ObservationNode, AnomalyNode
from causal_trace.engine.thrashing import ThrashingDetector


def test_shannon_entropy_calculation():
    detector = ThrashingDetector()
    # High entropy: all distinct
    diverse = ["cmd_a", "cmd_b", "cmd_c", "cmd_d"]
    h_diverse = detector.calculate_shannon_entropy(diverse)
    assert h_diverse == 2.0  # log2(4)

    # Collapsed entropy: identical
    collapsed = ["cmd_a", "cmd_a", "cmd_a", "cmd_a"]
    h_collapsed = detector.calculate_shannon_entropy(collapsed)
    assert h_collapsed == 0.0


def test_periodic_cycle_detection_period_1():
    detector = ThrashingDetector()
    repeats = ["cmd_x", "cmd_x", "cmd_x"]
    cycle = detector.detect_periodic_cycles(repeats)
    assert cycle is not None
    period, pattern = cycle
    assert period == 1
    assert pattern == ["cmd_x"]


def test_periodic_cycle_detection_period_2():
    detector = ThrashingDetector()
    alternating = ["cmd_a", "cmd_b", "cmd_a", "cmd_b"]
    cycle = detector.detect_periodic_cycles(alternating)
    assert cycle is not None
    period, pattern = cycle
    assert period == 2
    assert pattern == ["cmd_a", "cmd_b"]


def test_thrashing_evaluation_on_dag():
    dag = CausalDAG()
    detector = ThrashingDetector(repetition_threshold=3)

    # Add repeated failed actions
    for i in range(1, 4):
        act = dag.add_node(
            ActionNode(tool_name="shell", command_str="curl localhost:8080", step_index=i)
        )
        obs = dag.add_node(
            ObservationNode(step_index=i, exit_code=7, is_error=True, stderr="Connection refused")
        )
        dag.add_node(AnomalyNode(step_index=i, symptom_summary="Refused"))

    alert = detector.evaluate(dag)
    assert alert.is_thrashing is True
    assert alert.cycle_period == 1
    assert "curl localhost:8080" in alert.culprit_action_signatures[0]
