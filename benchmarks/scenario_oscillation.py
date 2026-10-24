"""
Benchmark Scenario 2: Oscillating Action Thrashing.
Tests cycle entropy detection and deterministic action space barrier enforcement.
"""

from causal_trace.runtime.environment import ExecutionEnvironment
from causal_trace.runtime.agent import BaselineReActAgent, CausalAgent
from causal_trace.runtime.session import SessionRunner
from causal_trace.graph.nodes import ActionNode, ObservationNode, AnomalyNode
from causal_trace.engine.thrashing import ThrashingDetector


def run_oscillation_benchmark():
    detector = ThrashingDetector()
    signatures = [
        "shell:set_port 8080",
        "shell:set_port 8081",
        "shell:set_port 8080",
        "shell:set_port 8081",
    ]
    entropy = detector.calculate_shannon_entropy(signatures)
    cycle = detector.detect_periodic_cycles(signatures)
    return entropy, cycle
