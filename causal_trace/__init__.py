"""
CausalTrace: Neuro-Symbolic Causal State Kernel & Anti-Thrashing Engine for Autonomous AI Agents.
"""

from causal_trace.graph.dag import CausalDAG
from causal_trace.graph.nodes import (
    BaseNode,
    NodeType,
    GoalNode,
    HypothesisNode,
    ActionNode,
    ObservationNode,
    InvariantNode,
    AnomalyNode,
)
from causal_trace.graph.edges import CausalEdge, EdgeType
from causal_trace.engine.thrashing import ThrashingDetector, ThrashingAlert
from causal_trace.engine.attribution import CounterfactualEngine, RootCauseReport
from causal_trace.engine.pruner import CausalContextCompiler, CompiledContext
from causal_trace.engine.guardrail import GuardrailEngine, GuardrailVerdict
from causal_trace.runtime.environment import ExecutionEnvironment
from causal_trace.runtime.agent import BaselineReActAgent, CausalAgent
from causal_trace.runtime.session import SessionRunner

__version__ = "1.0.0"

__all__ = [
    "CausalDAG",
    "BaseNode",
    "NodeType",
    "GoalNode",
    "HypothesisNode",
    "ActionNode",
    "ObservationNode",
    "InvariantNode",
    "AnomalyNode",
    "CausalEdge",
    "EdgeType",
    "ThrashingDetector",
    "ThrashingAlert",
    "CounterfactualEngine",
    "RootCauseReport",
    "CausalContextCompiler",
    "CompiledContext",
    "GuardrailEngine",
    "GuardrailVerdict",
    "ExecutionEnvironment",
    "BaselineReActAgent",
    "CausalAgent",
    "SessionRunner",
]
