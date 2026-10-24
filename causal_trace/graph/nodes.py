"""
CausalTrace Node Definitions.
Represents structural nodes in the Agent Execution Causal DAG.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import time
import uuid


class NodeType(str, Enum):
    GOAL = "GOAL"
    HYPOTHESIS = "HYPOTHESIS"
    ACTION = "ACTION"
    OBSERVATION = "OBSERVATION"
    INVARIANT = "INVARIANT"
    ANOMALY = "ANOMALY"


class InvariantStatus(str, Enum):
    SATISFIED = "SATISFIED"
    VIOLATED = "VIOLATED"
    UNKNOWN = "UNKNOWN"


class AnomalySeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class BaseNode:
    id: str = field(default_factory=lambda: f"node_{uuid.uuid4().hex[:8]}")
    node_type: NodeType = NodeType.ACTION
    content: str = ""
    step_index: int = 0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    state_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.node_type.value,
            "content": self.content,
            "step_index": self.step_index,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "state_hash": self.state_hash,
        }


@dataclass
class GoalNode(BaseNode):
    node_type: NodeType = NodeType.GOAL
    target_criteria: List[str] = field(default_factory=list)
    parent_goal_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["target_criteria"] = self.target_criteria
        d["parent_goal_id"] = self.parent_goal_id
        return d


@dataclass
class HypothesisNode(BaseNode):
    node_type: NodeType = NodeType.HYPOTHESIS
    confidence: float = 1.0
    rationale: str = ""
    superseded_by: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["confidence"] = self.confidence
        d["rationale"] = self.rationale
        d["superseded_by"] = self.superseded_by
        return d


@dataclass
class ActionNode(BaseNode):
    node_type: NodeType = NodeType.ACTION
    tool_name: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    command_str: str = ""
    intent: str = ""
    cost_tokens: int = 0

    def signature(self) -> str:
        """Deterministic fingerprint of action semantics."""
        return f"{self.tool_name}:{self.command_str or sorted(self.parameters.items())}"

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["tool_name"] = self.tool_name
        d["parameters"] = self.parameters
        d["command_str"] = self.command_str
        d["intent"] = self.intent
        d["cost_tokens"] = self.cost_tokens
        return d


@dataclass
class ObservationNode(BaseNode):
    node_type: NodeType = NodeType.OBSERVATION
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    duration_ms: float = 0.0
    is_error: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["exit_code"] = self.exit_code
        d["stdout"] = self.stdout
        d["stderr"] = self.stderr
        d["duration_ms"] = self.duration_ms
        d["is_error"] = self.is_error
        return d


@dataclass
class InvariantNode(BaseNode):
    node_type: NodeType = NodeType.INVARIANT
    predicate: str = ""
    status: InvariantStatus = InvariantStatus.UNKNOWN
    expected_value: Any = None
    actual_value: Any = None
    system_resource: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["predicate"] = self.predicate
        d["status"] = self.status.value
        d["expected_value"] = self.expected_value
        d["actual_value"] = self.actual_value
        d["system_resource"] = self.system_resource
        return d


@dataclass
class AnomalyNode(BaseNode):
    node_type: NodeType = NodeType.ANOMALY
    error_type: str = ""
    stack_trace: str = ""
    severity: AnomalySeverity = AnomalySeverity.MEDIUM
    resolved: bool = False
    symptom_summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["error_type"] = self.error_type
        d["stack_trace"] = self.stack_trace
        d["severity"] = self.severity.value
        d["resolved"] = self.resolved
        d["symptom_summary"] = self.symptom_summary
        return d
