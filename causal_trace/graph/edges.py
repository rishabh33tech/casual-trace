"""
CausalTrace Edge Definitions.
Encodes formal causal, temporal, and semantic dependency relationships.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict


class EdgeType(str, Enum):
    # Causal Preconditions
    ENABLES = "ENABLES"           # A creates prerequisites for B
    DEPENDS_ON = "DEPENDS_ON"     # B structurally or logically depends on A
    
    # State & Invariant Violations
    INVALIDATES = "INVALIDATES"   # A violates invariant or mutates state required by B
    
    # Agent Reflex & Teleology
    TRIGGERS = "TRIGGERS"         # An observation or anomaly sparks a hypothesis/action
    MITIGATES = "MITIGATES"       # An action fixes or counteracts an anomaly
    SUPERSEDES = "SUPERSEDES"     # Hypothesis B refutes/replaces Hypothesis A
    OBSERVES = "OBSERVES"         # Action produced this observation


@dataclass
class CausalEdge:
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source_id,
            "target": self.target_id,
            "type": self.edge_type.value,
            "weight": self.weight,
            "metadata": self.metadata,
        }
