"""
CausalDAG Engine: Directed Acyclic Causal Graph.
Provides high-performance topological, dependency, and ancestor analysis.
"""

from __future__ import annotations
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set
from causal_trace.graph.nodes import (
    BaseNode,
    NodeType,
    InvariantNode,
    AnomalyNode,
    InvariantStatus,
)
from causal_trace.graph.edges import CausalEdge, EdgeType


class CausalDAG:
    def __init__(self, name: str = "ExecutionTraceDAG"):
        self.name = name
        self.nodes: Dict[str, BaseNode] = {}
        self.edges: List[CausalEdge] = []
        self._out_edges: Dict[str, List[CausalEdge]] = defaultdict(list)
        self._in_edges: Dict[str, List[CausalEdge]] = defaultdict(list)

    def add_node(self, node: BaseNode) -> BaseNode:
        if node.id in self.nodes:
            raise ValueError(f"Node with id {node.id} already exists in DAG.")
        self.nodes[node.id] = node
        return node

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CausalEdge:
        if source_id not in self.nodes:
            raise KeyError(f"Source node {source_id} does not exist in DAG.")
        if target_id not in self.nodes:
            raise KeyError(f"Target node {target_id} does not exist in DAG.")

        edge = CausalEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            weight=weight,
            metadata=metadata or {},
        )
        self.edges.append(edge)
        self._out_edges[source_id].append(edge)
        self._in_edges[target_id].append(edge)
        return edge

    def get_node(self, node_id: str) -> Optional[BaseNode]:
        return self.nodes.get(node_id)

    def get_out_edges(
        self, node_id: str, edge_type: Optional[EdgeType] = None
    ) -> List[CausalEdge]:
        edges = self._out_edges.get(node_id, [])
        if edge_type is None:
            return list(edges)
        return [e for e in edges if e.edge_type == edge_type]

    def get_in_edges(
        self, node_id: str, edge_type: Optional[EdgeType] = None
    ) -> List[CausalEdge]:
        edges = self._in_edges.get(node_id, [])
        if edge_type is None:
            return list(edges)
        return [e for e in edges if e.edge_type == edge_type]

    def get_parents(
        self, node_id: str, edge_type: Optional[EdgeType] = None
    ) -> List[BaseNode]:
        edges = self.get_in_edges(node_id, edge_type)
        return [self.nodes[e.source_id] for e in edges]

    def get_children(
        self, node_id: str, edge_type: Optional[EdgeType] = None
    ) -> List[BaseNode]:
        edges = self.get_out_edges(node_id, edge_type)
        return [self.nodes[e.target_id] for e in edges]

    def is_acyclic(self) -> bool:
        """Kahn's algorithm cycle check."""
        in_degree = {nid: len(self._in_edges[nid]) for nid in self.nodes}
        queue = deque([nid for nid, deg in in_degree.items() if deg == 0])
        visited_count = 0

        while queue:
            curr = queue.popleft()
            visited_count += 1
            for edge in self._out_edges[curr]:
                in_degree[edge.target_id] -= 1
                if in_degree[edge.target_id] == 0:
                    queue.append(edge.target_id)

        return visited_count == len(self.nodes)

    def topological_sort(self) -> List[BaseNode]:
        """Returns topological ordering of nodes."""
        in_degree = {nid: len(self._in_edges[nid]) for nid in self.nodes}
        queue = deque([nid for nid, deg in in_degree.items() if deg == 0])
        order = []

        while queue:
            curr = queue.popleft()
            order.append(self.nodes[curr])
            for edge in self._out_edges[curr]:
                in_degree[edge.target_id] -= 1
                if in_degree[edge.target_id] == 0:
                    queue.append(edge.target_id)

        if len(order) != len(self.nodes):
            # Fallback if cyclical: sort by step_index and timestamp
            return sorted(self.nodes.values(), key=lambda n: (n.step_index, n.timestamp))
        return order

    def ancestors(
        self, node_id: str, allowed_edge_types: Optional[Set[EdgeType]] = None
    ) -> Set[str]:
        """Extracts all upstream ancestors of a node."""
        visited: Set[str] = set()
        queue = deque([node_id])

        while queue:
            curr = queue.popleft()
            for edge in self._in_edges.get(curr, []):
                if allowed_edge_types and edge.edge_type not in allowed_edge_types:
                    continue
                parent_id = edge.source_id
                if parent_id not in visited and parent_id != node_id:
                    visited.add(parent_id)
                    queue.append(parent_id)
        return visited

    def descendants(
        self, node_id: str, allowed_edge_types: Optional[Set[EdgeType]] = None
    ) -> Set[str]:
        """Extracts all downstream descendants of a node."""
        visited: Set[str] = set()
        queue = deque([node_id])

        while queue:
            curr = queue.popleft()
            for edge in self._out_edges.get(curr, []):
                if allowed_edge_types and edge.edge_type not in allowed_edge_types:
                    continue
                child_id = edge.target_id
                if child_id not in visited and child_id != node_id:
                    visited.add(child_id)
                    queue.append(child_id)
        return visited

    def find_paths(self, start_id: str, end_id: str) -> List[List[str]]:
        """Finds all directed causal paths between start and end."""
        results: List[List[str]] = []

        def dfs(curr: str, path: List[str]):
            if curr == end_id:
                results.append(list(path))
                return
            for edge in self._out_edges.get(curr, []):
                nxt = edge.target_id
                if nxt not in path:
                    path.append(nxt)
                    dfs(nxt, path)
                    path.pop()

        if start_id in self.nodes and end_id in self.nodes:
            dfs(start_id, [start_id])
        return results

    def find_active_invariants(self) -> List[InvariantNode]:
        return [
            n for n in self.nodes.values()
            if isinstance(n, InvariantNode) and n.status == InvariantStatus.SATISFIED
        ]

    def find_violated_invariants(self) -> List[InvariantNode]:
        return [
            n for n in self.nodes.values()
            if isinstance(n, InvariantNode) and n.status == InvariantStatus.VIOLATED
        ]

    def find_unresolved_anomalies(self) -> List[AnomalyNode]:
        return [
            n for n in self.nodes.values()
            if isinstance(n, AnomalyNode) and not n.resolved
        ]

    def export_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
            "stats": {
                "total_nodes": len(self.nodes),
                "total_edges": len(self.edges),
                "anomalies": len([n for n in self.nodes.values() if isinstance(n, AnomalyNode)]),
                "is_acyclic": self.is_acyclic(),
            }
        }
