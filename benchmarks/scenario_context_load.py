"""
Benchmark Scenario 3: Deep Trace Context Compression.
Evaluates token overhead scaling across 30 sequential agent steps.
"""

from causal_trace.graph.dag import CausalDAG
from causal_trace.graph.nodes import (
    ActionNode,
    GoalNode,
    InvariantNode,
    InvariantStatus,
    ObservationNode,
)
from causal_trace.graph.edges import EdgeType
from causal_trace.engine.pruner import CausalContextCompiler


def run_context_load_benchmark(num_steps: int = 30):
    dag = CausalDAG()
    compiler = CausalContextCompiler()

    g = dag.add_node(GoalNode(content="Compile & Deploy Microservice Fleet"))
    prev_id = g.id

    for i in range(1, num_steps + 1):
        act = dag.add_node(ActionNode(command_str=f"step_task_{i} --verbose", step_index=i))
        obs = dag.add_node(
            ObservationNode(
                step_index=i,
                stdout=f"Log output block for step {i}: " + ("DEBUG TRACE " * 40),
            )
        )
        dag.add_edge(prev_id, act.id, EdgeType.ENABLES)
        dag.add_edge(act.id, obs.id, EdgeType.OBSERVES)
        prev_id = act.id

    dag.add_node(InvariantNode(predicate="ClusterHealthy", status=InvariantStatus.SATISFIED))

    compiled = compiler.compile(dag, target_goal_id=g.id)
    return {
        "raw_tokens": compiled.raw_token_estimate,
        "pruned_tokens": compiled.pruned_token_estimate,
        "compression_ratio": compiled.compression_ratio,
        "causal_completeness": compiled.causal_completeness,
    }
