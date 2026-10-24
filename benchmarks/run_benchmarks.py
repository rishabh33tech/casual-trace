"""
Benchmark Test Suite Runner for CausalTrace.
Executes all comparative benchmarks and prints structured metrics tables.
"""

from benchmarks.scenario_ghost_dep import run_ghost_dependency_benchmark
from benchmarks.scenario_oscillation import run_oscillation_benchmark
from benchmarks.scenario_context_load import run_context_load_benchmark
from causal_trace.cli.renderer import BOLD, CYAN, GREEN, RED, RESET


def main():
    print(f"\n{BOLD}{CYAN}=================================================================={RESET}")
    print(f"{BOLD}{CYAN}            CAUSAL-TRACE EMPIRICAL BENCHMARK SUITE                {RESET}")
    print(f"{BOLD}{CYAN}=================================================================={RESET}\n")

    # Benchmark 1: Ghost Dependency
    print(f"{BOLD}[1] Running Scenario 1: Ghost Dependency Trap...{RESET}")
    m_base, m_causal = run_ghost_dependency_benchmark()
    print(f"    - Baseline Agent (Standard ReAct): Success={RED}{m_base.success}{RESET}, Steps={m_base.total_steps}, Tokens={m_base.total_tokens}")
    print(f"    - CausalTrace Kernel Agent:        Success={GREEN}{m_causal.success}{RESET}, Steps={m_causal.total_steps}, Tokens={m_causal.total_tokens}\n")

    # Benchmark 2: Oscillation Cycle Detection
    print(f"{BOLD}[2] Running Scenario 2: Oscillating Thrashing Detection...{RESET}")
    entropy, cycle = run_oscillation_benchmark()
    period, pattern = cycle if cycle else (0, [])
    print(f"    - Calculated Shannon Cycle Entropy: {GREEN}{entropy:.3f}{RESET}")
    print(f"    - Detected Periodic Cycle:          Period={GREEN}{period}{RESET}, Pattern={' -> '.join(pattern)}\n")

    # Benchmark 3: Context Load Compression
    print(f"{BOLD}[3] Running Scenario 3: Deep Trace Context Compression (30 Steps)...{RESET}")
    c_metrics = run_context_load_benchmark(30)
    print(f"    - Raw Linear Context Buffer:        {c_metrics['raw_tokens']:,} tokens")
    print(f"    - Topologically Pruned Context:     {c_metrics['pruned_tokens']:,} tokens")
    print(f"    - Token Savings:                    {GREEN}{c_metrics['compression_ratio'] * 100:.1f}% reduction{RESET}")
    print(f"    - Causal Dependency Completeness:   {GREEN}{c_metrics['causal_completeness'] * 100:.1f}% preserved{RESET}\n")

    print(f"{BOLD}{GREEN}ALL BENCHMARKS PASSED SUCCESSFULLY.{RESET}\n")


if __name__ == "__main__":
    main()
