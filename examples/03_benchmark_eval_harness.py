"""
Example 03: Autonomous Code Generation Benchmark Evaluation & Pass@k.
"""

from agent_lab.eval import BenchmarkHarness, BenchmarkTask, PassAtKEstimator


def main():
    task = BenchmarkTask(
        task_id="demo/factorial",
        prompt="Write a recursive function `factorial(n)`.",
        entry_point="factorial",
        test_code="assert factorial(0) == 1\nassert factorial(5) == 120\n",
    )

    harness = BenchmarkHarness(tasks=[task])

    # Sample candidate completions
    candidate_1 = "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)\n"
    candidate_2 = "def factorial(n):\n    return n * 2  # incorrect\n"

    res1 = harness.evaluate_candidate("demo/factorial", candidate_1)
    res2 = harness.evaluate_candidate("demo/factorial", candidate_2)

    print(f"Candidate 1 evaluation: passed={res1['passed']}, AST complexity={res1['ast']['cyclomatic_complexity']}")
    print(f"Candidate 2 evaluation: passed={res2['passed']}, error={res2['error']}")

    # Pass@k calculation over n=10 samples with c=8 correct
    metrics = PassAtKEstimator.compute_all(n=10, c=8, k_values=[1, 5])
    print(f"Pass@k Metrics (n=10, c=8): {metrics}")


if __name__ == "__main__":
    main()
