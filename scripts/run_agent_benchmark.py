#!/usr/bin/env python3
"""Run agent E2E benchmark against a live API and write JSON report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from legal_assistant.evaluation.agent_benchmark import BenchmarkRunner, write_report

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "agent_benchmark_report.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run agent benchmark and write report JSON.")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="API base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Report output path (default: {DEFAULT_OUTPUT.name} in project root)",
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=None,
        help="Optional tasks.yaml path (default: bundled agent_benchmark/tasks.yaml)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP timeout per chat request in seconds",
    )
    args = parser.parse_args()

    runner_kwargs: dict = {
        "base_url": args.base_url,
        "timeout": args.timeout,
    }
    if args.tasks is not None:
        runner_kwargs["tasks_path"] = args.tasks

    with BenchmarkRunner(**runner_kwargs) as runner:
        results = runner.run_all()

    report = write_report(results, args.output)
    print(f"Wrote report: {args.output.resolve()}")
    print(json.dumps(report.summary, ensure_ascii=False, indent=2))
    return 0 if report.summary.get("task_success_rate", 0) == 1.0 else 1


if __name__ == "__main__":
    sys.exit(main())
