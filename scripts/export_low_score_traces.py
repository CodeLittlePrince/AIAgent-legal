#!/usr/bin/env python3
"""Export low-scoring Langfuse traces as new agent benchmark tasks.

This is a stub documenting the feedback-loop workflow described in the design spec.
Production usage would:

1. Authenticate to Langfuse with ``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY``.
2. Query traces with user feedback scores below a threshold (default: 3 on a 1-5 scale).
3. Map each trace to a benchmark task entry (session turns + expected intent/outcome).
4. Append new tasks to ``src/legal_assistant/evaluation/agent_benchmark/tasks.yaml``.

Example (not implemented):

    from langfuse import Langfuse

    client = Langfuse()
    traces = client.fetch_traces(
        filter='scores.user_feedback < 3',
        limit=100,
    )
    for trace in traces:
        # Convert trace input/output spans into benchmark turns
        ...

Run manually after configuring Langfuse:

    python scripts/export_low_score_traces.py --threshold 3 --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_TASKS_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/legal_assistant/evaluation/agent_benchmark/tasks.yaml"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export low-score Langfuse traces to agent benchmark tasks (stub)."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=3.0,
        help="Maximum score (exclusive) to export. Default: 3.0",
    )
    parser.add_argument(
        "--tasks-path",
        type=Path,
        default=DEFAULT_TASKS_PATH,
        help="Path to tasks.yaml",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without writing tasks.yaml",
    )
    args = parser.parse_args()

    print("export_low_score_traces.py is a documentation stub.")
    print(f"Would export traces with score < {args.threshold} to {args.tasks_path}")
    if args.dry_run:
        print("Dry run: no files modified.")
    else:
        print("No Langfuse export performed. Implement Langfuse API integration to enable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
