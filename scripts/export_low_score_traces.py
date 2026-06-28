#!/usr/bin/env python3
"""Export low-scoring Langfuse traces as new agent benchmark tasks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import httpx
import yaml

from legal_assistant.config import settings

DEFAULT_TASKS_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/legal_assistant/evaluation/agent_benchmark/tasks.yaml"
)
SCORE_NAME = "user_feedback"


class LangfuseExportClient:
    """Minimal Langfuse public API client for score/trace export."""

    def __init__(self, host: str, public_key: str, secret_key: str) -> None:
        self._client = httpx.Client(
            base_url=host.rstrip("/"),
            auth=(public_key, secret_key),
            timeout=30.0,
            trust_env=False,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> LangfuseExportClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def list_scores(self, *, name: str, page: int = 1, limit: int = 100) -> dict[str, Any]:
        response = self._client.get(
            "/api/public/scores",
            params={"name": name, "page": page, "limit": limit},
        )
        response.raise_for_status()
        return response.json()

    def get_trace(self, trace_id: str) -> dict[str, Any]:
        response = self._client.get(f"/api/public/traces/{trace_id}")
        response.raise_for_status()
        return response.json()


def _coerce_message(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def extract_user_message(trace: dict[str, Any], score: dict[str, Any]) -> str | None:
    trace_input = trace.get("input")
    if isinstance(trace_input, str):
        return _coerce_message(trace_input)
    if isinstance(trace_input, dict):
        for key in ("message", "user_message", "query", "content"):
            message = _coerce_message(trace_input.get(key))
            if message:
                return message

    metadata = trace.get("metadata") or {}
    if isinstance(metadata, dict):
        for key in ("message", "user_message"):
            message = _coerce_message(metadata.get(key))
            if message:
                return message

    return _coerce_message(score.get("comment"))


def infer_tools_used(trace: dict[str, Any]) -> list[str]:
    trace_output = trace.get("output")
    if isinstance(trace_output, dict):
        tools = trace_output.get("tools_used")
        if isinstance(tools, list):
            return [str(item) for item in tools if item]

    metadata = trace.get("metadata") or {}
    if isinstance(metadata, dict):
        raw = metadata.get("tools_used")
        if isinstance(raw, str) and raw != "none":
            return [part for part in raw.split(",") if part]

    return []


def build_task_from_trace(
    trace_id: str,
    trace: dict[str, Any],
    score: dict[str, Any],
) -> dict[str, Any] | None:
    user_message = extract_user_message(trace, score)
    if not user_message:
        return None

    tools_used = infer_tools_used(trace)
    task_id = f"exported_{trace_id.replace('-', '')[:12]}"
    expect: dict[str, Any] = {"task_success": True}
    if "search_legal_knowledge" in tools_used:
        expect["tools_contains"] = "search_legal_knowledge"
        expect["citations"] = True
        expect["disclaimer"] = True
    elif "get_weather_forecast" in tools_used:
        expect["tools_contains"] = "get_weather_forecast"
        expect["tool_success"] = True
    elif not tools_used:
        expect["tools_empty"] = True

    comment = _coerce_message(score.get("comment"))
    name = f"导出低分用例 ({trace_id[:8]})"
    if comment:
        name = f"低分反馈: {comment[:40]}"

    return {
        "id": task_id,
        "name": name,
        "category": "feedback_export",
        "source_trace_id": trace_id,
        "source_score": score.get("value"),
        "turns": [{"user": user_message, "expect": expect}],
    }


def fetch_low_scores(
    client: LangfuseExportClient,
    *,
    threshold: float,
    score_name: str,
) -> list[dict[str, Any]]:
    low_scores: list[dict[str, Any]] = []
    page = 1

    while True:
        payload = client.list_scores(name=score_name, page=page, limit=100)
        for score in payload.get("data") or []:
            value = score.get("value")
            if value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if numeric < threshold:
                low_scores.append(score)

        meta = payload.get("meta") or {}
        total_pages = int(meta.get("totalPages") or 1)
        if page >= total_pages:
            break
        page += 1

    return low_scores


def append_tasks(tasks_path: Path, new_tasks: list[dict[str, Any]], *, dry_run: bool) -> int:
    with tasks_path.open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}

    existing = list(document.get("tasks") or [])
    existing_ids = {task.get("id") for task in existing}
    added = 0

    for task in new_tasks:
        task_id = task.get("id")
        if not task_id or task_id in existing_ids:
            continue
        existing.append(task)
        existing_ids.add(task_id)
        added += 1

    document["tasks"] = existing
    if not dry_run and added:
        with tasks_path.open("w", encoding="utf-8") as handle:
            yaml.dump(
                document,
                handle,
                allow_unicode=True,
                sort_keys=False,
                default_flow_style=False,
            )

    return added


def export_low_score_traces(
    *,
    host: str,
    public_key: str,
    secret_key: str,
    threshold: float,
    tasks_path: Path,
    score_name: str = SCORE_NAME,
    dry_run: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    new_tasks: list[dict[str, Any]] = []

    with LangfuseExportClient(host, public_key, secret_key) as client:
        low_scores = fetch_low_scores(client, threshold=threshold, score_name=score_name)
        for score in low_scores:
            trace_id = score.get("traceId") or score.get("trace_id")
            if not trace_id:
                continue
            try:
                trace = client.get_trace(str(trace_id))
            except httpx.HTTPError:
                continue
            task = build_task_from_trace(str(trace_id), trace, score)
            if task:
                new_tasks.append(task)

    added = append_tasks(tasks_path, new_tasks, dry_run=dry_run)
    return new_tasks, added


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export low-score Langfuse traces to agent benchmark tasks."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Export scores strictly below this value (default: 1.0 for 0/1 feedback).",
    )
    parser.add_argument(
        "--score-name",
        default=SCORE_NAME,
        help=f"Langfuse score name to filter (default: {SCORE_NAME}).",
    )
    parser.add_argument(
        "--tasks-path",
        type=Path,
        default=DEFAULT_TASKS_PATH,
        help="Path to tasks.yaml",
    )
    parser.add_argument(
        "--host",
        default=settings.langfuse_host,
        help="Langfuse host URL",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without writing tasks.yaml",
    )
    args = parser.parse_args()

    public_key = settings.langfuse_public_key
    secret_key = settings.langfuse_secret_key
    if not public_key or not secret_key:
        print(
            "Langfuse credentials missing. Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY.",
            file=sys.stderr,
        )
        return 1

    try:
        new_tasks, added = export_low_score_traces(
            host=args.host,
            public_key=public_key,
            secret_key=secret_key,
            threshold=args.threshold,
            tasks_path=args.tasks_path,
            score_name=args.score_name,
            dry_run=args.dry_run,
        )
    except httpx.HTTPError as exc:
        print(f"Langfuse API request failed: {exc}", file=sys.stderr)
        return 1

    print(f"Found {len(new_tasks)} exportable low-score trace(s) below {args.threshold}.")
    for task in new_tasks:
        print(f"  - {task['id']}: {task['name']}")

    if args.dry_run:
        print(f"Dry run: would append {added} task(s) to {args.tasks_path}")
    else:
        print(f"Appended {added} new task(s) to {args.tasks_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
