#!/usr/bin/env python3
"""Render STARK_Benchmark evaluation progress with Rich tables.

Progress uses the same completion rule as ``scripts/run_parallel_eval.sh``:
a case is completed only when the latest matching result row under
``results/<model>/<mode>/`` has a finite numeric score. Missing rows, NaN, inf,
non-numeric scores, and unreadable result files are treated as pending.

Examples:
    uv run python scripts/estimate_eval_progress.py
    uv run python scripts/estimate_eval_progress.py --model qwen3-4b-thinking-2507 --mode text
    uv run python scripts/estimate_eval_progress.py --suite tier2 --top-k 20
    uv run python scripts/estimate_eval_progress.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

try:  # Supports both `python scripts/foo.py` and `python -m scripts.foo`.
    from report_results import load_results, load_task_meta, latest_rows, read_task_names
except ModuleNotFoundError:  # pragma: no cover - import layout fallback.
    from scripts.report_results import load_results, load_task_meta, latest_rows, read_task_names

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_TASKS_DIR = REPO_ROOT / "tasks"
DEFAULT_MODE = "text"


@dataclass(frozen=True)
class CaseSpec:
    tier: str
    task: str
    index: int


@dataclass(frozen=True)
class TaskProgress:
    tier: str
    task: str
    completed: int
    observed: int
    nonfinite: int
    missing: int
    total: int
    progress: float
    score_sum: float
    mean_score_done: float | None
    exact_matches: int
    exact_total: int
    exact_match_done: float | None


@dataclass(frozen=True)
class TierProgress:
    tier: str
    completed: int
    observed: int
    nonfinite: int
    missing: int
    total: int
    progress: float
    score_sum: float
    mean_score_done: float | None
    exact_matches: int
    exact_total: int
    exact_match_done: float | None
    completed_tasks: int
    started_tasks: int
    total_tasks: int


@dataclass(frozen=True)
class RunProgress:
    model: str
    mode: str
    completed: int
    observed: int
    nonfinite: int
    missing: int
    total: int
    progress: float
    score_sum: float
    mean_score_done: float | None
    exact_matches: int
    exact_total: int
    exact_match_done: float | None
    completed_tasks: int
    started_tasks: int
    total_tasks: int
    tiers: list[TierProgress]
    tasks: list[TaskProgress]


def split_values(values: list[str] | None) -> list[str]:
    if not values:
        return []
    out: list[str] = []
    for value in values:
        out.extend(part.strip() for part in value.split(",") if part.strip())
    return out


def read_tasks_file(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")]


def selected_tasks(args: argparse.Namespace) -> list[str]:
    explicit_tasks = split_values(args.task)
    if explicit_tasks:
        return explicit_tasks
    if args.tasks_file:
        return read_tasks_file(args.tasks_file)
    if args.suite == "all":
        tasks: list[str] = []
        for tier in ("tier1", "tier2", "tier3"):
            tasks.extend(read_task_names(args.tasks_dir, tier))
        return tasks
    return read_task_names(args.tasks_dir, args.suite)


def build_cases(args: argparse.Namespace) -> list[CaseSpec]:
    task_meta = load_task_meta(args.tasks_dir, args.data_dir)
    cases: list[CaseSpec] = []
    unknown_tasks: list[str] = []
    for task in selected_tasks(args):
        meta = task_meta.get(task)
        if meta is None:
            unknown_tasks.append(task)
            continue
        indices = meta.expected_indices[: args.limit] if args.limit is not None else meta.expected_indices
        for index in indices:
            cases.append(CaseSpec(tier=meta.tier, task=task, index=int(index)))
    if unknown_tasks:
        print(f"WARN: skipped tasks without metadata/data: {', '.join(unknown_tasks)}", file=sys.stderr)
    if not cases:
        raise SystemExit("ERROR: no cases generated; check --suite/--task/--tasks-file and data/.")
    return cases


def discover_model_modes(results_dir: Path, models: list[str], modes: list[str]) -> list[tuple[str, str]]:
    if models and modes:
        return [(model, mode) for model in models for mode in modes]

    discovered: dict[str, set[str]] = {}
    if results_dir.exists():
        for model_dir in sorted(path for path in results_dir.iterdir() if path.is_dir() and not path.name.startswith("_")):
            if models and model_dir.name not in models:
                continue
            for mode_dir in sorted(path for path in model_dir.iterdir() if path.is_dir()):
                if modes and mode_dir.name not in modes:
                    continue
                discovered.setdefault(model_dir.name, set()).add(mode_dir.name)

    if models:
        pairs: list[tuple[str, str]] = []
        for model in models:
            model_modes = modes or sorted(discovered.get(model, [])) or [DEFAULT_MODE]
            pairs.extend((model, mode) for mode in model_modes)
        return pairs

    return [(model, mode) for model, model_modes in sorted(discovered.items()) for mode in sorted(model_modes)]


def latest_score_lookup(latest: pd.DataFrame) -> dict[tuple[str, str, str, int], float | None]:
    lookup: dict[tuple[str, str, str, int], float | None] = {}
    if latest.empty:
        return lookup

    for row in latest.itertuples(index=False):
        try:
            index = int(row.index)
        except Exception:
            continue
        score = getattr(row, "score")
        finite_score = float(score) if pd.notna(score) and np.isfinite(float(score)) else None
        lookup[(str(row.model), str(row.mode), str(row.task), index)] = finite_score
    return lookup


def progress_style(progress: float) -> str:
    if progress >= 1.0:
        return "green"
    if progress > 0:
        return "yellow"
    return "red"


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def format_optional_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return format_pct(value)


def format_optional_score(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def exact_style(value: float | None) -> str:
    if value is None:
        return "dim"
    if value >= 0.8:
        return "green"
    if value >= 0.5:
        return "yellow"
    return "red"


def ratio_text(done: int, total: int) -> str:
    return f"{done:,}/{total:,}"


def summarize_run(model: str, mode: str, cases: list[CaseSpec], lookup: dict[tuple[str, str, str, int], float | None]) -> RunProgress:
    by_task: dict[tuple[str, str], dict[str, int | str]] = {}
    for case in cases:
        key = (model, mode, case.task, case.index)
        has_row = key in lookup
        score = lookup.get(key)
        is_complete = score is not None
        task_key = (case.tier, case.task)
        entry = by_task.setdefault(
            task_key,
            {
                "tier": case.tier,
                "task": case.task,
                "completed": 0,
                "observed": 0,
                "nonfinite": 0,
                "missing": 0,
                "total": 0,
                "score_sum": 0.0,
                "exact_matches": 0,
                "exact_total": 0,
            },
        )
        entry["total"] = int(entry["total"]) + 1
        if is_complete:
            entry["completed"] = int(entry["completed"]) + 1
            entry["observed"] = int(entry["observed"]) + 1
            entry["score_sum"] = float(entry["score_sum"]) + float(score)
            if case.tier in {"tier2", "tier3"}:
                entry["exact_total"] = int(entry["exact_total"]) + 1
                if float(score) == 0.0:
                    entry["exact_matches"] = int(entry["exact_matches"]) + 1
        elif has_row:
            entry["observed"] = int(entry["observed"]) + 1
            entry["nonfinite"] = int(entry["nonfinite"]) + 1
        else:
            entry["missing"] = int(entry["missing"]) + 1

    task_progress: list[TaskProgress] = []
    for entry in by_task.values():
        completed = int(entry["completed"])
        total = int(entry["total"])
        score_sum = float(entry["score_sum"])
        exact_matches = int(entry["exact_matches"])
        exact_total = int(entry["exact_total"])
        task_progress.append(
            TaskProgress(
                tier=str(entry["tier"]),
                task=str(entry["task"]),
                completed=completed,
                observed=int(entry["observed"]),
                nonfinite=int(entry["nonfinite"]),
                missing=int(entry["missing"]),
                total=total,
                progress=completed / total if total else 0.0,
                score_sum=score_sum,
                mean_score_done=score_sum / completed if completed else None,
                exact_matches=exact_matches,
                exact_total=exact_total,
                exact_match_done=exact_matches / exact_total if exact_total else None,
            )
        )
    task_progress.sort(key=lambda item: (item.tier, item.progress, item.task))

    tiers: list[TierProgress] = []
    for tier in ("tier1", "tier2", "tier3"):
        tier_tasks = [item for item in task_progress if item.tier == tier]
        if not tier_tasks:
            continue
        completed = sum(item.completed for item in tier_tasks)
        observed = sum(item.observed for item in tier_tasks)
        nonfinite = sum(item.nonfinite for item in tier_tasks)
        missing = sum(item.missing for item in tier_tasks)
        total = sum(item.total for item in tier_tasks)
        score_sum = sum(item.score_sum for item in tier_tasks)
        exact_matches = sum(item.exact_matches for item in tier_tasks)
        exact_total = sum(item.exact_total for item in tier_tasks)
        tiers.append(
            TierProgress(
                tier=tier,
                completed=completed,
                observed=observed,
                nonfinite=nonfinite,
                missing=missing,
                total=total,
                progress=completed / total if total else 0.0,
                score_sum=score_sum,
                mean_score_done=score_sum / completed if completed else None,
                exact_matches=exact_matches,
                exact_total=exact_total,
                exact_match_done=exact_matches / exact_total if exact_total else None,
                completed_tasks=sum(1 for item in tier_tasks if item.completed >= item.total),
                started_tasks=sum(1 for item in tier_tasks if item.observed > 0),
                total_tasks=len(tier_tasks),
            )
        )

    completed = sum(item.completed for item in task_progress)
    observed = sum(item.observed for item in task_progress)
    nonfinite = sum(item.nonfinite for item in task_progress)
    missing = sum(item.missing for item in task_progress)
    total = sum(item.total for item in task_progress)
    score_sum = sum(item.score_sum for item in task_progress)
    exact_matches = sum(item.exact_matches for item in task_progress)
    exact_total = sum(item.exact_total for item in task_progress)
    return RunProgress(
        model=model,
        mode=mode,
        completed=completed,
        observed=observed,
        nonfinite=nonfinite,
        missing=missing,
        total=total,
        progress=completed / total if total else 0.0,
        score_sum=score_sum,
        mean_score_done=score_sum / completed if completed else None,
        exact_matches=exact_matches,
        exact_total=exact_total,
        exact_match_done=exact_matches / exact_total if exact_total else None,
        completed_tasks=sum(1 for item in task_progress if item.completed >= item.total),
        started_tasks=sum(1 for item in task_progress if item.observed > 0),
        total_tasks=len(task_progress),
        tiers=tiers,
        tasks=task_progress,
    )


def status_text(task: TaskProgress) -> Text:
    if task.completed >= task.total:
        return Text("done", style="green")
    parts = Text()
    if task.missing:
        parts.append(f"missing {task.missing}", style="red")
    if task.nonfinite:
        if len(parts):
            parts.append(" · ", style="dim")
        parts.append(f"nonfinite {task.nonfinite}", style="magenta")
    if not len(parts):
        parts.append("partial", style="yellow")
    return parts


def render_header(console: Console, args: argparse.Namespace, cases: list[CaseSpec], runs: list[RunProgress]) -> None:
    text = Text()
    text.append("Data: ", style="bold")
    text.append(str(args.data_dir))
    text.append("\nResults: ", style="bold")
    text.append(str(args.results_dir))
    text.append("\nSuite: ", style="bold")
    text.append(args.suite)
    if args.limit is not None:
        text.append("   Limit/task: ", style="bold")
        text.append(str(args.limit))
    text.append("\nCases: ", style="bold")
    text.append(f"{len(cases):,}")
    text.append("   Tasks: ", style="bold")
    text.append(str(len({(case.tier, case.task) for case in cases})))
    text.append("   Runs: ", style="bold")
    text.append(str(len(runs)))
    console.print(Panel(text, title="STARK Evaluation Progress", border_style="cyan", box=box.ROUNDED))


def render_overall(console: Console, runs: list[RunProgress]) -> None:
    if not runs:
        console.print("[yellow]No model/mode result directories found. Pass --model/--mode to inspect an empty or future run.[/yellow]")
        return

    table = Table(title="Overall by run", box=box.SIMPLE_HEAVY, header_style="bold cyan")
    table.add_column("Run", min_width=30, overflow="fold")
    table.add_column("Progress", justify="right", no_wrap=True)
    table.add_column("Bar", min_width=18)
    table.add_column("Finite", justify="right", no_wrap=True)
    table.add_column("Mean(done)", justify="right", no_wrap=True)
    table.add_column("Exact(done)", justify="right", no_wrap=True)
    table.add_column("Pending", justify="right", no_wrap=True)
    table.add_column("Tasks done", justify="right", no_wrap=True)

    for run in runs:
        style = progress_style(run.progress)
        pending = run.missing + run.nonfinite
        pending_text = Text()
        pending_text.append(f"{pending:,}", style=progress_style(0 if pending else 1))
        if pending:
            pending_text.append(" (", style="dim")
            pending_text.append(f"miss {run.missing:,}", style="red" if run.missing else "dim")
            pending_text.append(" / ", style="dim")
            pending_text.append(f"nan {run.nonfinite:,}", style="magenta" if run.nonfinite else "dim")
            pending_text.append(")", style="dim")
        table.add_row(
            f"{run.model} / {run.mode}",
            Text(format_pct(run.progress), style=style),
            ProgressBar(total=1.0, completed=run.progress, width=20, complete_style=style),
            ratio_text(run.completed, run.total),
            format_optional_score(run.mean_score_done),
            Text(format_optional_pct(run.exact_match_done), style=exact_style(run.exact_match_done)),
            pending_text,
            ratio_text(run.completed_tasks, run.total_tasks),
        )
    console.print(table)


def render_tiers(console: Console, run: RunProgress) -> None:
    table = Table(title=f"{run.model} / {run.mode} by tier", box=box.SIMPLE, header_style="bold magenta")
    table.add_column("Tier", no_wrap=True)
    table.add_column("Progress", justify="right", no_wrap=True)
    table.add_column("Bar", min_width=20)
    table.add_column("Finite", justify="right", no_wrap=True)
    table.add_column("Mean(done)", justify="right", no_wrap=True)
    table.add_column("Exact(done)", justify="right", no_wrap=True)
    table.add_column("Observed", justify="right", no_wrap=True)
    table.add_column("Missing", justify="right", no_wrap=True)
    table.add_column("Nonfinite", justify="right", no_wrap=True)
    table.add_column("Tasks", justify="right", no_wrap=True)

    for tier in run.tiers:
        style = progress_style(tier.progress)
        table.add_row(
            tier.tier,
            Text(format_pct(tier.progress), style=style),
            ProgressBar(total=1.0, completed=tier.progress, width=22, complete_style=style),
            ratio_text(tier.completed, tier.total),
            format_optional_score(tier.mean_score_done),
            Text(format_optional_pct(tier.exact_match_done), style=exact_style(tier.exact_match_done)),
            ratio_text(tier.observed, tier.total),
            f"{tier.missing:,}",
            f"{tier.nonfinite:,}",
            f"{ratio_text(tier.completed_tasks, tier.total_tasks)} done",
        )
    console.print(table)


def render_attention(console: Console, run: RunProgress, *, top_k: int, all_tasks: bool) -> None:
    tasks = run.tasks if all_tasks else [task for task in run.tasks if task.completed < task.total]
    tasks = sorted(tasks, key=lambda item: (item.progress >= 1.0, item.progress, -item.nonfinite, -item.missing, item.tier, item.task))
    if top_k >= 0:
        tasks = tasks[:top_k]
    if not tasks:
        console.print(f"[green]{run.model} / {run.mode}: all selected tasks complete.[/green]")
        return

    title = f"{run.model} / {run.mode} incomplete tasks" if not all_tasks else f"{run.model} / {run.mode} tasks"
    table = Table(title=title, box=box.SIMPLE, header_style="bold yellow")
    table.add_column("Tier", no_wrap=True)
    table.add_column("Task", overflow="fold")
    table.add_column("Finite", justify="right", no_wrap=True)
    table.add_column("Progress", justify="right", no_wrap=True)
    table.add_column("Mean(done)", justify="right", no_wrap=True)
    table.add_column("Exact(done)", justify="right", no_wrap=True)
    table.add_column("Missing", justify="right", no_wrap=True)
    table.add_column("Nonfinite", justify="right", no_wrap=True)
    table.add_column("Status")

    for task in tasks:
        style = progress_style(task.progress)
        table.add_row(
            task.tier,
            task.task,
            ratio_text(task.completed, task.total),
            Text(format_pct(task.progress), style=style),
            format_optional_score(task.mean_score_done),
            Text(format_optional_pct(task.exact_match_done), style=exact_style(task.exact_match_done)),
            f"{task.missing:,}",
            f"{task.nonfinite:,}",
            status_text(task),
        )
    console.print(table)


def make_console() -> Console:
    probe = Console()
    return Console(width=max(probe.size.width, 160))


def render_report(args: argparse.Namespace, cases: list[CaseSpec], runs: list[RunProgress]) -> None:
    console = make_console()
    render_header(console, args, cases, runs)
    render_overall(console, runs)
    if args.no_details:
        return
    for run in runs:
        render_tiers(console, run)
        render_attention(console, run, top_k=args.top_k, all_tasks=args.all_tasks)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR, help="Evaluation results directory. Default: ./results")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Extracted STARK data directory. Default: ./data")
    parser.add_argument("--tasks-dir", type=Path, default=DEFAULT_TASKS_DIR, help="Task-list directory. Default: ./tasks")
    parser.add_argument("--model", action="append", help="Only include this model; can be repeated or comma-separated.")
    parser.add_argument("--mode", action="append", help="Only include this mode; can be repeated or comma-separated.")
    parser.add_argument("--suite", choices=("all", "tier1", "tier2", "tier3"), default="all", help="Task suite denominator. Default: all")
    parser.add_argument("--task", action="append", help="Only include this task; can be repeated or comma-separated. Overrides --suite/--tasks-file.")
    parser.add_argument("--tasks-file", type=Path, default=None, help="Read selected tasks from a file, one task per line.")
    parser.add_argument("--limit", type=int, default=None, help="Count at most N indices per task, matching run_parallel_eval.sh smoke-test semantics.")
    parser.add_argument("--top-k", type=int, default=25, help="Number of incomplete tasks to show per run. Use -1 for all.")
    parser.add_argument("--all-tasks", action="store_true", help="Show every selected task instead of only incomplete tasks.")
    parser.add_argument("--no-details", action="store_true", help="Only print the overall run table.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of Rich tables.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("ERROR: --limit must be a positive integer.")
    if args.top_k < -1:
        raise SystemExit("ERROR: --top-k must be -1 or a non-negative integer.")

    models = split_values(args.model)
    modes = split_values(args.mode)
    cases = build_cases(args)
    model_modes = discover_model_modes(args.results_dir, models, modes)

    raw, skipped = load_results(args.results_dir, set(models) if models else None, set(modes) if modes else None)
    latest = latest_rows(raw)
    lookup = latest_score_lookup(latest)
    runs = [summarize_run(model, mode, cases, lookup) for model, mode in model_modes]
    runs.sort(key=lambda item: (item.model, item.mode))

    if args.json:
        payload = {
            "results_dir": str(args.results_dir),
            "data_dir": str(args.data_dir),
            "tasks_dir": str(args.tasks_dir),
            "suite": args.suite,
            "limit": args.limit,
            "num_cases": len(cases),
            "num_tasks": len({(case.tier, case.task) for case in cases}),
            "skipped_result_files": skipped.to_dict(orient="records") if not skipped.empty else [],
            "runs": [asdict(run) for run in runs],
        }
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return

    if not skipped.empty:
        print(f"WARN: skipped {len(skipped)} unreadable/malformed result files; use --json for details.", file=sys.stderr)
    render_report(args, cases, runs)


if __name__ == "__main__":
    main()
