#!/usr/bin/env python3
"""Summarize STARK_Benchmark evaluation outputs under results/.

Typical usage after an evaluation run:

    uv run python scripts/report_results.py

The script scans result CSVs written by main.py under:

    results/<model>/<mode>/*.csv

It writes normalized rows and summaries to:

    results/_reports/<timestamp>/

All paths under results/ are ignored by git.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

FIELD_VARIABLE_TASKS = {
    "spatial_impute",
    "spatiotemporal_forecast",
    "spatiotemporal_impute",
    "temporal_impute",
}


@dataclass(frozen=True)
class TaskMeta:
    tier: str
    task: str
    expected_indices: tuple[int, ...]

    @property
    def expected_count(self) -> int:
        return len(self.expected_indices)

    @property
    def metric_hint(self) -> str:
        if self.task in FIELD_VARIABLE_TASKS:
            return "relative_error"
        if self.tier in {"tier2", "tier3"}:
            return "exact_match_error"
        return "rmse"

    @property
    def exact_match_task(self) -> bool:
        return self.tier in {"tier2", "tier3"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report all STARK_Benchmark experiment results under results/."
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--tasks-dir", type=Path, default=Path("tasks"))
    parser.add_argument("--model", action="append", help="Only include this model; can be repeated.")
    parser.add_argument("--mode", action="append", help="Only include this mode; can be repeated.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for report CSVs. Default: results/_reports/<timestamp>.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=15,
        help="Number of worst tasks to print per model/mode.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Only print a terminal summary; do not write report CSV files.",
    )
    parser.add_argument(
        "--expect-all-tasks",
        action="store_true",
        help="When checking missing cases, require every task in tasks/*.txt. Default: only check tasks that have at least one result for the model/mode.",
    )
    return parser.parse_args()


def read_task_names(tasks_dir: Path, tier: str) -> list[str]:
    path = tasks_dir / f"{tier}.txt"
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def task_data_dir(data_dir: Path, task: str) -> Path | None:
    mapping = {
        "loc_range": data_dir / "localization_range_0.01",
        "loc_bearing": data_dir / "localization_bearing_0.01",
        "loc_range_bearing": data_dir / "localization_range_bearing_[0.01, 0.01]",
        "loc_region": data_dir / "localization_region_4.0",
        "loc_event_temp": data_dir / "localization_event_temp_0.01",
        "loc_event_spatio": data_dir / "localization_event_spatio_0.01",
        "loc_event_spatio_temp": data_dir / "localization_event_spatio_temp_0.01",
        "track_range_online": data_dir / "tracking_range_0.01",
        "track_bearing_online": data_dir / "tracking_bearing_0.01",
        "track_range_bearing_online": data_dir / "tracking_range_bearing_[0.01, 0.01]",
        "track_region_online": data_dir / "tracking_region_4.0",
        "track_event_spatio_online": data_dir / "tracking_event_spatio_0.01",
        "track_event_temp_online": data_dir / "tracking_event_temp_0.01",
        "track_event_spatiotemp_online": data_dir / "tracking_event_spatio_temp_0.01",
        "spatial_impute": data_dir / "spatial_impute",
        "spatiotemporal_forecast": data_dir / "spatiotemporal_forecast",
        "spatiotemporal_impute": data_dir / "spatiotemporal_impute",
        "temporal_impute": data_dir / "temporal_impute",
    }
    if task in mapping:
        return mapping[task]

    for candidate in (
        data_dir / "tier2_v2_spatial" / task,
        data_dir / "tier2_v2_temporal" / task,
        data_dir / "tier2_v2_spatial_temporal" / task,
        data_dir / "tier3_world_knowledge" / task,
    ):
        if candidate.exists():
            return candidate
    return None


def expected_indices_for_task(data_dir: Path, task: str) -> tuple[int, ...]:
    directory = task_data_dir(data_dir, task)
    if directory is None or not directory.exists():
        return tuple()

    indices: list[int] = []
    if task.startswith(("loc_", "track_")):
        for path in directory.glob("*_object_location.csv"):
            try:
                indices.append(int(path.name.split("_", 1)[0]))
            except ValueError:
                continue
    else:
        for path in directory.glob("*.csv"):
            try:
                indices.append(int(path.stem))
            except ValueError:
                continue
    return tuple(sorted(set(indices)))


def load_task_meta(tasks_dir: Path, data_dir: Path) -> dict[str, TaskMeta]:
    meta: dict[str, TaskMeta] = {}
    for tier in ("tier1", "tier2", "tier3"):
        for task in read_task_names(tasks_dir, tier):
            meta[task] = TaskMeta(
                tier=tier,
                task=task,
                expected_indices=expected_indices_for_task(data_dir, task),
            )
    return meta


def iter_result_files(results_dir: Path, models: set[str] | None, modes: set[str] | None) -> Iterable[Path]:
    if not results_dir.exists():
        return []
    files: list[Path] = []
    for model_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        if model_dir.name.startswith("_"):
            continue
        if models and model_dir.name not in models:
            continue
        for mode_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            if modes and mode_dir.name not in modes:
                continue
            files.extend(sorted(mode_dir.glob("*.csv")))
    return files


def load_results(results_dir: Path, models: set[str] | None, modes: set[str] | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []

    for path in iter_result_files(results_dir, models, modes):
        try:
            rel = path.relative_to(results_dir)
            model, mode = rel.parts[0], rel.parts[1]
        except Exception:
            skipped.append({"source_file": str(path), "reason": "unexpected path layout"})
            continue

        try:
            df = pd.read_csv(path)
        except Exception as exc:  # pragma: no cover - defensive CLI handling
            skipped.append({"source_file": str(path), "reason": f"read error: {exc}"})
            continue

        required = {"index", "task", "score"}
        if not required.issubset(df.columns):
            skipped.append({"source_file": str(path), "reason": f"missing columns: {sorted(required - set(df.columns))}"})
            continue

        mtime = path.stat().st_mtime
        for row_number, row in df.iterrows():
            try:
                index = int(row["index"])
            except Exception:
                index = np.nan
            rows.append(
                {
                    "model": model,
                    "mode": mode,
                    "task": str(row["task"]),
                    "index": index,
                    "score_raw": row["score"],
                    "score": pd.to_numeric(row["score"], errors="coerce"),
                    "source_file": str(path),
                    "source_mtime": mtime,
                    "row_number": int(row_number),
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(skipped)


def latest_rows(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw.copy()
    latest = raw.sort_values(["source_mtime", "row_number", "source_file"])
    latest = latest.drop_duplicates(["model", "mode", "task", "index"], keep="last")
    return latest.reset_index(drop=True)


def enrich_with_task_meta(df: pd.DataFrame, task_meta: dict[str, TaskMeta]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    meta_rows = []
    for task, meta in task_meta.items():
        meta_rows.append(
            {
                "task": task,
                "tier": meta.tier,
                "expected_count": meta.expected_count,
                "metric_hint": meta.metric_hint,
                "exact_match_task": meta.exact_match_task,
            }
        )
    meta_df = pd.DataFrame(meta_rows)
    out = df.merge(meta_df, on="task", how="left")
    out["tier"] = out["tier"].fillna("unknown")
    out["metric_hint"] = out["metric_hint"].fillna("unknown")
    out["exact_match_task"] = out["exact_match_task"].fillna(False).astype(bool)
    return out


def summarize_by_task(latest: pd.DataFrame) -> pd.DataFrame:
    if latest.empty:
        return pd.DataFrame()

    rows = []
    group_cols = ["model", "mode", "tier", "task", "metric_hint", "exact_match_task", "expected_count"]
    for keys, group in latest.groupby(group_cols, dropna=False):
        model, mode, tier, task, metric_hint, exact_match_task, expected_count = keys
        finite = group["score"].dropna()
        row = {
            "model": model,
            "mode": mode,
            "tier": tier,
            "task": task,
            "metric_hint": metric_hint,
            "expected_count": int(expected_count) if not pd.isna(expected_count) else 0,
            "observed_count": int(len(group)),
            "finite_count": int(finite.shape[0]),
            "nan_count": int(group["score"].isna().sum()),
            "mean_score": finite.mean() if not finite.empty else np.nan,
            "median_score": finite.median() if not finite.empty else np.nan,
            "std_score": finite.std(ddof=1) if finite.shape[0] > 1 else np.nan,
            "min_score": finite.min() if not finite.empty else np.nan,
            "max_score": finite.max() if not finite.empty else np.nan,
        }
        if bool(exact_match_task) and not finite.empty:
            row["exact_match_rate"] = float((finite == 0).mean())
            row["error_rate"] = float((finite != 0).mean())
        else:
            row["exact_match_rate"] = np.nan
            row["error_rate"] = np.nan
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["model", "mode", "tier", "task"]).reset_index(drop=True)


def summarize_by_tier(task_summary: pd.DataFrame) -> pd.DataFrame:
    if task_summary.empty:
        return pd.DataFrame()
    rows = []
    for (model, mode, tier), group in task_summary.groupby(["model", "mode", "tier"], dropna=False):
        finite_weight = group["finite_count"].replace(0, np.nan)
        weighted_mean = np.average(
            group["mean_score"].fillna(0),
            weights=group["finite_count"],
        ) if group["finite_count"].sum() > 0 else np.nan

        exact_group = group[group["exact_match_rate"].notna()]
        if not exact_group.empty and exact_group["finite_count"].sum() > 0:
            exact_rate = np.average(exact_group["exact_match_rate"], weights=exact_group["finite_count"])
        else:
            exact_rate = np.nan

        rows.append(
            {
                "model": model,
                "mode": mode,
                "tier": tier,
                "task_count": int(group["task"].nunique()),
                "expected_count": int(group["expected_count"].sum()),
                "observed_count": int(group["observed_count"].sum()),
                "finite_count": int(group["finite_count"].sum()),
                "nan_count": int(group["nan_count"].sum()),
                "mean_score_weighted": weighted_mean,
                "exact_match_rate_weighted": exact_rate,
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "mode", "tier"]).reset_index(drop=True)


def summarize_by_run(task_summary: pd.DataFrame) -> pd.DataFrame:
    if task_summary.empty:
        return pd.DataFrame()
    tier_summary = summarize_by_tier(task_summary)
    rows = []
    for (model, mode), group in tier_summary.groupby(["model", "mode"], dropna=False):
        weighted_mean = np.average(
            group["mean_score_weighted"].fillna(0),
            weights=group["finite_count"],
        ) if group["finite_count"].sum() > 0 else np.nan
        exact_group = group[group["exact_match_rate_weighted"].notna()]
        exact_rate = np.average(
            exact_group["exact_match_rate_weighted"],
            weights=exact_group["finite_count"],
        ) if not exact_group.empty and exact_group["finite_count"].sum() > 0 else np.nan
        rows.append(
            {
                "model": model,
                "mode": mode,
                "task_count": int(task_summary[(task_summary["model"] == model) & (task_summary["mode"] == mode)]["task"].nunique()),
                "expected_count": int(group["expected_count"].sum()),
                "observed_count": int(group["observed_count"].sum()),
                "finite_count": int(group["finite_count"].sum()),
                "nan_count": int(group["nan_count"].sum()),
                "mean_score_weighted": weighted_mean,
                "exact_match_rate_weighted": exact_rate,
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "mode"]).reset_index(drop=True)


def missing_cases(latest: pd.DataFrame, task_meta: dict[str, TaskMeta], expect_all_tasks: bool) -> pd.DataFrame:
    if latest.empty:
        return pd.DataFrame(columns=["model", "mode", "tier", "task", "index"])
    rows = []
    runs = latest[["model", "mode"]].drop_duplicates().itertuples(index=False)
    for model, mode in runs:
        run_df = latest[(latest["model"] == model) & (latest["mode"] == mode)]
        tasks_to_check = set(task_meta) if expect_all_tasks else set(run_df["task"].dropna().astype(str))
        for task in sorted(tasks_to_check):
            meta = task_meta.get(task)
            if meta is None or not meta.expected_indices:
                continue
            observed = set(run_df.loc[run_df["task"] == task, "index"].dropna().astype(int).tolist())
            for index in sorted(set(meta.expected_indices) - observed):
                rows.append({"model": model, "mode": mode, "tier": meta.tier, "task": task, "index": index})
    return pd.DataFrame(rows)


def duplicate_cases(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    counts = raw.groupby(["model", "mode", "task", "index"], dropna=False).size().reset_index(name="row_count")
    return counts[counts["row_count"] > 1].sort_values(["model", "mode", "task", "index"])


def write_reports(output_dir: Path, frames: dict[str, pd.DataFrame]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)


def fmt(value: object) -> str:
    if value is None:
        return "n/a"
    try:
        if pd.isna(value):
            return "n/a"
    except Exception:
        pass
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value)


def print_summary(run_summary: pd.DataFrame, tier_summary: pd.DataFrame, task_summary: pd.DataFrame, missing: pd.DataFrame, nan_cases: pd.DataFrame, output_dir: Path | None, top_k: int) -> None:
    if run_summary.empty:
        print("No result rows found under results/.")
        return

    print("STARK result report")
    if output_dir is not None:
        print(f"  report_dir: {output_dir}")

    for _, run in run_summary.iterrows():
        model, mode = run["model"], run["mode"]
        print("\n" + "=" * 80)
        print(f"model={model} mode={mode}")
        print(
            "  cases: observed={observed}/{expected} finite={finite} nan={nan} tasks={tasks}".format(
                observed=fmt(run["observed_count"]),
                expected=fmt(run["expected_count"]),
                finite=fmt(run["finite_count"]),
                nan=fmt(run["nan_count"]),
                tasks=fmt(run["task_count"]),
            )
        )
        print(
            "  weighted_mean_score={mean} exact_match_rate={exact}".format(
                mean=fmt(run["mean_score_weighted"]),
                exact=fmt(run["exact_match_rate_weighted"]),
            )
        )

        tiers = tier_summary[(tier_summary["model"] == model) & (tier_summary["mode"] == mode)]
        if not tiers.empty:
            print("  by tier:")
            for _, tier in tiers.iterrows():
                print(
                    "    {tier}: observed={observed}/{expected} mean={mean} exact={exact} nan={nan}".format(
                        tier=tier["tier"],
                        observed=fmt(tier["observed_count"]),
                        expected=fmt(tier["expected_count"]),
                        mean=fmt(tier["mean_score_weighted"]),
                        exact=fmt(tier["exact_match_rate_weighted"]),
                        nan=fmt(tier["nan_count"]),
                    )
                )

        task_rows = task_summary[(task_summary["model"] == model) & (task_summary["mode"] == mode)]
        if not task_rows.empty and top_k > 0:
            worst = task_rows.sort_values("mean_score", ascending=False).head(top_k)
            print(f"  worst {len(worst)} tasks by mean score:")
            for _, row in worst.iterrows():
                print(
                    "    {tier}/{task}: mean={mean} count={count}/{expected} exact={exact}".format(
                        tier=row["tier"],
                        task=row["task"],
                        mean=fmt(row["mean_score"]),
                        count=fmt(row["observed_count"]),
                        expected=fmt(row["expected_count"]),
                        exact=fmt(row["exact_match_rate"]),
                    )
                )

        miss_count = len(missing[(missing["model"] == model) & (missing["mode"] == mode)]) if not missing.empty else 0
        nan_count = len(nan_cases[(nan_cases["model"] == model) & (nan_cases["mode"] == mode)]) if not nan_cases.empty else 0
        if miss_count or nan_count:
            print(f"  attention: missing_cases={miss_count} nan_cases={nan_count}")


def main() -> None:
    args = parse_args()
    models = set(args.model) if args.model else None
    modes = set(args.mode) if args.mode else None

    task_meta = load_task_meta(args.tasks_dir, args.data_dir)
    raw, skipped_files = load_results(args.results_dir, models, modes)
    latest = latest_rows(raw)
    latest = enrich_with_task_meta(latest, task_meta)

    task_summary = summarize_by_task(latest)
    tier_summary = summarize_by_tier(task_summary)
    run_summary = summarize_by_run(task_summary)
    missing = missing_cases(latest, task_meta, args.expect_all_tasks)
    duplicates = duplicate_cases(raw)
    nan_cases = latest[latest["score"].isna()].copy() if not latest.empty else pd.DataFrame()

    output_dir = None
    if not args.no_write:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = args.output_dir or (args.results_dir / "_reports" / timestamp)
        write_reports(
            output_dir,
            {
                "all_rows": raw,
                "latest_results": latest,
                "summary_by_run": run_summary,
                "summary_by_tier": tier_summary,
                "summary_by_task": task_summary,
                "missing_cases": missing,
                "nan_cases": nan_cases,
                "duplicate_cases": duplicates,
                "skipped_files": skipped_files,
            },
        )

    print_summary(run_summary, tier_summary, task_summary, missing, nan_cases, output_dir, args.top_k)


if __name__ == "__main__":
    main()
