#!/usr/bin/env bash
# Parallel evaluator for STARK_Benchmark against OpenAI-compatible endpoints such as vLLM.
# One sample is evaluated by one main.py process; --jobs controls how many processes run concurrently.
#
# Quick start from the repository root on the evaluation server:
#
#   uv sync
#   chmod +x scripts/run_parallel_eval.sh
#
#   scripts/run_parallel_eval.sh \
#     --model 你的_vllm_model_name \
#     --base-url http://localhost:8000/v1 \
#     --api-key EMPTY \
#     --mode text \
#     --jobs 32 \
#     --suite all
#
# Set parallelism with either:
#
#   --jobs 32
#
# or:
#
#   JOBS=32
#
# Small smoke test before the full run:
#
#   scripts/run_parallel_eval.sh \
#     --model 你的_vllm_model_name \
#     --jobs 8 \
#     --task loc_range \
#     --limit 5
#
# Generate and inspect jobs without running evaluation:
#
#   scripts/run_parallel_eval.sh \
#     --model 你的_vllm_model_name \
#     --suite tier1 \
#     --limit 2 \
#     --dry-run
#
# Per-sample logs are written to:
#
#   logs/parallel_eval/<时间戳>/
#
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_parallel_eval.sh --model MODEL [options]

Required:
  --model MODEL              Model name sent to the OpenAI-compatible endpoint.
                             Can also be set with MODEL=...

Common options:
  --base-url URL             Endpoint base URL. Default: BASE_URL or http://localhost:8000/v1
  --api-key KEY              API key. Default: API_KEY or EMPTY
  --mode text|code           Evaluation mode. Default: MODE or text
  --jobs N                   Parallel main.py processes. Default: JOBS or 8
  --suite all|tier1|tier2|tier3
                             Task suite to run. Default: all
  --task TASK                Run one task; can be repeated. Overrides --suite/--tasks-file.
  --tasks-file FILE          Read tasks from a file, one task per line.
  --ts-len N                 Forwarded to main.py for online tracking tasks. Default: TS_LEN or 10
  --noise-level LEVEL        Forwarded to main.py. Default: NOISE_LEVEL or normal
  --limit N                  Run at most N indices per task, useful for smoke tests.
  --dry-run                  Only print generated jobs; do not run evaluation.
  -h, --help                 Show this help.

Examples:
  MODEL=qwen3-4b-instruct-2507 JOBS=32 scripts/run_parallel_eval.sh --suite tier1

  scripts/run_parallel_eval.sh \
    --model qwen3-4b-instruct-2507 \
    --base-url http://localhost:8000/v1 \
    --api-key EMPTY \
    --mode text \
    --jobs 32 \
    --suite all

  scripts/run_parallel_eval.sh --model qwen3-4b-instruct-2507 --task loc_range --limit 5 --jobs 5
EOF
}

MODEL=${MODEL:-}
BASE_URL=${BASE_URL:-http://localhost:8000/v1}
API_KEY=${API_KEY:-EMPTY}
MODE=${MODE:-text}
JOBS=${JOBS:-8}
SUITE=${SUITE:-all}
TASKS_FILE=${TASKS_FILE:-}
TS_LEN=${TS_LEN:-10}
NOISE_LEVEL=${NOISE_LEVEL:-normal}
LIMIT=${LIMIT:-}
DRY_RUN=0
TASKS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL=${2:?missing value for --model}; shift 2 ;;
    --base-url) BASE_URL=${2:?missing value for --base-url}; shift 2 ;;
    --api-key) API_KEY=${2:?missing value for --api-key}; shift 2 ;;
    --mode) MODE=${2:?missing value for --mode}; shift 2 ;;
    --jobs) JOBS=${2:?missing value for --jobs}; shift 2 ;;
    --suite) SUITE=${2:?missing value for --suite}; shift 2 ;;
    --task) TASKS+=("${2:?missing value for --task}"); shift 2 ;;
    --tasks-file) TASKS_FILE=${2:?missing value for --tasks-file}; shift 2 ;;
    --ts-len) TS_LEN=${2:?missing value for --ts-len}; shift 2 ;;
    --noise-level) NOISE_LEVEL=${2:?missing value for --noise-level}; shift 2 ;;
    --limit) LIMIT=${2:?missing value for --limit}; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$MODEL" ]]; then
  echo "ERROR: --model is required, or set MODEL=..." >&2
  exit 2
fi
if ! [[ "$JOBS" =~ ^[0-9]+$ ]] || [[ "$JOBS" -lt 1 ]]; then
  echo "ERROR: --jobs must be a positive integer, got: $JOBS" >&2
  exit 2
fi
if [[ -n "$LIMIT" ]] && (! [[ "$LIMIT" =~ ^[0-9]+$ ]] || [[ "$LIMIT" -lt 1 ]]); then
  echo "ERROR: --limit must be a positive integer, got: $LIMIT" >&2
  exit 2
fi
case "$MODE" in
  text|code) ;;
  *) echo "ERROR: --mode must be text or code, got: $MODE" >&2; exit 2 ;;
esac

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is required. Install uv, then run: uv sync" >&2
  exit 2
fi

if [[ ! -f main.py || ! -d data || ! -d tasks ]]; then
  echo "ERROR: run this script from the STARK_Benchmark repository root after extracting data/" >&2
  exit 2
fi

# Return the data directory used by main.py for a task.
task_data_dir() {
  local task="$1"
  case "$task" in
    loc_range) echo "data/localization_range_0.01" ;;
    loc_bearing) echo "data/localization_bearing_0.01" ;;
    loc_range_bearing) echo "data/localization_range_bearing_[0.01, 0.01]" ;;
    loc_region) echo "data/localization_region_4.0" ;;
    loc_event_temp) echo "data/localization_event_temp_0.01" ;;
    loc_event_spatio) echo "data/localization_event_spatio_0.01" ;;
    loc_event_spatio_temp) echo "data/localization_event_spatio_temp_0.01" ;;
    track_range_online) echo "data/tracking_range_0.01" ;;
    track_bearing_online) echo "data/tracking_bearing_0.01" ;;
    track_range_bearing_online) echo "data/tracking_range_bearing_[0.01, 0.01]" ;;
    track_region_online) echo "data/tracking_region_4.0" ;;
    track_event_spatio_online) echo "data/tracking_event_spatio_0.01" ;;
    track_event_temp_online) echo "data/tracking_event_temp_0.01" ;;
    track_event_spatiotemp_online) echo "data/tracking_event_spatio_temp_0.01" ;;
    spatial_impute|temporal_impute|spatiotemporal_forecast|spatiotemporal_impute)
      echo "data/$task" ;;
    *)
      local d
      for d in \
        "data/tier2_v2_spatial/$task" \
        "data/tier2_v2_temporal/$task" \
        "data/tier2_v2_spatial_temporal/$task" \
        "data/tier3_world_knowledge/$task"
      do
        if [[ -d "$d" ]]; then
          echo "$d"
          return 0
        fi
      done
      return 1
      ;;
  esac
}

# Print available indices for a task, one per line.
task_indices() {
  local task="$1"
  local dir="$2"
  if [[ "$task" == loc_* || "$task" == track_* ]]; then
    find "$dir" -maxdepth 1 -type f -name '*_object_location.csv' -print \
      | sed -E 's#.*/([0-9]+)_object_location\.csv#\1#' \
      | sort -n
  else
    find "$dir" -maxdepth 1 -type f -name '*.csv' -print \
      | sed -E 's#.*/([0-9]+)\.csv#\1#' \
      | sort -n
  fi
}

collect_tasks() {
  if [[ "${#TASKS[@]}" -gt 0 ]]; then
    printf '%s\n' "${TASKS[@]}"
    return 0
  fi

  if [[ -n "$TASKS_FILE" ]]; then
    if [[ ! -f "$TASKS_FILE" ]]; then
      echo "ERROR: tasks file not found: $TASKS_FILE" >&2
      exit 2
    fi
    grep -Ev '^[[:space:]]*($|#)' "$TASKS_FILE"
    return 0
  fi

  case "$SUITE" in
    all) cat tasks/tier1.txt tasks/tier2.txt tasks/tier3.txt ;;
    tier1) cat tasks/tier1.txt ;;
    tier2) cat tasks/tier2.txt ;;
    tier3) cat tasks/tier3.txt ;;
    *) echo "ERROR: --suite must be all, tier1, tier2, or tier3; got: $SUITE" >&2; exit 2 ;;
  esac
}

RUN_ID=$(date '+%Y%m%d_%H%M%S')
LOG_ROOT="logs/parallel_eval/${RUN_ID}"
STATUS_ROOT="${LOG_ROOT}/status"
JOBS_FILE="${LOG_ROOT}/jobs.tsv"
mkdir -p "$LOG_ROOT" "$STATUS_ROOT"

# Pre-create directories touched by parallel workers to avoid startup races.
mkdir -p \
  "results/$MODEL/$MODE" \
  "results_npy/$MODEL/$MODE" \
  "conv_history/$MODEL/$MODE"

: > "$JOBS_FILE"
while IFS= read -r task; do
  [[ -z "$task" ]] && continue
  if ! dir=$(task_data_dir "$task"); then
    echo "WARN: skip task without known data directory: $task" >&2
    continue
  fi
  if [[ ! -d "$dir" ]]; then
    echo "WARN: skip task because data directory is missing: $task -> $dir" >&2
    continue
  fi

  count=0
  while IFS= read -r idx; do
    [[ -z "$idx" ]] && continue
    printf '%s %s\n' "$task" "$idx" >> "$JOBS_FILE"
    count=$((count + 1))
    if [[ -n "$LIMIT" && "$count" -ge "$LIMIT" ]]; then
      break
    fi
  done < <(task_indices "$task" "$dir")

  if [[ "$count" -eq 0 ]]; then
    echo "WARN: no indices found for task: $task -> $dir" >&2
  fi
done < <(collect_tasks)

TOTAL_JOBS=$(wc -l < "$JOBS_FILE" | tr -d '[:space:]')
if [[ "$TOTAL_JOBS" -eq 0 ]]; then
  echo "ERROR: no jobs generated" >&2
  exit 2
fi

cat <<EOF
STARK parallel evaluation
  model      : $MODEL
  base_url   : $BASE_URL
  mode       : $MODE
  jobs       : $JOBS
  total jobs : $TOTAL_JOBS
  jobs file  : $JOBS_FILE
  logs dir   : $LOG_ROOT
EOF

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run jobs:"
  cat "$JOBS_FILE"
  exit 0
fi

if ! uv run --no-sync python -c "import openai, scipy, numpy, transformers, shapely, pandas, httpx, filterpy" >/dev/null 2>&1; then
  echo "ERROR: uv environment is not ready. Run 'uv sync' from the repository root first." >&2
  exit 2
fi

export MODEL BASE_URL API_KEY MODE TS_LEN NOISE_LEVEL LOG_ROOT STATUS_ROOT

set +e
xargs -n 2 -P "$JOBS" bash -c '
  set -u
  task="$1"
  idx="$2"
  log_file="${LOG_ROOT}/${task}__${idx}.log"
  status_file="${STATUS_ROOT}/${task}__${idx}.status"

  echo "START task=${task} index=${idx}"
  if uv run --no-sync python main.py \
      --openai "$MODEL" \
      --base_url "$BASE_URL" \
      --api_key "$API_KEY" \
      --dataset "$task" \
      --index "$idx" \
      --mode "$MODE" \
      --ts_len "$TS_LEN" \
      --noise_level "$NOISE_LEVEL" \
      > "$log_file" 2>&1; then
    echo "OK task=${task} index=${idx} log=${log_file}"
    echo "OK" > "$status_file"
  else
    code=$?
    echo "FAIL task=${task} index=${idx} exit=${code} log=${log_file}" >&2
    echo "FAIL ${code}" > "$status_file"
    exit "$code"
  fi
' _ < "$JOBS_FILE"
XARGS_STATUS=$?
set -e

OK_COUNT=$(find "$STATUS_ROOT" -type f -exec grep -l '^OK' {} \; | wc -l | tr -d '[:space:]')
FAIL_COUNT=$(find "$STATUS_ROOT" -type f -exec grep -l '^FAIL' {} \; | wc -l | tr -d '[:space:]')

cat <<EOF
Finished STARK parallel evaluation
  ok       : $OK_COUNT
  failed   : $FAIL_COUNT
  expected : $TOTAL_JOBS
  logs dir : $LOG_ROOT
EOF

if [[ "$FAIL_COUNT" -gt 0 ]]; then
  echo "Failed jobs:"
  find "$STATUS_ROOT" -type f -exec grep -l '^FAIL' {} \; \
    | sed -E "s#^${STATUS_ROOT}/##; s#\.status$##" \
    | sort
fi

exit "$XARGS_STATUS"
