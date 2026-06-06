#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: ./sync_logs_to_hf_bucket.sh [OPTIONS]

Sync this repository's benchmark artifacts to a Hugging Face bucket.

Default sync pairs:
  logs/         -> hf://buckets/zhangdw/leo-benchmark/STARK/logs/
  conv_history/ -> hf://buckets/zhangdw/leo-benchmark/STARK/conv_history/

Delete behavior: remote files that are absent locally are not deleted unless
--delete is passed.

Options:
  --dry-run             Print the sync plan without uploading.
  --delete              Delete remote files that are absent locally.
  --bucket BUCKET_ID    Bucket ID, e.g. zhangdw/leo-benchmark.
  --stark-prefix PREFIX Remote STARK prefix inside the bucket. Default: STARK.
  --logs-dir DIR        Local logs directory. Default: logs.
  --conv-dir DIR        Local conversation-history directory. Default: conv_history.
  --skip-logs           Do not sync logs/.
  --skip-conv-history   Do not sync conv_history/.
  -h, --help            Show this help.

Environment overrides:
  HF_BUCKET_ID          Same as --bucket.
  HF_STARK_PREFIX       Same as --stark-prefix.
  HF_LOGS_DIR           Same as --logs-dir.
  HF_CONV_HISTORY_DIR   Same as --conv-dir.
  HF_CLI                Command used to run hf. Default: "uvx hf".
USAGE
}

BUCKET_ID="${HF_BUCKET_ID:-zhangdw/leo-benchmark}"
STARK_PREFIX="${HF_STARK_PREFIX:-STARK}"
LOGS_DIR="${HF_LOGS_DIR:-logs}"
CONV_HISTORY_DIR="${HF_CONV_HISTORY_DIR:-conv_history}"
HF_CLI_STRING="${HF_CLI:-uvx hf}"
DRY_RUN=0
DELETE=0
SYNC_LOGS=1
SYNC_CONV_HISTORY=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --delete)
      DELETE=1
      shift
      ;;
    --bucket)
      [[ $# -ge 2 ]] || { echo "ERROR: --bucket requires a value" >&2; exit 2; }
      BUCKET_ID="$2"
      shift 2
      ;;
    --stark-prefix)
      [[ $# -ge 2 ]] || { echo "ERROR: --stark-prefix requires a value" >&2; exit 2; }
      STARK_PREFIX="$2"
      shift 2
      ;;
    --logs-dir)
      [[ $# -ge 2 ]] || { echo "ERROR: --logs-dir requires a value" >&2; exit 2; }
      LOGS_DIR="$2"
      shift 2
      ;;
    --conv-dir)
      [[ $# -ge 2 ]] || { echo "ERROR: --conv-dir requires a value" >&2; exit 2; }
      CONV_HISTORY_DIR="$2"
      shift 2
      ;;
    --skip-logs)
      SYNC_LOGS=0
      shift
      ;;
    --skip-conv-history)
      SYNC_CONV_HISTORY=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$SYNC_LOGS" -eq 0 && "$SYNC_CONV_HISTORY" -eq 0 ]]; then
  echo "ERROR: nothing to sync; both --skip-logs and --skip-conv-history were passed" >&2
  exit 2
fi

STARK_PREFIX="${STARK_PREFIX#/}"
STARK_PREFIX="${STARK_PREFIX%/}"

# Split HF_CLI on shell words so the default "uvx hf" works without eval.
read -r -a HF_CMD <<< "$HF_CLI_STRING"

sync_one() {
  local local_dir="$1"
  local remote_subdir="$2"

  if [[ ! -d "$local_dir" ]]; then
    echo "ERROR: local directory not found: $local_dir" >&2
    exit 1
  fi

  local remote_uri="hf://buckets/${BUCKET_ID}/${STARK_PREFIX}/${remote_subdir}"
  local sync_args=("buckets" "sync" "$local_dir" "$remote_uri")
  if [[ "$DELETE" -eq 1 ]]; then
    sync_args+=("--delete")
  fi
  if [[ "$DRY_RUN" -eq 1 ]]; then
    sync_args+=("--dry-run")
  fi

  echo "Local:  $local_dir"
  echo "Remote: $remote_uri"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "Mode:   dry-run"
  else
    echo "Mode:   apply"
  fi
  if [[ "$DELETE" -eq 1 ]]; then
    echo "Delete: enabled"
  else
    echo "Delete: disabled"
  fi

  "${HF_CMD[@]}" "${sync_args[@]}"
}

if [[ "$SYNC_LOGS" -eq 1 ]]; then
  sync_one "$LOGS_DIR" "logs"
fi

if [[ "$SYNC_CONV_HISTORY" -eq 1 ]]; then
  sync_one "$CONV_HISTORY_DIR" "conv_history"
fi
