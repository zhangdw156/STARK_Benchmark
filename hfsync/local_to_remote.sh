#!/usr/bin/env bash
set -euo pipefail

# Direction: local STARK_Benchmark artifacts -> Hugging Face bucket.
# Safe default: no remote deletion. Pass --delete only from an authoritative
# complete local copy; it deletes remote files absent locally for the selected
# artifact prefixes.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BUCKET_ID="${HF_BUCKET_ID:-zhangdw/leo-benchmark}"
REMOTE_PREFIX="${HF_STARK_PREFIX:-STARK}"
HF_CLI_STRING="${HF_CLI:-uvx hf}"
DRY_RUN=0
DELETE=0
IGNORE_EXISTING=0
SELECTED_ARTIFACTS=()
SKIPPED_ARTIFACTS=()
EXTRA_ARGS=()
DEFAULT_ARTIFACTS=(logs conv_history results)

usage() {
  cat <<'USAGE'
Usage: hfsync/local_to_remote.sh [options] [-- extra hf sync args]

Direction:
  LOCAL STARK_Benchmark artifacts -> HF bucket

Default sync pairs:
  logs/         -> hf://buckets/zhangdw/leo-benchmark/STARK/logs
  conv_history/ -> hf://buckets/zhangdw/leo-benchmark/STARK/conv_history
  results/      -> hf://buckets/zhangdw/leo-benchmark/STARK/results

Safe defaults:
  - Does not delete remote files absent locally.
  - May update same-path remote files if local files differ.
  - Use --ignore-existing / --new-only for strictly create-only behavior.

Options:
  --dry-run                 Print the sync plan without uploading.
  --delete                  Delete REMOTE files absent locally. Use carefully.
  --ignore-existing         Skip remote files that already exist; only upload new files.
  --new-only                Alias for --ignore-existing.
  --artifact NAME           Sync only this artifact; can repeat or use comma list.
                            Default: logs,conv_history,results.
                            Also supported: results_npy.
  --skip NAME               Skip an artifact; can repeat or use comma list.
  --bucket BUCKET_ID        Bucket ID. Default: zhangdw/leo-benchmark.
  --prefix PREFIX           Remote prefix inside the bucket. Default: STARK.
  -h, --help                Show this help.

Environment overrides:
  HF_BUCKET_ID              Same as --bucket.
  HF_STARK_PREFIX           Same as --prefix.
  HF_CLI                    Command used to run hf. Default: "uvx hf".

Examples:
  hfsync/local_to_remote.sh --dry-run
  hfsync/local_to_remote.sh --artifact results --dry-run
  hfsync/local_to_remote.sh --new-only
  hfsync/local_to_remote.sh --artifact results_npy
USAGE
}

artifact_path() {
  case "$1" in
    logs) printf '%s\n' 'logs' ;;
    conv_history) printf '%s\n' 'conv_history' ;;
    results) printf '%s\n' 'results' ;;
    results_npy) printf '%s\n' 'results_npy' ;;
    *) return 1 ;;
  esac
}

validate_artifact() {
  if ! artifact_path "$1" >/dev/null; then
    echo "ERROR: unknown artifact '$1'. Expected one of: logs, conv_history, results, results_npy" >&2
    exit 2
  fi
}

trim_artifact() {
  printf '%s' "$1" | tr -d '[:space:]'
}

add_selected_artifacts() {
  local raw="$1"
  local parts=()
  local part
  IFS=',' read -r -a parts <<< "$raw"
  for part in "${parts[@]}"; do
    part="$(trim_artifact "$part")"
    [[ -n "$part" ]] || continue
    validate_artifact "$part"
    SELECTED_ARTIFACTS+=("$part")
  done
}

add_skipped_artifacts() {
  local raw="$1"
  local parts=()
  local part
  IFS=',' read -r -a parts <<< "$raw"
  for part in "${parts[@]}"; do
    part="$(trim_artifact "$part")"
    [[ -n "$part" ]] || continue
    validate_artifact "$part"
    SKIPPED_ARTIFACTS+=("$part")
  done
}

contains_artifact() {
  local needle="$1"
  shift || true
  local item
  for item in "$@"; do
    [[ "$item" == "$needle" ]] && return 0
  done
  return 1
}

should_sync_artifact() {
  local artifact="$1"
  if [[ ${#SELECTED_ARTIFACTS[@]} -gt 0 ]]; then
    contains_artifact "$artifact" "${SELECTED_ARTIFACTS[@]}" || return 1
  fi
  if [[ ${#SKIPPED_ARTIFACTS[@]} -gt 0 ]]; then
    contains_artifact "$artifact" "${SKIPPED_ARTIFACTS[@]}" && return 1
  fi
  return 0
}

remote_uri_for() {
  local artifact="$1"
  local remote_subdir
  remote_subdir="$(artifact_path "$artifact")"
  if [[ -n "$REMOTE_PREFIX" ]]; then
    printf 'hf://buckets/%s/%s/%s\n' "$BUCKET_ID" "$REMOTE_PREFIX" "$remote_subdir"
  else
    printf 'hf://buckets/%s/%s\n' "$BUCKET_ID" "$remote_subdir"
  fi
}

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
    --ignore-existing|--new-only)
      IGNORE_EXISTING=1
      shift
      ;;
    --artifact|--only)
      [[ $# -ge 2 ]] || { echo "ERROR: $1 requires a value" >&2; exit 2; }
      add_selected_artifacts "$2"
      shift 2
      ;;
    --skip)
      [[ $# -ge 2 ]] || { echo "ERROR: --skip requires a value" >&2; exit 2; }
      add_skipped_artifacts "$2"
      shift 2
      ;;
    --bucket)
      [[ $# -ge 2 ]] || { echo "ERROR: --bucket requires a value" >&2; exit 2; }
      BUCKET_ID="$2"
      shift 2
      ;;
    --prefix)
      [[ $# -ge 2 ]] || { echo "ERROR: --prefix requires a value" >&2; exit 2; }
      REMOTE_PREFIX="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      echo "ERROR: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

REMOTE_PREFIX="${REMOTE_PREFIX#/}"
REMOTE_PREFIX="${REMOTE_PREFIX%/}"
read -r -a HF_CMD <<< "$HF_CLI_STRING"

synced_any=0
sync_one() {
  local artifact="$1"
  local rel_dir local_dir remote_uri
  rel_dir="$(artifact_path "$artifact")"
  local_dir="${REPO_ROOT}/${rel_dir}"
  remote_uri="$(remote_uri_for "$artifact")"

  if [[ ! -d "$local_dir" ]]; then
    echo "==> Artifact: $artifact"
    echo "Direction: LOCAL -> REMOTE"
    echo "Local:     $local_dir"
    echo "Remote:    $remote_uri"
    echo "Skip:      local directory does not exist"
    echo
    return 0
  fi

  local cmd=("${HF_CMD[@]}" buckets sync "$local_dir" "$remote_uri")
  [[ "$DELETE" -eq 1 ]] && cmd+=(--delete)
  [[ "$DRY_RUN" -eq 1 ]] && cmd+=(--dry-run)
  [[ "$IGNORE_EXISTING" -eq 1 ]] && cmd+=(--ignore-existing)
  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    cmd+=("${EXTRA_ARGS[@]}")
  fi

  echo "==> Artifact: $artifact"
  echo "Direction: LOCAL -> REMOTE"
  echo "Local:     $local_dir"
  echo "Remote:    $remote_uri"
  [[ "$DRY_RUN" -eq 1 ]] && echo "Mode:      dry-run" || echo "Mode:      apply"
  [[ "$DELETE" -eq 1 ]] && echo "Delete:    enabled (REMOTE files absent locally may be deleted)" || echo "Delete:    disabled"
  [[ "$IGNORE_EXISTING" -eq 1 ]] && echo "Existing:  skip remote-existing files" || echo "Existing:  update same-path remote files if changed"
  printf 'Command:  '
  printf ' %q' "${cmd[@]}"
  printf '\n'
  synced_any=1
  "${cmd[@]}"
  echo
}

ARTIFACTS_TO_CONSIDER=()
if [[ ${#SELECTED_ARTIFACTS[@]} -gt 0 ]]; then
  ARTIFACTS_TO_CONSIDER=("${SELECTED_ARTIFACTS[@]}")
else
  ARTIFACTS_TO_CONSIDER=("${DEFAULT_ARTIFACTS[@]}")
fi

for artifact in "${ARTIFACTS_TO_CONSIDER[@]}"; do
  should_sync_artifact "$artifact" || continue
  sync_one "$artifact"
done

if [[ "$synced_any" -eq 0 ]]; then
  echo "ERROR: no artifacts selected after applying --artifact/--skip." >&2
  exit 2
fi
