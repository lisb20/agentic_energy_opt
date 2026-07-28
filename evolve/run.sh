#!/usr/bin/env bash
set -euo pipefail

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY before running optimization}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_DIR="$(dirname -- "$SCRIPT_DIR")"
DOMAIN="${DOMAIN:-V2G}"
case "$DOMAIN" in
  PV) INITIAL_PROGRAM="$SCRIPT_DIR/initial_program_PV.py" ;;
  V2G) INITIAL_PROGRAM="$SCRIPT_DIR/initial_program_V2G.py" ;;
  *) echo "DOMAIN must be PV or V2G" >&2; exit 2 ;;
esac

EXTRA_ARGS=()
if [[ -n "${OPENEVOLVE_ITERATIONS:-}" ]]; then
  EXTRA_ARGS+=(--iterations "$OPENEVOLVE_ITERATIONS")
fi
if [[ -n "${OPENEVOLVE_MODEL:-}" ]]; then
  EXTRA_ARGS+=(--primary-model "$OPENEVOLVE_MODEL")
fi
if [[ -n "${OPENAI_BASE_URL:-}" ]]; then
  EXTRA_ARGS+=(--api-base "$OPENAI_BASE_URL")
fi

PYTHONPATH="$RELEASE_DIR/openevolve${PYTHONPATH:+:$PYTHONPATH}" \
python "$SCRIPT_DIR/openevolve-run.py" \
  "$INITIAL_PROGRAM" \
  "$SCRIPT_DIR/evaluator.py" \
  --config "$SCRIPT_DIR/config.yaml" \
  --output "$RELEASE_DIR/output" \
  --log-level INFO \
  "${EXTRA_ARGS[@]}"
