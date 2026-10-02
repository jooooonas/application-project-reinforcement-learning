#!/usr/bin/env bash
#SBATCH --job-name=osworld_run
#SBATCH --output=slurm-%x.%j.out
#SBATCH --error=slurm-%x.%j.out

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -P)}}"
cd "$ROOT_DIR"

if [[ "${SLURM_HET_SIZE:-0}" -ne 2 ]]; then
  echo "ERROR: run_osworld.sbatch requires exactly two heterogeneous components" >&2
  exit 2
fi
if [[ -z "${SLURM_JOB_NUM_NODES_HET_GROUP_0:-}" ]]; then
  echo "ERROR: Slurm did not expose fleet component node metadata" >&2
  exit 2
fi
if [[ -z "${SLURM_JOB_NUM_NODES_HET_GROUP_1:-}" ]]; then
  echo "ERROR: Slurm did not expose PrimeRL component node metadata" >&2
  exit 2
fi

PYTHON="${PYTHON:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: repository Python is missing: $PYTHON" >&2
  exit 127
fi

export ROOT_DIR PYTHON
export OSWORLD_RUN_ID="${OSWORLD_RUN_ID:-${SLURM_JOB_ID%%+*}}"
export OSWORLD_FLEET_RUN_ID="${OSWORLD_FLEET_RUN_ID:-$OSWORLD_RUN_ID}"
export OSWORLD_FLEET_ALLOCATED_NODES="${OSWORLD_FLEET_ALLOCATED_NODES:-$SLURM_JOB_NUM_NODES_HET_GROUP_0}"

if [[ "$OSWORLD_FLEET_ALLOCATED_NODES" -ne "$SLURM_JOB_NUM_NODES_HET_GROUP_0" ]]; then
  echo "ERROR: fleet allocation has $SLURM_JOB_NUM_NODES_HET_GROUP_0 nodes; expected $OSWORLD_FLEET_ALLOCATED_NODES" >&2
  exit 2
fi
if [[ "${OSWORLD_RUN_PRIME_NODES:?OSWORLD_RUN_PRIME_NODES must be set}" -ne "$SLURM_JOB_NUM_NODES_HET_GROUP_1" ]]; then
  echo "ERROR: PrimeRL allocation has $SLURM_JOB_NUM_NODES_HET_GROUP_1 nodes; expected $OSWORLD_RUN_PRIME_NODES" >&2
  exit 2
fi

RUN_ROOT="$OSWORLD_RUN_BASE/$OSWORLD_RUN_ID"
export OSWORLD_ENV_FLEET_REGISTRY="${OSWORLD_ENV_FLEET_REGISTRY:-$RUN_ROOT/env_fleet/env_registry.json}"
export OSWORLD_PRIME_RL_CONFIG_PATH="${OSWORLD_PRIME_RL_CONFIG_PATH:-$RUN_ROOT/prime_rl_fleet.toml}"
export OSWORLD_PRIME_RL_OUTPUT_DIR="${OSWORLD_PRIME_RL_OUTPUT_DIR:-$RUN_ROOT/prime_rl}"

CHILD_PIDS=()
terminate_children() {
  local pid
  for pid in "${CHILD_PIDS[@]}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
}
cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM
  terminate_children
  local pid
  for pid in "${CHILD_PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  exit "$exit_code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_for_first() {
  WAITED_PID=""
  WAITED_EXIT_CODE=0
  set +e
  wait -n -p WAITED_PID "$@"
  WAITED_EXIT_CODE=$?
  set -e
}

echo "Combined OSWorld run started: $(date --iso-8601=seconds)"
echo "Slurm job id: $SLURM_JOB_ID"
echo "Fleet nodes: $SLURM_JOB_NUM_NODES_HET_GROUP_0"
echo "PrimeRL nodes: $SLURM_JOB_NUM_NODES_HET_GROUP_1"
echo "Run id: $OSWORLD_RUN_ID"
echo "Run root: $RUN_ROOT"

"$PYTHON" scripts/osworld_fleet.py run --het-group 0 &
FLEET_PID=$!
CHILD_PIDS+=("$FLEET_PID")

"$PYTHON" scripts/wait_env_fleet_ready.py \
  --registry "$OSWORLD_ENV_FLEET_REGISTRY" \
  --timeout-s "${OSWORLD_ENV_FLEET_READY_TIMEOUT:-3600}" \
  --verbose &
READINESS_PID=$!
CHILD_PIDS+=("$READINESS_PID")

wait_for_first "$FLEET_PID" "$READINESS_PID"
if [[ "$WAITED_PID" == "$FLEET_PID" ]]; then
  echo "ERROR: OSWorld fleet stopped before becoming ready (exit $WAITED_EXIT_CODE)" >&2
  if [[ "$WAITED_EXIT_CODE" -eq 0 ]]; then
    exit 1
  fi
  exit "$WAITED_EXIT_CODE"
fi
if [[ "$WAITED_EXIT_CODE" -ne 0 ]]; then
  echo "ERROR: OSWorld fleet readiness failed (exit $WAITED_EXIT_CODE)" >&2
  exit "$WAITED_EXIT_CODE"
fi
echo "OSWorld fleet is ready: $(date --iso-8601=seconds)"

"$PYTHON" scripts/render_prime_rl_fleet_config.py \
  --base-config "${OSWORLD_PRIME_BASE_CONFIG:?OSWORLD_PRIME_BASE_CONFIG must be set}" \
  --registry "$OSWORLD_ENV_FLEET_REGISTRY" \
  --output "$OSWORLD_PRIME_RL_CONFIG_PATH" \
  --output-dir "$OSWORLD_PRIME_RL_OUTPUT_DIR" \
  --inflight-per-worker "${OSWORLD_PRIME_INFLIGHT_PER_WORKER:-1}"

PRIME_PREPARE_ARGS=(
  @ "$OSWORLD_PRIME_RL_CONFIG_PATH"
  --dry-run
)
if [[ "${OSWORLD_RUN_CLEAN_OUTPUT_DIR:-1}" == "1" ]]; then
  PRIME_PREPARE_ARGS+=(--clean-output-dir)
fi
"$PYTHON" scripts/prime_rl.py "${PRIME_PREPARE_ARGS[@]}"

PRIME_SCRIPT="$OSWORLD_PRIME_RL_OUTPUT_DIR/rl.sbatch"
"$PYTHON" scripts/prime_rl.py run \
  --script "$PRIME_SCRIPT" \
  --het-group 1 \
  --expected-nodes "$OSWORLD_RUN_PRIME_NODES" &
PRIME_PID=$!
CHILD_PIDS+=("$PRIME_PID")

wait_for_first "$FLEET_PID" "$PRIME_PID"
if [[ "$WAITED_PID" == "$FLEET_PID" ]]; then
  echo "ERROR: OSWorld fleet stopped while PrimeRL was running (exit $WAITED_EXIT_CODE)" >&2
  if [[ "$WAITED_EXIT_CODE" -eq 0 ]]; then
    exit 1
  fi
  exit "$WAITED_EXIT_CODE"
fi

echo "PrimeRL stopped with exit code $WAITED_EXIT_CODE; stopping the fleet"
exit "$WAITED_EXIT_CODE"
