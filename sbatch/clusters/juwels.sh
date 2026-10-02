#!/usr/bin/env bash

case "${RL_CLUSTER_WORKLOAD:-}" in
  prime_rl)
    module load Stages/2025 CUDA/12 GCC/13.3.0
    ;;
  osworld)
    module load Stages/2026
    ;;
  "")
    echo "ERROR: RL_CLUSTER_WORKLOAD is required for JUWELS setup" >&2
    return 2 2>/dev/null || exit 2
    ;;
  *)
    echo "ERROR: unsupported RL_CLUSTER_WORKLOAD for JUWELS setup: ${RL_CLUSTER_WORKLOAD}" >&2
    return 2 2>/dev/null || exit 2
    ;;
esac

if [[ -n "${JUTIL_PROJECT:-}" ]]; then
    jutil env activate -p "$JUTIL_PROJECT"
else
    echo "JUTIL_PROJECT is unset; skipping jutil env activate" >&2
fi
module list
