#!/usr/bin/env bash

SLURM_INTERACT_PORT="$(
  set -e
  set +u
  source /etc/profile.d/slurm.sh >&2
  printf '%s' "${SLURM_INTERACT_PORT:?SLURM_INTERACT_PORT was not assigned}"
)"

export SLURM_INTERACT_PORT
export OSWORLD_GATEWAY_PORT="$SLURM_INTERACT_PORT"
