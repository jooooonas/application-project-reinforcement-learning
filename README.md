# Reinforcement Learning Pipeline

This repository contains a cluster-native reinforcement-learning pipeline built
around Verifiers, PrimeRL, and OSWorld computer-use tasks.

This is a snapshot of the application project code as of 30 July 2026.
Generated container images and VM disks are not included.

## Quick start

Prerequisites: Python 3.12, `git`, `uv`, and the `hf` command from the
Hugging Face CLI.

### 1. Clone the repository

```bash
git -c 'url.https://github.com/.insteadOf=git@github.com:' clone --recurse-submodules https://github.com/jooooonas/application-project-reinforcement-learning.git
cd application-project-reinforcement-learning
```

PrimeRL and OSWorldRL are pinned submodules and do not need to be cloned
separately. The URL setting also fetches their nested public submodules over
HTTPS.

To build the excluded `apptainer/images/osworld.sif`, run the
`build-osworld-sif` GitHub Actions workflow and install its artifact with
`scripts/install_osworld_sif_artifact.py`.

### 2. Install the Python environments

The root project, PrimeRL, and OSWorldRL use separate environments:

```bash
uv sync --locked
UV_PROJECT_ENVIRONMENT="$PWD/deps/prime-rl/.venv" \
  uv sync --project deps/prime-rl --locked --extra all
uv sync --project deps/OSWorldRL --locked
```

### 3. Download the OSWorld VM

The default layout places the VM beside the repository. Download the ZIP archive
and extract `Ubuntu.qcow2` from it:

```bash
mkdir -p ../osworld_deployment
hf download xlangai/ubuntu_osworld Ubuntu.qcow2.zip \
  --repo-type dataset \
  --local-dir ../osworld_deployment
unzip -o ../osworld_deployment/Ubuntu.qcow2.zip \
  -d ../osworld_deployment
```

If you already have the VM locally, set `OSWORLD_QCOW_PATH` to its location.

### 4. Configure the runtime

Create the ignored, machine-local runtime file and adapt it to your environment:

```bash
cp .env.example .env
```

Most paths already default to the repository layout. Review these user-specific
settings:

- `SCRATCH`
- `OSWORLD_QCOW_PATH`
- `PRIME_RL_SLURM_ACCOUNT`
- `PRIME_RL_SLURM_PARTITION`
- `OSWORLD_FLEET_SLURM_PARTITION`

PrimeRL and OSWorld fleet resource requests are controlled by the corresponding
`PRIME_RL_*` and `OSWORLD_FLEET_SLURM_*` settings. If you are unsure about the
remaining values, ask an LLM to check `.env` against your repository and runtime
locations. See [`.env.example`](.env.example) for all settings and defaults.

### 5. Verify the setup

Run the test suite and preview the fleet submission without starting a job:

```bash
uv run --no-sync python -m pytest
uv run --no-sync python scripts/osworld_fleet.py submit --dry-run
```

## Run PrimeRL with OSWorld

A run consists of 2 components: The environment fleet and the prime-rl job which contains our trainer. To submit both components as one heterogeneous Slurm job, run:

```bash
uv run --no-sync python scripts/osworld_run.py submit
```

The components remain usable independently. For fleet development or a
standalone service, use `scripts/osworld_fleet.py submit`; for a manually
rendered PrimeRL config, use `scripts/prime_rl.py`.

## Development

`pre-commit` is installed by `uv sync`. Register the hooks once after cloning;
they run style and typing checks before each commit:

```bash
uv run --no-sync pre-commit install
```
