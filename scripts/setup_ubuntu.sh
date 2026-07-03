#!/usr/bin/env bash
# Bootstrap DroneSim (EVRP + TRSP) on a fresh Ubuntu machine.
#
# Installs: system build tools, Miniconda (if missing), a conda env with
# Python 3.10, SUMO (via the self-contained eclipse-sumo pip wheel — no
# apt/PPA needed), torch (CPU or CUDA build), and every pip dependency
# under environment/requirements.txt. Gurobi's Python bindings are
# installed too, but the license itself is account-specific and must be
# obtained manually — see docs/SETUP.md.
#
# Usage:
#   scripts/setup_ubuntu.sh [--env-name NAME] [--cuda cu128|cu124|cu121|cpu] [--notebooks]
#
# Safe to re-run: every step is idempotent.

set -euo pipefail

ENV_NAME="dronesim"
CUDA_CHOICE="auto"
WITH_NOTEBOOKS=0
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name) ENV_NAME="$2"; shift 2 ;;
    --cuda) CUDA_CHOICE="$2"; shift 2 ;;
    --notebooks) WITH_NOTEBOOKS=1; shift ;;
    -h|--help)
      grep '^#' "$0" | sed -E 's/^# ?//'
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

log() { printf '\n\033[1;32m==>\033[0m %s\n' "$1"; }

if [[ ! -f /etc/os-release ]] || ! grep -qi ubuntu /etc/os-release; then
  echo "Warning: this script targets Ubuntu; unrecognized OS, continuing anyway." >&2
fi

# ---------------------------------------------------------------------------
# 1. System build tools
# ---------------------------------------------------------------------------
log "Installing system packages (sudo required)"
sudo apt-get update -y
sudo apt-get install -y \
  build-essential cmake git curl wget ca-certificates \
  libgl1 libglib2.0-0

# ---------------------------------------------------------------------------
# 2. Miniconda (skip if conda already available)
# ---------------------------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
  log "conda not found — installing Miniconda to \$HOME/miniconda3"
  MINICONDA_SH="/tmp/miniconda_installer.sh"
  curl -fsSL -o "$MINICONDA_SH" \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash "$MINICONDA_SH" -b -p "$HOME/miniconda3"
  rm -f "$MINICONDA_SH"
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi

# ---------------------------------------------------------------------------
# 3. Conda env
# ---------------------------------------------------------------------------
if conda env list | grep -qE "^\s*${ENV_NAME}\s"; then
  log "Conda env '${ENV_NAME}' already exists — reusing it"
else
  log "Creating conda env '${ENV_NAME}' (python=3.10)"
  conda create -y -n "$ENV_NAME" python=3.10
fi
conda activate "$ENV_NAME"

python -m pip install --upgrade pip

# ---------------------------------------------------------------------------
# 4. torch — CPU or CUDA build
# ---------------------------------------------------------------------------
if [[ "$CUDA_CHOICE" == "auto" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    DRIVER_VER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)"
    log "NVIDIA driver detected (version ${DRIVER_VER}) — defaulting to cu128"
    CUDA_CHOICE="cu128"
  else
    log "No NVIDIA GPU detected — installing CPU-only torch"
    CUDA_CHOICE="cpu"
  fi
fi

log "Installing torch (${CUDA_CHOICE})"
if [[ "$CUDA_CHOICE" == "cpu" ]]; then
  python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
else
  python -m pip install torch --index-url "https://download.pytorch.org/whl/${CUDA_CHOICE}"
fi

# ---------------------------------------------------------------------------
# 5. Project dependencies
# ---------------------------------------------------------------------------
log "Installing pip requirements"
if [[ "$WITH_NOTEBOOKS" -eq 1 ]]; then
  python -m pip install -r "$REPO_ROOT/environment/requirements-notebooks.txt"
else
  python -m pip install -r "$REPO_ROOT/environment/requirements.txt"
fi

# ---------------------------------------------------------------------------
# 6. Wire SUMO_HOME to the pip-installed SUMO tree, activated automatically
#    whenever this conda env is activated.
# ---------------------------------------------------------------------------
log "Configuring SUMO_HOME for env '${ENV_NAME}'"
SUMO_PKG_DIR="$(python -c 'import sumo, os; print(os.path.dirname(sumo.__file__))')"
CONDA_ENV_DIR="$(conda info --base)/envs/${ENV_NAME}"
ACTIVATE_DIR="${CONDA_ENV_DIR}/etc/conda/activate.d"
DEACTIVATE_DIR="${CONDA_ENV_DIR}/etc/conda/deactivate.d"
mkdir -p "$ACTIVATE_DIR" "$DEACTIVATE_DIR"

cat > "${ACTIVATE_DIR}/dronesim_env_vars.sh" <<EOF
export SUMO_HOME_OLD="\${SUMO_HOME:-}"
export SUMO_HOME="${SUMO_PKG_DIR}"
EOF

cat > "${DEACTIVATE_DIR}/dronesim_env_vars.sh" <<'EOF'
export SUMO_HOME="$SUMO_HOME_OLD"
unset SUMO_HOME_OLD
EOF

export SUMO_HOME="$SUMO_PKG_DIR"

# ---------------------------------------------------------------------------
# 7. Gurobi — bindings installed via pip already; license is manual.
# ---------------------------------------------------------------------------
log "Gurobi Python bindings installed. A license is still required for the"
log "MILP planner (gurobipy 13.0.1 supports node-locked, floating, and WLS"
log "licenses). See docs/SETUP.md for how to obtain and install one."

# ---------------------------------------------------------------------------
# 8. Smoke test
# ---------------------------------------------------------------------------
log "Running smoke test"
python "$REPO_ROOT/scripts/smoke_test_env.py" || \
  echo "Smoke test reported issues — see docs/SETUP.md for troubleshooting." >&2

log "Done. Activate with: conda activate ${ENV_NAME}"
