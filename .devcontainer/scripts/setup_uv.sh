#!/usr/bin/env bash
set -e

MAX_RETRIES=10
RETRY_DELAY=1

GROUP="${1:-dev}"
WORK_DIR="${2:-/workspaces/dev}"
VENV_DIR="${WORK_DIR}/.venv"

WHEEL_DIR="${WORK_DIR}/.vendor"
mkdir -p "${WHEEL_DIR}"

cd "${WORK_DIR}"

echo "[INFO] install uv"
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "[INFO] Build pybind11 (to ${WHEEL_DIR})"
if command -v python3 &>/dev/null; then
    PYBIN="python3"
else
    echo "[ERROR] python3가 필요함." >&2
    exit 1
fi

if ls "${WHEEL_DIR}/pybind11-"*.whl >/dev/null 2>&1; then
    echo "[INFO] Existing pybind11 wheel found, skipping build."
else
    ${PYBIN} -m pip install -U pip setuptools wheel
    ${PYBIN} -m pip wheel --no-cache-dir -w "${WHEEL_DIR}" pybind11==3.0.1
fi

PYBIND_WHL="$(ls -t "${WHEEL_DIR}"/pybind11-*.whl | head -n1)"
if [[ -z "${PYBIND_WHL:-}" ]]; then
    echo "[ERROR] Failed build pybind11 wheel" >&2
    exit 1
fi
echo "[INFO] using pybind11 wheel: ${PYBIND_WHL}"

attempt=1
while [[ $attempt -le $MAX_RETRIES ]]; do
    echo "[INFO] uv sync try $attempt/$MAX_RETRIES..."
    if uv sync --group dev; then
        echo "[INFO] uv sync succeeded."
        break
    else
        echo "[INFO] uv sync failed, (busy or other error). wait for retry..."
        sleep $RETRY_DELAY
    fi
    ((attempt++))
done

if [[ $attempt -gt $MAX_RETRIES ]]; then
    echo "[ERROR] Failed uv sync" >&2
    exit 1
fi

echo "[INFO] Done setup uv."
