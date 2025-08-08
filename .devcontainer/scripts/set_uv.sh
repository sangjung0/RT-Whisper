#!/usr/bin/env bash
set -e

WORK_DIR="/workspaces/dev/"
cd "${WORK_DIR}"

VENV_DIR="${WORK_DIR}.venv"
MAX_RETRIES=10
RETRY_DELAY=1

attempt=1
while [[ $attempt -le $MAX_RETRIES ]]; do
    echo "uv sync 시도 $attempt/$MAX_RETRIES..."
    if uv sync --group dev; then
        echo "uv sync 성공"
        break
    else
        echo "uv sync 실패 (busy or other error). 재시도 대기 중..."
        sleep $RETRY_DELAY
    fi
    ((attempt++))
done

if [[ $attempt -gt $MAX_RETRIES ]]; then
    echo "오류: uv sync가 $MAX_RETRIES회 연속 실패함. 수동 확인 필요." >&2
    exit 1
fi
