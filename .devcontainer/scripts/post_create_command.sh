#!/usr/bin/env bash
set -euo pipefail

# args: <username>
if [[ $# -lt 1 || -z "${1:-}" ]]; then
    echo "Usage: $0 <username>" >&2; exit 1
fi

USER_NAME="$1"
if ! id "$USER_NAME" &>/dev/null; then
    echo "Error: user '$USER_NAME' does not exist." >&2; exit 1
fi

OWNER="${USER_NAME}:${USER_NAME}"

# 1) 타깃 경로
declare -a TARGETS=(
    "/workspaces/dev"
    "/home/${USER_NAME}/.cache"
    "/workspaces/dev/.datasets/asr-rankformer-datasets"
)

# 2) 타깃별 제외(:로 구분; 공백 포함 경로 안전)
declare -A EXCLUDES
EXCLUDES["/workspaces/dev"]="/workspaces/dev/.datasets"
EXCLUDES["/home/${USER_NAME}/.cache"]=""
EXCLUDES["/workspaces/dev/.datasets/asr-rankformer-datasets"]=""

build_prune_args() {
    local target="$1"
    local excludes_str="${EXCLUDES[$target]-}"

    [[ -z "$excludes_str" ]] && return 0

    local IFS=':'
    local -a ex_arr
    read -r -a ex_arr <<< "$excludes_str"
    ((${#ex_arr[@]}==0)) && return 0

    local -a out
    out+=( \( )
    local first=1
    for e in "${ex_arr[@]}"; do
        [[ -z "$e" ]] && continue
        if (( first )); then
            out+=( -path "$e" )
            first=0
        else
            out+=( -o -path "$e" )
        fi
    done
    out+=( \) -prune -o )

    printf '%s\0' "${out[@]}"
}

chown_with_excludes() {
    local target="$1" owner="$2"

    local -a args
    args+=( "$target" -mindepth 0 )

    # prune 절 안전하게 읽기 (NUL 구분)
    local -a prune=()
    mapfile -d '' -t prune < <(build_prune_args "$target" || true)
    (( ${#prune[@]} )) && args+=( "${prune[@]}" )

    args+=( \( -not -user "$USER_NAME" -o -not -group "$USER_NAME" \) )

    if (( EUID == 0 )); then
        args+=( -exec chown "$owner" {} + )
    else
        args+=( -exec sudo chown "$owner" {} + )
    fi

    sudo find "${args[@]}" || true
}


for t in "${TARGETS[@]}"; do
    [[ -e "$t" ]] && chown_with_excludes "$t" "$OWNER"
done

# 후속 작업
bash /workspaces/dev/.devcontainer/scripts/set_uv.sh
bash /workspaces/dev/.devcontainer/scripts/set_lhotse.sh
