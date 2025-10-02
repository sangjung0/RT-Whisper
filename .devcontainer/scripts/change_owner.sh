#!/usr/bin/env bash
set -euo pipefail

# usage
usage() {
    echo "Usage: $0 <username> [--target <path[:exclude1:exclude2:...]> ...]" >&2
    exit 1
}

# args: <username> [--target <path[:exclude1:exclude2:...]> ...]
if [[ $# -lt 1 ]]; then
    usage
fi

CONTAINER_USER="$1"
shift

if ! id "$CONTAINER_USER" &>/dev/null; then
    echo "Error: user '$CONTAINER_USER' does not exist." >&2
    exit 1
fi

if [[ "$CONTAINER_USER" == "root" ]]; then
    echo "[INFO] root user detected → skipping ownership change"
    exit 0
fi

OWNER="${CONTAINER_USER}:${CONTAINER_USER}"

declare -a TARGETS=()
declare -A EXCLUDES

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target)
            [[ $# -lt 2 ]] && usage
            arg="$2"
            shift 2
            tgt="${arg%%:*}"
            exc="${arg#*:}"
            TARGETS+=("$tgt")
            [[ "$exc" == "$tgt" ]] && exc=""
            EXCLUDES["$tgt"]="$exc"
            ;;
        *)
            usage
            ;;
    esac
done

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

    local -a prune=()
    mapfile -d '' -t prune < <(build_prune_args "$target" || true)
    (( ${#prune[@]} )) && args+=( "${prune[@]}" )

    args+=( \( -not -user "$CONTAINER_USER" -o -not -group "$CONTAINER_USER" \) )

    if (( EUID == 0 )); then
        args+=( -exec chown "$owner" {} + )
        find "${args[@]}" || true
    else
        args+=( -exec sudo chown "$owner" {} + )
        sudo find "${args[@]}" || true
    fi

}


for t in "${TARGETS[@]}"; do
    [[ -e "$t" ]] && chown_with_excludes "$t" "$OWNER"
done

echo "[INFO] Ownership change completed."
