#!/usr/bin/env bash
set -euo pipefail

# usage
usage() {
    echo "Usage: $0 <username> [--target <path[:exclude1:exclude2:...]> ...]" >&2
    exit 1
}

[[ $# -lt 1 ]] && usage


USER_NAME="$1"
shift

if ! id "$USER_NAME" &>/dev/null; then
    echo "Error: user '$USER_NAME' does not exist." >&2
    exit 1
fi

OWNER="${USER_NAME}:${USER_NAME}"

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

    args+=( \( -not -user "$USER_NAME" -o -not -group "$USER_NAME" \) )

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
