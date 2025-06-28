#!/bin/bash

TARGET_DIRS=(
    "/root/.cache/hugging"
    "/root/.cache/torch"
    # "../.model"
)

for DIR in "${TARGET_DIRS[@]}"; do
    [ -d "$DIR" ] || {
        echo "❗ 디렉터리 없음: $DIR"
        continue
    }

    # 공백·특수문자 안전 처리
    find "$DIR" -type l -print0 | while IFS= read -r -d '' symlink; do
        link_dir="$(dirname -- "$symlink")"
        target_rel="$(readlink -- "$symlink")"
        target_abs="$(realpath -m -- "$link_dir/$target_rel")"

        [[ -f $target_abs ]] || continue

        rel_path="$(realpath --relative-to="$link_dir" -- "$target_abs")"

        (
            cd "$link_dir" || exit
            rm -- "$(basename -- "$symlink")" || {
                echo "❌ rm 실패: $symlink"
                exit 1
            }
            ln -- "$rel_path" "$(basename -- "$symlink")" || {
                echo "❌ ln 실패: $symlink → $rel_path"
            }
        )
    done
done
