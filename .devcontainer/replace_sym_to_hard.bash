#!/bin/bash

TARGET_DIR="../.model"

find "$TARGET_DIR" -type l | while read -r symlink; do
    link_dir="$(dirname "$symlink")"
    target_rel="$(readlink "$symlink")"
    target_abs="$(realpath -m "$link_dir/$target_rel")"

    if [ -f "$target_abs" ]; then
        rel_path="$(realpath --relative-to="$link_dir" "$target_abs")"

        (
            cd "$link_dir" || exit
            rm "$(basename "$symlink")" || {
                echo "❌ rm 실패: $symlink"
                exit 1
            }
            ln "$rel_path" "$(basename "$symlink")" || echo "❌ ln 실패: $symlink → $rel_path"
        )
    fi
done

