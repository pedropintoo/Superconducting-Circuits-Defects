#!/bin/bash
# Usage: ./count-used-images.sh ../train_val_dataset

set -euo pipefail

base_dir="${1:-$(pwd)}"
label_dirs=("$base_dir/labels/train" "$base_dir/labels/val")

declare -A prefix_numbers

shopt -s nullglob
for dir in "${label_dirs[@]}"; do
    [[ -d "$dir" ]] || continue
    for file in "$dir"/*-[0-9][0-9][0-9][0-9][0-9][0-9].txt; do
        prefix="${file##*/}"
        prefix="${prefix%-[0-9][0-9][0-9][0-9][0-9][0-9].txt}"
        num="${file##*-}"
        num="${num%.txt}"
        prefix_numbers["$prefix"]+="$num "
    done
done

for prefix in $(printf '%s\n' "${!prefix_numbers[@]}" | sort); do
    mapfile -t numbers < <(printf '%s\n' ${prefix_numbers[$prefix]} | sort -n)
    [[ ${#numbers[@]} -eq 0 ]] && continue

    start=${numbers[0]}
    prev=${numbers[0]}

    for ((i=1; i<${#numbers[@]}; i++)); do
        curr=${numbers[i]}
        if [[ $((10#$curr)) -ne $((10#$prev + 1)) ]]; then
            if [[ $start == $prev ]]; then
                printf "%s/%06d\n" "$prefix" $((10#$start))
            else
                printf "%s/%06d to %06d\n" "$prefix" $((10#$start)) $((10#$prev))
            fi
            start=$curr
        fi
        prev=$curr
    done

    if [[ $start == $prev ]]; then
        printf "%s/%06d\n" "$prefix" $((10#$start))
    else
        printf "%s/%06d to %06d\n" "$prefix" $((10#$start)) $((10#$prev))
    fi
done