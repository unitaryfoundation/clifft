#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# Read imports from every proof module, including ones not yet in the root import file.
modules=()
while IFS= read -r module; do
    modules+=("$module")
done < <(grep -rhE '^[[:space:]]*(public[[:space:]]+)?import[[:space:]]+Mathlib' ClifftProofs* |
    sed -E 's/^[[:space:]]*(public[[:space:]]+)?import[[:space:]]+(Mathlib[^[:space:]]*).*/\2/' |
    sort -u)

# An empty argument list would fetch all of Mathlib.
if [ "${#modules[@]}" -eq 0 ]; then
    echo "No Mathlib imports found in the proof sources" >&2
    exit 1
fi
lake exe cache get "${modules[@]}"
