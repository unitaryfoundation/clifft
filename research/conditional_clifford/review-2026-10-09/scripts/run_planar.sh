#!/bin/bash
# Run the ordinary and combined (prototype) workers on the planar fold-transversal circuits.
# Requires the branch's .venv and the opt-in build-profile targets (see ../../REPRODUCING.md).
set -u
ROOT=$(cd "$(dirname "$0")/../../../.." && pwd)
PY=$ROOT/.venv/bin/python
SURVEY=$ROOT/tools/profile/survey_conditional_capability.py
WORKER=$ROOT/build-profile/profile_prefix_trace_reuse
EXPORTER=$ROOT/build-profile/export_optimized_prefix
cd "$(dirname "$0")/../circuits"
OUT=../results/planar_results.jsonl
: > "$OUT"
for d in 3 5; do for name in check1 check2 T_check1 T_check2; do for noise in ideal noisy; do
  f=planar_d${d}_${name}_${noise}.stim
  for wk in ordinary combined; do
    line=$(OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 timeout 900 "$PY" "$SURVEY" --native-worker "$WORKER" --exporter "$EXPORTER" --output /dev/null --worker $wk --source "$f" --shots 8 --max-width 16 2>&1 | tail -1)
    if [ "${line:0:1}" = "{" ]; then
      echo "{\"case\": \"$f\", \"worker\": \"$wk\", \"row\": $line}" >> "$OUT"
    else
      printf '%s' "$line" | "$PY" -c "import json,sys; print(json.dumps({'case': '$f', 'worker': '$wk', 'row': {'error': sys.stdin.read()[-700:]}}))" >> "$OUT"
    fi
  done
done; done; done
echo '{"done": true}' >> "$OUT"
