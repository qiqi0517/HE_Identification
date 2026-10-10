#!/usr/bin/env bash
# Run the declared expert comparisons; no LLM calls or test-set evaluation.
set -euo pipefail
cd "$(dirname "$0")/../.."
if [ "$#" -eq 0 ]; then set -- 42; fi
for seed in "$@"; do
  case "$seed" in 42|43|44) ;; *) echo 'Seeds must be 42, 43 or 44' >&2; exit 2 ;; esac
done
run_candidate() {
  local candidate="$1" variant="$2"
  shift 2
  python scripts/experiments/train_typing_expert.py train \
    --variant "$variant" --seed "$seed" \
    --output "runs/expert_typing/adjustments/$candidate/$seed" "$@"
}
for seed in "$@"; do
  run_candidate S1_default S1
  run_candidate S1_standard S1 --feature-scaling standard
  run_candidate S1_long S1 --epochs 200 --early-stop-metric loss --patience 20
  run_candidate S2_default S2
  run_candidate S2_wd1e3 S2 --weight-decay 0.001
  run_candidate S2_wd1e2 S2 --weight-decay 0.01
  run_candidate S2_weighted S2 --class-weight inverse_sqrt
  run_candidate S3_default S3
  run_candidate S3_head1e3 S3 --finetune-head-lr 0.001
done
