#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 E1|E2 /absolute/path/to/step_003000_checkpoint.pth" >&2
  exit 2
fi

condition="$1"
resume_checkpoint="$2"
case "$condition" in
  E1|E2) ;;
  *)
    echo "Condition must be E1 or E2: $condition" >&2
    exit 2
    ;;
esac

project=/data/yaoxt/RAL-rebuttal/thesis/RoboTest
run_root=/data/yaoxt/RoboRuns/rebuttal_P0_5object_T4_E1E2_seeds42_43_SSD_resume_20260821
python_bin=/home/yaoxt/.conda/envs/meta-calvin-bw/bin/python

cd "$project"
mkdir -p "$run_root"

echo "$(date) $condition chain waiting for CUDA"
until "$python_bin" -c 'import torch; assert torch.cuda.is_available(); torch.ones(1, device="cuda")' >/dev/null 2>&1; do
  sleep 30
done

echo "$(date) CUDA ready; resume T4${condition} seed42 from completed step 3000"
ROBO_DISABLE_GPU_LOCK=1 \
ROBO_5OBJ_RUN_ROOT="$run_root" \
ROBO_STOP_AFTER_EPOCHS=1 \
ROBO_SAVE_EVERY_ITER=3000 \
ROBO_RESUME_FROM_CHECKPOINT="$resume_checkpoint" \
ROBO_RESUME_BATCH_OFFSET=-1 \
bash train/rebuttal/run_t4_5object_seed.sh "$condition" 42

echo "$(date) T4${condition} seed42 completed one epoch; starting seed43"
ROBO_DISABLE_GPU_LOCK=1 \
ROBO_5OBJ_RUN_ROOT="$run_root" \
ROBO_STOP_AFTER_EPOCHS=1 \
ROBO_SAVE_EVERY_ITER=3000 \
bash train/rebuttal/run_t4_5object_seed.sh "$condition" 43

echo "$(date) $condition seed42 -> seed43 chain complete"
