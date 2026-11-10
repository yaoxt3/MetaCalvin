# Training setting
The current launcher implements E0/E1/E2/E3 for 7-Object and C0/C1 for Combined. E3 uses manifest-backed Episodic sampling; C0/C1 use the corrected Combined P0 split. 

Frozen controls for both conditions:

- OpenFlamingo initialization;
- Non-Episodic sampling, window size 12;
- batch size 12 on one GPU, gradient accumulation 1 (effective batch 12);
- BF16 mixed precision, AdamW, three epochs;
- requested learning rate `1e-5` (the inherited trainer applies its documented
  `batch_size / 6` scaling, producing optimizer LR `2e-5`);
- seeds 42, 43, and 44;
- validation at each epoch end;
- periodic checkpoints every 3,000 completed global batches, with a
  denser checkpoint every 1,000 batches from step 9,000 through 15,000;
- checkpoint selection by minimum mean validation behavior-cloning loss.

For window size 12, the frozen manifests contain 163,110 training windows and 47,216 validation windows. With batch size 12, each epoch uses 13,592 complete training batches (163,104 windows; six are dropped consistently), giving 40,776 optimizer updates over three epochs.

Run one seed:

```bash
conda activate meta-calvin-bw
cd /the-path-to-MetaCalvin/
bash train/launch/run_e0_e2_seed.sh E0 42
```

Canonical entry point and smoke-test examples:

```bash
ROBO_SMOKE_TEST=1 bash train/launch/run_f0_seed.sh E3 42
ROBO_SMOKE_TEST=1 bash train/launch/run_f0_seed.sh C0 42
ROBO_SMOKE_TEST=1 bash train/launch/run_f0_seed.sh C1 42
```

After each smoke test exits successfully, remove `ROBO_SMOKE_TEST=1` to start the corresponding formal three-epoch run.  Run E3 and C1 sequentially on the single workstation GPU.

After inspecting the first formal run, execute all six runs sequentially:

```bash
bash train/launch/run_all_e0_e2_seeds.sh
```

Each run directory contains:

- `run_manifest.json`: arguments, code/data/foundation hashes, environment,
  seed, effective batch, exact step budget, and checkpoint-selection rule;
- `train.log`: complete stdout/stderr with ETA;
- `metrics.jsonl`: machine-readable training, validation, and checkpoint events;
- `latest_checkpoint.pth`: epoch-boundary recovery state;
- `epoch_01_checkpoint.pth`, `epoch_02_checkpoint.pth`, and
  `epoch_03_checkpoint.pth`: retained epoch-end states. Each is a hard link to the corresponding version of `latest`, so it does not duplicate storage;
- `step_XXXXXX_checkpoint.pth`: periodic evaluation states (full model,
  optimizer, and scheduler state); filenames use one-indexed completed steps;
- `best_validation_checkpoint.pth` and `best_checkpoint.json`: the only model eligible for final evaluation.
