# E0/E2 legacy 7-object evaluation

These launchers reproduce the supplied legacy evaluation settings. They do not
consume the frozen P0 common-episode manifest, so their results are diagnostic
legacy results rather than formal P0 final-test results.

Run both experiments serially on the single RTX 4090:

```bash
cd /home/MetaCalvin
bash eval/launch/run_e0_then_e2_legacy_eval.sh
```

Run only one experiment:

```bash
bash eval/launch/run_e0_legacy_eval.sh
bash eval/launch/run_e2_legacy_eval.sh
```

E0 independently fixes `feedback=train_feedback`; E2 independently fixes
`feedback=alt_feedback`. Both use their own seed-42 best-validation checkpoint
and the requested legacy parameters: LSTM decoder, window size 12, 7 objects,
5 trials, 300 steps per trial, 15 resets, evaluation mode, and evaluation seed
1. A file lock prevents the two GPU evaluations from running concurrently.

The default output root is:

```text
/home/legacy_e0_e2/
```

Every invocation creates a new UTC timestamped directory. It contains:

- `run.log`: complete stdout and stderr.
- `summary.csv` and `summary.json`: aggregate reward, reward-found ratio, and
  episode-length results.
- `trials.jsonl`: one durable record per completed trial, including feedback,
  template identity, reward, completed tasks, failure category, actions, and
  inference latency.
- `episodes.jsonl`: one record per completed meta-episode, including initial
  state, candidate/meta task, first-reward trial, action/latency summaries, and
  video path.
- `videos/*.mp4`: one interaction video per meta-episode.
- `run_manifest.json`: exact arguments, checkpoint SHA256 and metadata, source
  hashes, runtime versions, GPU, and completion status.
- `checkpoint_load_report.json`: missing/unexpected model keys.
- `command.sh`, environment/GPU/Git snapshots, training metadata copies, and a
  snapshot of the evaluator source.
- `launcher_status.json`: final process status, including failures.

Video recording is enabled by default. To disable it for a faster/lower-disk
run, set `ROBO_EVAL_RECORD_VIDEO=0`; doing so changes only recording, not the
evaluation parameters.

## E1, E3, and C1

The three independent launchers are:

```bash
bash eval/launch/run_e1_legacy_eval.sh
bash eval/launch/run_e3_legacy_eval.sh
bash eval/launch/run_c1_legacy_eval.sh
```

They may be started from three terminals. These three launchers explicitly
disable the optional GPU lock, so all three processes start immediately.
Outputs go to:

```text
/home/evaluation_runs/legacy_e1_e3_c1/
```

Run evaluate sequencely:
```bash
bash eval/launch/run_E0-E7_C0-C7_resource_rerun.sh
```

E1 uses the 7-Object Fixed/LSTM/window-12 legacy protocol. E3 uses the
7-Object Fixed/`lstm_seq` effective-window-1 legacy protocol. C1 uses the
repository's existing Combined legacy evaluation parameters: Fixed/LSTM,
4 trials, 500 steps per trial, 25 resets, and the four held-out task pairs.
The Combined structured logs additionally save completed subtask events,
partial/full pair success, and canonical task-pair identity.
