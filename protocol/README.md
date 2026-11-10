# protocol artifacts

The files in `manifests/` implement the frozen P0 split for the 7-Object
experiments. The split unit is an entire object configuration, so frames and meta-episodes from one configuration can never occur in more than one split.

Protocol `MetaCALVIN-RA-L-P0-7Object` uses:

- 17 object configurations for training;
- 4 object configurations for validation;
- the original 10 `slider + led` held-out configurations for final test only.

The validation configurations are selected only from configuration metadata,
using `random.Random(20260813).sample` over the sorted 21 eligible
configurations. The selected four cover all seven objects. Rewards, model
losses, and test performance are never consulted.

Regenerate and verify the manifests with:

```bash
python protocol/create_7object_manifests.py \
  --dataset-dir /data/training_7_objs_pos_4 \
  --output-dir protocol/manifests
```

The final-test manifest must not be passed to the training program.

## 5-Object T4 controlled comparison

The files `5object_train.json`, `5object_validation.json`, and
`5object_test.json` define protocol `MetaCALVIN-RA-L-P0-5Object-T4-v1`.
The legacy 5-object source contains all ten three-object configurations,
including the three `slider + led` held-out configurations. The manifests
quarantine those three groups as final-test-only.

Of the remaining seven groups, validation uses two complete groups whose union covers all five objects and whose combined episode count is closest to 20% of the eligible episodes. This selection uses configuration membership and source episode counts only, never rewards, losses, or model outputs. The resulting split contains:

- 5 train groups, 422 episodes, and 115,252 window-size-12 samples;
- 2 validation groups, 111 episodes, and 29,753 samples;
- 3 final-test groups, 367 episodes, and 88,747 samples.

Regenerate and verify the manifests with:

```bash
python protocol/create_5object_manifests.py \
  --dataset-dir /data/training_5_objs \
  --output-dir protocol/manifests
```

The paired training launcher is `train/rebuttal/run_t4_5object_seed.sh`.
`non_episodic` launches `T4E4`, the 5-object analogue of E4, and `episodic`
launches `T4E7`, the analogue of E7. Both start from the same CALVIN-pretrained
RoboFlamingo checkpoint, use diverse (`alt_feedback`) feedback, and receive
9,604 optimizer updates per epoch at effective batch size 12. The only paired
factor is Non-Episodic versus Episodic sampling/decoder state handling.

Run the Non-Episodic seed-42 smoke test first, then remove the smoke flag for
the formal three-epoch run:

```bash
ROBO_SMOKE_TEST=1 bash train/launch/run_t4_5object_seed.sh non_episodic 42
bash train/launch/run_t4_5object_seed.sh non_episodic 42
```

The paired Episodic command is:

```bash
bash train/launch/run_t4_5object_seed.sh episodic 42
```

## Combined-Explore correction

Generate the audited Combined split with:

```bash
python protocol/create_combined_manifests.py \
  --dataset-dir /data/combined_explore \
  --output-dir protocol/manifests
```

There are 24 valid unordered task pairs. The frozen
Combined protocol therefore uses 16 train, 4 validation, and 4 final-test task pairs, quarantines all source episodes belonging to a held-out pair, and checks the recovered 285 episode labels against `meta_task_stats.npz`. See `manifests/combined_split_audit.json` for the exact evidence.
