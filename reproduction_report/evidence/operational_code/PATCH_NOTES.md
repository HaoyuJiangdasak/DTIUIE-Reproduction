# Monitored training patch notes

Original: `../underwater_train.py`
Original SHA256: `213bdf088fc32e8294bcb9e7f1c13f5cf1dd5270884926141ccdc2d5405143e0`
Generated copy SHA256: `f52f96b3bfb1369c3eb33bee194aafd616dd02baa8628348fc7101a91a93cef7`

Regenerate with `python cloud_reproduction/build_monitored_train.py` from the project root.
The builder requires each exact anchor to occur once and checks the generated AST.
Original files are never edited. Only `underwater_train_monitored.py` and these notes are generated.

Operational additions:
- `--run_dir`: an empty new directory for `run_metadata.json`, atomic `heartbeat.json`,
  rolling `latest_full.pth`, and a `completed.json` or `failed.json` marker.
- Heartbeat every 10 completed global iterations, additionally the first/last batch of each epoch.
  Reports the three losses, epoch/batch/global step, PID, elapsed time, GPU peak memory,
  actual optimizer learning rates/weight decay, and all three GradScaler scales.
- Fail on non-finite phase loss before backward. No gradient clipping or loss modification.
- After validation and the author's best saves each epoch, atomically replace this run's
  rolling full checkpoint. It contains three models/optimizers/schedulers/scalers,
  Python/NumPy/Torch/CUDA RNG, best criteria, global step, completed epoch, args, and timestamp.
- `--resume_full`: restore this monitored format and start at the next epoch. A checkpoint
  after epoch 100 can finish the original final evaluation/save without another training epoch.
  Restore RNG only after constructor initialization. Keep the saved training configuration
  and timestamp; `--run_dir` can select a new recovery directory. An explicitly supplied
  `--cudnn_benchmark` can enable autotuning on recovery.
- `--cudnn_benchmark`: enable cuDNN kernel autotuning only. AMP is not enabled.
- Finish/error markers are operational status, not new evaluation metrics.

Author behavior intentionally retained: L1/nearest; default batch 4, 100 epochs, warmup 10;
inactive seed CLI; actual AdamW default weight decay 0.01; unweighted three CE losses;
SA scheduler's original binding and absence of stepping; original mode/gradient behavior;
original best/final evaluation network choice and metric code; original best/final filenames.
The ineffective original `--resume`/`--pretrained` options are not used by the observer.
Do not enable the author's broken `--save_iter_checkpoint` branch.

Launch from the project root. Resume only full checkpoints produced by this observer.
Ordinary author checkpoints and the released inference checkpoint are not resumable here.
