"""Extract archival DTIUIE evidence; never contact the cloud or load credentials.

Run with the existing dtiuie Python environment.  Outputs are deterministic CSV/
JSON.  TensorBoard keeps full precision; terminal progress is only a rounded
cross-check.  No sampling gaps are interpolated and power is never fabricated.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from datetime import datetime, timezone, timedelta
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

CST = timezone(timedelta(hours=8))


def stamp(t):
    return datetime.fromtimestamp(t, timezone.utc).isoformat()


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def load_json(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))


def dump_json(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(p, rows, fields=None):
    p.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    with p.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def digest(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def cosine_lr(epoch, base, eta):
    # Code-derived post-scheduler-step values, NOT a measured TB tag.
    restart, period = (0, 10) if epoch <= 10 else (10, 90)
    return eta + (base - eta) * (1 + math.cos(math.pi * (epoch - restart) / period)) / 2


def stats(values):
    values = [float(v) for v in values if v is not None]
    return {"n": len(values), "min": min(values), "max": max(values),
            "mean_unweighted_samples": statistics.mean(values), "median": statistics.median(values),
            "quartiles_exclusive": statistics.quantiles(values, n=4) if len(values) > 1 else values} if values else {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path(r"C:\Users\Lenovo\Desktop\DTIUIE"))
    parser.add_argument("--monitor", type=Path, default=Path(r"C:\Users\Lenovo\autodl-ssh\monitor"))
    parser.add_argument("--portable", action="store_true", help="Rebuild from this report's included hash-verified evidence")
    args = parser.parse_args()
    report = Path(__file__).resolve().parents[1] if args.portable else args.project / "reproduction_report"
    out = report / "data"
    archive = report / "evidence" if args.portable else args.project / "cloud_training_results/results-20261003-ebcddb62d624"
    run = archive / "run" if args.portable else archive / "training_runs/from_scratch_20261003"
    if args.portable:
        args.monitor = archive / "monitor"
    sources = [run / x for x in ("run_metadata.json", "completed.json", "training_result_summary.json", "supervisor_state.json", "heartbeat.json", "first_checkpoint_verified.json")]
    log_parent = run if args.portable else archive / "training_runs"
    sources += [log_parent / "from_scratch_20261003.train.log", log_parent / "from_scratch_20261003.supervisor.log"]
    metadata, completed, original_summary = [load_json(run / n) for n in ("run_metadata.json", "completed.json", "training_result_summary.json")]
    start = parse_time(metadata["created_at_utc"])
    end = parse_time(completed["completed_at_utc"])
    event_path = next(archive.rglob("events.out.tfevents*"))
    sources.append(event_path)
    event = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
    event.Reload()
    scalars = {tag: sorted(event.Scalars(tag), key=lambda x: (x.step, x.wall_time)) for tag in event.Tags()["scalars"]}
    tags_schema = []
    all_events = []
    invalid_tags = {"training/epoch_loss": "AverageMeter.update is commented out; placeholder 0, not total objective", "validation/epoch_uciqe": "Evaluator returns 0 without calculating UCIQE", "validation/epoch_uiqm": "Evaluator returns 0 without calculating UIQM", "validation/epoch_acc": "Original pixel accuracy applies softmax(...).long() before argmax: probabilities are truncated to zero, invalidating predicted labels; do not report as valid pixAcc"}
    for tag, values in scalars.items():
        assert len({x.step for x in values}) == len(values), f"Duplicate TB steps: {tag}"
        for x in values:
            assert math.isfinite(x.value), f"Nonfinite TB value: {tag}, {x.step}"
            all_events.append({"tag": tag, "step": x.step, "value": x.value, "wall_time_unix": x.wall_time, "timestamp_utc": stamp(x.wall_time), "elapsed_hours": (x.wall_time - start) / 3600, "valid_for_report": tag not in invalid_tags})
        tags_schema.append({"tag": tag, "count": len(values), "first_step": values[0].step, "last_step": values[-1].step,
                            "first_timestamp_utc": stamp(values[0].wall_time), "last_timestamp_utc": stamp(values[-1].wall_time),
                            "valid_for_report": tag not in invalid_tags, "invalid_reason": invalid_tags.get(tag)})
    write_csv(out / "tensorboard_scalars.csv", all_events)
    n_batch = 381
    full_tags = {t: {x.step: x for x in vals} for t, vals in scalars.items() if t.startswith("training/iteration")}
    assert len(full_tags["training/iteration_loss_E"]) == 38100
    assert set(full_tags["training/iteration_loss_E"]) == set(range(38100))
    assert set(full_tags["training/iteration_loss_enc_feature"]) == set(range(3810, 38100))
    terminal = sources[6].read_text(encoding="utf-8")
    pbar = re.compile(r"Epoch (\d+)/100:.*?\|\s*(\d+)/381 \[([^<\]]+)[^\]]*?loss_E=([\d.eE+\-]+), loss_S=([\d.eE+\-]+), lr=([\d.eE+\-]+)")
    progress = {}
    match_count = 0
    for line in terminal.splitlines():
        m = pbar.search(line)
        if m:
            epoch, batch = int(m[1]), int(m[2])
            if batch:
                match_count += 1
                # tqdm first emits progress using stale postfix then emits updated
                # postfix; last occurrence per epoch/batch is the valid cross-check.
                progress[(epoch, batch)] = {"epoch": epoch, "batch": batch, "elapsed_in_epoch_tqdm": m[3], "loss_E_rounded": float(m[4]), "loss_S_rounded": float(m[5]), "lr_E_used_rounded": float(m[6])}
    assert len(progress) == 38100, len(progress)
    write_csv(out / "terminal_iterations_deduplicated.csv", list(progress.values()))
    iterations = []
    for step in range(38100):
        epoch, batch = step // n_batch + 1, step % n_batch + 1
        ts = full_tags["training/iteration_loss_E"][step].wall_time
        row = {"global_step": step, "global_step_one_based": step + 1, "epoch": epoch, "batch": batch,
               "wall_time_unix": ts, "timestamp_utc": stamp(ts), "elapsed_hours": (ts - start) / 3600}
        for tag, values in full_tags.items():
            row[tag.removeprefix("training/iteration_")] = values[step].value if step in values else None
        row["lr_E_used_rounded"] = progress[(epoch, batch)]["lr_E_used_rounded"]
        iterations.append(row)
    write_csv(out / "iterations.csv", iterations)
    e_identity_error = max(abs(r["loss_E"] - r["loss_enc_pixel"] - (r["loss_enc_feature"] or 0)) for r in iterations)
    s_identity_error = max(abs(r["loss_S"] - r["loss_seg_enc"] - r["loss_seg_raw"] - r["loss_seg_mix"]) for r in iterations)
    assert e_identity_error < 1e-7
    assert s_identity_error < 1e-5
    epoch_events = {tag: {x.step: x for x in vals} for tag, vals in scalars.items() if "/epoch" in tag or tag == "training/learning_rate"}
    epochs = []
    previous_end = start
    for epoch in range(1, 101):
        samples = iterations[(epoch - 1) * n_batch:epoch * n_batch]
        val_events = [epoch_events[t][epoch] for t in scalars if t.startswith("validation/epoch")]
        val_end = max(x.wall_time for x in val_events)
        train_end = epoch_events["training/epoch_loss_E"][epoch].wall_time
        val_duration = val_end - train_end
        row = {"epoch": epoch, "loss_E": epoch_events["training/epoch_loss_E"][epoch].value,
               "loss_S": epoch_events["training/epoch_loss_S"][epoch].value,
               "loss_SA_mean": statistics.mean(x["loss_seg_sa"] for x in samples),
               "loss_enc_pixel_mean": statistics.mean(x["loss_enc_pixel"] for x in samples),
               "loss_enc_feature_mean": statistics.mean(x["loss_enc_feature"] for x in samples) if epoch > 10 else None,
               "psnr": epoch_events["validation/epoch_psnr"][epoch].value,
               "ssim": epoch_events["validation/epoch_ssim"][epoch].value,
               "miou": epoch_events["validation/epoch_miou"][epoch].value,
               "dice": epoch_events["validation/epoch_dice"][epoch].value,
               "lr_E": epoch_events["training/learning_rate"][epoch].value,
               "lr_E_used": metadata["optimizer_configuration"]["optimizer_E"][0]["lr"] if epoch == 1 else epochs[-1]["lr_E"],
               "lr_S_derived": cosine_lr(epoch, 0.02, 0.0001), "lr_SA_code": 0.02,
               "epoch_end_utc": stamp(val_end), "elapsed_hours": (val_end - start) / 3600,
               "epoch_duration_seconds": val_end - previous_end, "train_duration_seconds": train_end - previous_end,
               "val_duration_seconds": val_duration, "iteration_samples": len(samples), "validation_guidance": "model_SA",
               "training_total_loss_placeholder": epoch_events["training/epoch_loss"][epoch].value,
               "pixel_accuracy_invalid": epoch_events["validation/epoch_acc"][epoch].value,
               "uciqe_placeholder": epoch_events["validation/epoch_uciqe"][epoch].value,
               "uiqm_placeholder": epoch_events["validation/epoch_uiqm"][epoch].value}
        assert abs(row["loss_E"] - statistics.mean(x["loss_E"] for x in samples)) < 1e-7
        assert abs(row["loss_S"] - statistics.mean(x["loss_S"] for x in samples)) < 1e-6
        assert abs(row["lr_E"] - cosine_lr(epoch, 1e-4, 1e-6)) < 1e-10
        epochs.append(row)
        previous_end = val_end
    write_csv(out / "epochs.csv", epochs)
    history_paths = sorted(args.monitor.glob("history-*.json"))
    history = [(p, load_json(p)) for p in history_paths]
    # Poll filename is its completion time; telemetry occurs at checked_at_utc.
    history.sort(key=lambda pair: pair[1]["checked_at_utc"])
    sources.extend(history_paths)
    sources.extend([args.monitor / "persistent_connection.json", args.monitor / "backup_state.json"])
    checks, gpu_samples, checkpoints = [], [], []
    previous_checked = None
    gaps = []
    for source, j in history:
        h = j.get("training", {}).get("heartbeat") or {}
        ts = parse_time(j["checked_at_utc"])
        delta = ts - previous_checked if previous_checked is not None else None
        if delta is not None and delta > 420:
            gaps.append({"start_utc": stamp(previous_checked), "end_utc": stamp(ts), "gap_seconds": delta,
                         "nominal_interval_seconds": 300, "interpretation": "No instantaneous GPU measurements inside this gap; long checkpoint downloads also block sequential polling"})
        previous_checked = ts
        base = {"timestamp_utc": j["checked_at_utc"], "timestamp_cst": datetime.fromtimestamp(ts, CST).isoformat(), "elapsed_hours": (ts - start) / 3600,
                "epoch": h.get("epoch", j.get("epoch")), "batch": h.get("batch"), "total_batches": h.get("total_batches"),
                "global_step": h.get("global_step"), "training_status": j.get("training_status"), "connection_ok": j.get("connection_ok", False)}
        gpu = j.get("gpu") or {}
        gpu_values = [x.strip() for x in gpu.get("stdout", "").split(",")]
        row = {**base, "gpu_name": None, "gpu_util_pct": None, "vram_used_mib": None, "vram_total_mib": None, "temp_c": None, "power_w": None,
               "sample_interval_seconds": delta, "source_file": source.name}
        if gpu.get("exit_code") == 0 and len(gpu_values) == 6:
            row.update(gpu_name=gpu_values[1], vram_used_mib=int(gpu_values[2]), vram_total_mib=int(gpu_values[3]), gpu_util_pct=int(gpu_values[4]), temp_c=int(gpu_values[5]))
        gpu_samples.append(row)
        check = {**base, "connection_attempts": j.get("connection_attempts"), "process_alive": (j.get("process") or {}).get("alive"),
                 "heartbeat_age_seconds": j.get("heartbeat_age_seconds"), "last_saved_epoch": j.get("last_saved_epoch"),
                 "warning_count": len(j.get("warnings") or []), "warnings": json.dumps(j.get("warnings") or [], ensure_ascii=False),
                 "exit_code": j.get("exit_code"), "finished_at_utc": j.get("finished_at_utc"), "source_file": source.name}
        checks.append(check)
        backup = j.get("checkpoint_backup")
        if backup:
            backup_start = parse_time(j["checked_at_utc"])
            backup_end = parse_time(j["finished_at_utc"])
            checkpoints.append({"event_type": "local_checkpoint_backup", "epoch": backup["observed_last_saved_epoch"], "timestamp_utc": j["finished_at_utc"],
                                "start_utc": j["checked_at_utc"], "end_utc": j["finished_at_utc"], "elapsed_start_hours": (backup_start - start) / 3600, "elapsed_end_hours": (backup_end - start) / 3600,
                                "duration_seconds": backup_end - backup_start, "duration_basis": "whole poll with checkpoint download; exact transfer-only start timestamp unavailable",
                                "timestamp_basis": "local poll completion; backup epoch is an observation, not guaranteed saved checkpoint epoch", "path": backup["path"], "sha256": backup["sha256"],
                                "size_bytes": backup["size_bytes"], "source_file": source.name})
    write_csv(out / "gpu_samples.csv", gpu_samples)
    write_csv(out / "monitor_checks.csv", checks)
    for match in re.finditer(r"Save current best model @(ssim|miou).*?@epoch: (\d+)", terminal):
        kind, epoch = match[1], int(match[2])
        checkpoints.append({"event_type": "best_" + kind + "_updated", "epoch": epoch, "timestamp_utc": epochs[epoch - 1]["epoch_end_utc"],
                            "timestamp_basis": "TensorBoard validation completion proxy; exact checkpoint write timestamp absent", "path": None, "sha256": None, "size_bytes": None,
                            "source_file": sources[6].name})
    for kind, details in original_summary["exports"].items():
        checkpoints.append({"event_type": "export_" + kind, "epoch": details["epoch"], "timestamp_utc": original_summary["completed_at_utc"],
                            "timestamp_basis": "export summary creation", "path": details["checkpoint"], "sha256": details["sha256"], "size_bytes": None,
                            "source_file": "training_result_summary.json"})
    checkpoints.sort(key=lambda x: x["timestamp_utc"])
    write_csv(out / "checkpoints/chronology.csv", checkpoints)
    final_s = {}
    m = re.search(r"\*Final\* Validation PSNR: ([\d.]+), SSIM: ([\d.]+)", terminal)
    final_s.update(psnr_rounded=float(m[1]), ssim_rounded=float(m[2]))
    m = re.search(r"\*Final\* Validation pix_acc: ([\d.]+), dice_score: ([\d.]+), miou: ([\d.]+)", terminal)
    final_s.update(pixel_accuracy_invalid=float(m[1]), dice=float(m[2]), miou=float(m[3]), guidance="model_S", epoch=100)
    active = [r for r in gpu_samples if r["training_status"] == "running" and r["gpu_util_pct"] is not None]
    unique_times = [parse_time(r["timestamp_utc"]) for r in gpu_samples if r["gpu_util_pct"] is not None]
    run_hashes = [{"path": str(p), "size_bytes": p.stat().st_size, "sha256": digest(p)} for p in sources if p.exists()]
    summary = {
        "run_id": metadata["run_id"], "source_archive": str(archive), "started_at_utc": stamp(start), "completed_at_utc": stamp(end),
        "started_at_cst": datetime.fromtimestamp(start, CST).isoformat(), "completed_at_cst": datetime.fromtimestamp(end, CST).isoformat(),
        "elapsed_seconds_completed_record": completed["elapsed_seconds"], "wall_clock_seconds_start_to_complete": end - start,
        "final_evaluation_and_completion_after_epoch_100_seconds": end - parse_time(epochs[-1]["epoch_end_utc"]),
        "epochs": 100, "batches_per_epoch": 381, "batch_size": 4, "global_steps_completed_one_based": 38100, "tensorboard_global_steps_zero_based": [0, 38099],
        "iteration_feature_loss_first_epoch": 11, "iteration_feature_loss_count": 34290, "tensorboard_scalar_count": len(all_events),
        "terminal_progress_postfix_rows": match_count, "terminal_distinct_epoch_batch_rows": len(progress), "terminal_redundant_rows_removed": match_count - len(progress),
        "best_internal_validation": original_summary["best_internal_validation"], "epoch_100_model_SA": {k: epochs[-1][k] for k in ("psnr", "ssim", "miou", "dice")},
        "final_log_model_S_different_protocol": final_s, "training_epoch_1": epochs[0], "training_epoch_100": epochs[-1],
        "gpu": {"name": metadata["gpu"]["name"], "history_files": len(history_paths), "valid_samples": len(unique_times), "running_samples": len(active),
                "first_sample_utc": stamp(min(unique_times)), "last_sample_utc": stamp(max(unique_times)),
                "sample_span_hours": (max(unique_times) - min(unique_times)) / 3600,
                "running_util_pct": stats(r["gpu_util_pct"] for r in active), "running_vram_used_mib": stats(r["vram_used_mib"] for r in active),
                "running_temp_c": stats(r["temp_c"] for r in active), "gap_threshold_seconds": 420, "gaps": gaps,
                "gap_count_above_7_minutes": len(gaps), "longest_gap_seconds": max((g["gap_seconds"] for g in gaps), default=0),
                "sum_gap_seconds_above_7_minutes": sum(g["gap_seconds"] for g in gaps),
                "sum_extra_seconds_beyond_nominal_300_within_long_gaps": sum(g["gap_seconds"] - 300 for g in gaps),
                "power_recorded": False, "energy_kwh_available": False,
                "sampling_interpretation": "Discrete instantaneous telemetry, mostly five-minute checks; arithmetic mean is unweighted samples, not continuous/time-weighted GPU utilization", 
                "training_coverage_span_hours": (max(parse_time(r["timestamp_utc"]) for r in active) - min(parse_time(r["timestamp_utc"]) for r in active)) / 3600},
        "monitor": {"connection_success": sum(r["connection_ok"] for r in checks), "connection_failed": sum(not r["connection_ok"] for r in checks),
                    "warning_events": sum(r["warning_count"] for r in checks), "restarts": max((j.get("training", {}).get("supervisor_state") or {}).get("restarts", 0) for _, j in history),
                    "local_checkpoint_backups": len([x for x in checkpoints if x["event_type"] == "local_checkpoint_backup"]), "backup_observed_epochs": [x["epoch"] for x in checkpoints if x["event_type"] == "local_checkpoint_backup"]},
        "epoch_duration_statistics_seconds": stats(r["epoch_duration_seconds"] for r in epochs),
        "warmup_epochs_1_to_10_mean_seconds": statistics.mean(r["epoch_duration_seconds"] for r in epochs[:10]),
        "feature_phase_epochs_11_to_100_mean_seconds": statistics.mean(r["epoch_duration_seconds"] for r in epochs[10:]),
        "invalid_metrics": {**invalid_tags, "terminal/Train PSNR": "train_average_psnr_meter is never updated; logged 0 is a placeholder"},
        "lr_measurement": {"lr_E": "TensorBoard measured post-scheduler-step value", "lr_E_used": "initial optimizer lr for epoch 1; previous epoch measured lr thereafter",
                           "lr_S_derived": "code-derived cosine periods=[10,90], base=0.02, eta_min=0.0001; not a logged epoch measurement",
                           "lr_SA_code": "constant 0.02, code scheduler_sa targets optimizer_seg and is not stepped; supported by sampled heartbeat optimizer values"},
        "epoch_time_measurement": "Epoch completion from max wall_time of TensorBoard validation scalars. Difference to previous validation completion (first uses run start); includes boundary checkpoint/save overhead. train_duration_seconds includes preceding boundary overhead. val_duration is validation-end minus epoch training scalar timestamp. Exact per-kernel timings unavailable.",
        "evidence_limit": "Internal model_SA validation and the model_S final evaluation use different guidance/segmentation networks. These internal evaluations are not equivalent to paper Tables II/III/IV independent downstream retraining protocols. Paper Table V evaluates image quality on the separate SUIM-E/UIEB datasets with a separate metric protocol. No training PSNR, valid pixAcc, UIQM/UCIQE, GPU power or energy were measured.",
        "source_code_sha256": {k: metadata[k] for k in ("original_source_sha256", "monitored_source_sha256", "observer_source_sha256")},
        "packages": metadata["packages"], "torch": metadata["torch"], "cuda_runtime": metadata["cuda_runtime"], "seed_behavior": metadata["seed_behavior"], "args": metadata["args"],
    }
    dump_json(out / "summary.json", summary)
    dump_json(report / "facts.json", summary)
    dump_json(out / "reconstruction_checks.json", {
        "passed": True, "epochs_complete": len(epochs) == 100, "iteration_steps_contiguous": True,
        "tensorboard_duplicate_steps_per_tag": 0, "nonfinite_scalar_values": 0,
        "feature_start_step": 3810, "feature_only_epoch_11_plus": True,
        "E_equals_pixel_plus_feature_max_float32_error": e_identity_error,
        "S_equals_three_CE_losses_sum_max_float32_error": s_identity_error,
        "epoch_averages_checked_against_complete_iteration_values": True,
        "logged_E_cosine_schedule_checked_against_code": True,
        "gpu_timestamp_source": "checked_at_utc; history filename denotes completion time and can be much later during backup",
        "missing_power_left_empty": True, "invalid_total_loss_zero_excluded_from_valid_results": True,
        "terminal_postfix_duplicate_rule": "Last occurrence per (epoch,batch) wins; initial tqdm updates carry stale previous batch losses",
    })
    schema = {
        "schema_version": 1, "generated_by": str(Path(__file__)), "read_only_source_sha256": run_hashes,
        "tensorboard_tags": tags_schema, "timezone": "UTC; *_cst columns include UTC+08:00",
        "files": {p.relative_to(out).as_posix(): {"columns": next(csv.reader(p.open(encoding="utf-8-sig"))), "sha256": digest(p), "size_bytes": p.stat().st_size} for p in out.rglob("*.csv")},
        "source_schema": {"gpu_stdout": ["index", "gpu_name", "memory.used MiB", "memory.total MiB", "utilization.gpu percent", "temperature.gpu Celsius"],
                          "monitor_sample_timestamp": "checked_at_utc, not filename/finished_at_utc", "latest_json": "excluded duplicate of last history JSON",
                          "tensorboard_iteration_steps": "zero-based; heartbeat global_step is one-based count of completed steps"},
        "prohibitions": "No credentials loaded, no remote connections, no source log modified, no missing metrics synthesized",
    }
    dump_json(out / "provenance.json", schema)
    print(json.dumps({"output": str(out), "epochs": len(epochs), "iterations": len(iterations), "gpu_samples": len(gpu_samples), "summary": str(out / "summary.json")}, indent=2))


if __name__ == "__main__":
    main()
