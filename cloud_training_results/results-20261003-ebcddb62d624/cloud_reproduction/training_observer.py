"""Operational monitoring and full epoch-boundary recovery for the original trainer."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import socket
import time
import traceback
import uuid

import numpy as np
import torch


FORMAT = "DTIUIE_MONITORED_FULL"


def now():
    return datetime.now(timezone.utc).isoformat()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_value(value):
    if torch.is_tensor(value):
        value = value.detach().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_json(path, payload):
    temporary = path.with_name(".{}.{}.tmp".format(path.name, os.getpid()))
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(json_value(payload), handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


class TrainingObserver:
    def __init__(self, args, current_time, script_path, source_sha256):
        self.args = args
        self.current_time = current_time
        self.script_path = Path(script_path).resolve()
        self.source_sha256 = source_sha256
        self.script_sha256 = file_hash(self.script_path)
        self.started = time.perf_counter()
        self.pid = os.getpid()
        self.run_id = uuid.uuid4().hex
        self.run_dir = None
        self.resume = None
        self.models = self.optimizers = self.schedulers = self.scalers = {}
        self.device = torch.device("cpu")
        self.latest = {}
        self.metadata = {}

    def prepare(self):
        incoming = self.args
        resume_path = Path(incoming.resume_full).expanduser().resolve() if incoming.resume_full else None
        if resume_path:
            # Explicit trusted operational checkpoint: RNG contains NumPy/Python pickle objects.
            saved = torch.load(str(resume_path), map_location="cpu", weights_only=False)
            if not isinstance(saved, dict) or saved.get("format") != FORMAT or saved.get("schema_version") != 1:
                raise ValueError("--resume_full requires this observer's full checkpoint format")
            if saved.get("original_source_sha256") != self.source_sha256:
                raise ValueError("Original training source hash differs from the recovery checkpoint")
            self.resume = saved
            self.run_id = saved["run_id"]
            self.current_time = saved["current_time"]
            self.args = argparse.Namespace(**saved["args"])
            self.args.resume_full = str(resume_path)
            self.args.cudnn_benchmark = bool(getattr(self.args, "cudnn_benchmark", False) or incoming.cudnn_benchmark)
        selected = incoming.run_dir or (str(resume_path.parent) if resume_path else None)
        self.run_dir = Path(selected).expanduser().resolve() if selected else self.script_path.parent / "training_runs" / self.current_time
        if self.run_dir.exists() and any(self.run_dir.iterdir()):
            metadata_path = self.run_dir / "run_metadata.json"
            if not resume_path or not metadata_path.is_file():
                raise ValueError("Refusing to reuse a non-empty run directory without its full recovery checkpoint")
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
            if existing.get("run_id") != self.run_id:
                raise ValueError("Recovery checkpoint does not own the selected run directory")
            self.metadata = existing
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if resume_path:
            for marker_name in ('failed.json', 'completed.json'):
                marker_path = self.run_dir / marker_name
                if marker_path.is_file():
                    archived_marker = self.run_dir / ('previous_{}_{}.json'.format(
                        marker_path.stem, uuid.uuid4().hex[:12]))
                    os.replace(str(marker_path), str(archived_marker))
        self.args.run_dir = str(self.run_dir)
        if self.args.epochs < 1:
            raise ValueError("Training requires at least one epoch")
        if self.resume and self.resume["epoch"] > self.args.epochs:
            raise ValueError("Checkpoint epoch exceeds saved training length")
        self.metadata.update({
            "format": FORMAT, "schema_version": 1, "run_id": self.run_id,
            "original_source_sha256": self.source_sha256,
            "monitored_source_sha256": self.script_sha256,
            "observer_source_sha256": file_hash(__file__),
            "run_dir": str(self.run_dir), "current_time": self.current_time,
            "status": "initializing", "pid": self.pid, "hostname": socket.gethostname(),
            "updated_at_utc": now(), "args": vars(self.args),
            "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "seed_behavior": "Author seed setting remains commented out",
        })
        self.metadata.setdefault("created_at_utc", now())
        self.metadata.setdefault("attempts", []).append({
            "started_at_utc": now(), "pid": self.pid,
            "resume_full": str(resume_path) if resume_path else None,
        })
        atomic_json(self.run_dir / "run_metadata.json", self.metadata)
        atomic_json(self.run_dir / "heartbeat.json", {
            "run_id": self.run_id, "status": "initializing", "pid": self.pid,
            "updated_at_utc": now(), "elapsed_seconds": time.perf_counter() - self.started,
        })
        return self.args, self.current_time

    def bind(self, args, current_time, device, models, optimizers, schedulers, scalers,
             best_criterions, global_step, total_batches):
        self.args, self.current_time, self.device = args, current_time, device
        self.models, self.optimizers = models, optimizers
        self.schedulers, self.scalers = schedulers, scalers
        start_epoch = 1
        if self.resume:
            saved = self.resume
            for group_name, objects in (("models", models), ("optimizers", optimizers),
                                        ("schedulers", schedulers), ("scalers", scalers)):
                for key, obj in objects.items():
                    if group_name == "models":
                        obj.load_state_dict(saved[group_name][key], strict=True)
                    else:
                        obj.load_state_dict(saved[group_name][key])
            start_epoch = int(saved["epoch"]) + 1
            global_step = int(saved["global_step"])
            best_criterions = saved["best_criterions"]
            rng = saved["rng"]
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            if rng["cuda"]:
                if not torch.cuda.is_available() or len(rng["cuda"]) != torch.cuda.device_count():
                    raise ValueError("CUDA RNG recovery requires the same visible GPU count")
                torch.cuda.set_rng_state_all(rng["cuda"])
            self.resume = None
        self.latest = {"epoch": start_epoch, "batch": 0, "total_batches": total_batches,
                       "global_step": global_step}
        packages = {}
        for name in ("torchvision", "numpy", "Pillow", "timm", "einops", "pandas", "scikit-image", "scipy", "kornia", "tensorboardX"):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        self.metadata.update({"status": "running", "args": vars(args), "packages": packages,
                              "device": str(device), "cudnn_benchmark": torch.backends.cudnn.benchmark,
                              "start_epoch": start_epoch, "updated_at_utc": now(),
                              "optimizer_configuration": self.optimizer_details()})
        if device.type == "cuda":
            prop = torch.cuda.get_device_properties(device)
            self.metadata["gpu"] = {"name": prop.name, "total_memory_bytes": prop.total_memory,
                                    "compute_capability": [prop.major, prop.minor]}
        atomic_json(self.run_dir / "run_metadata.json", self.metadata)
        self.heartbeat("running")
        return start_epoch, global_step, best_criterions

    def optimizer_details(self):
        return {key: [{"lr": group["lr"], "weight_decay": group.get("weight_decay"),
                       "momentum": group.get("momentum"), "betas": group.get("betas")}
                      for group in obj.param_groups] for key, obj in self.optimizers.items()}

    def check_loss(self, loss, phase, epoch, batch, global_step):
        if not torch.is_tensor(loss) or not torch.isfinite(loss).all().item():
            self.latest.update(epoch=epoch, batch=batch, global_step=global_step, failed_phase=phase)
            raise FloatingPointError("Non-finite or missing {} loss at epoch {}, batch {}".format(phase, epoch, batch))

    def iteration(self, epoch, batch, total_batches, global_step, loss_E, loss_S, loss_SA):
        self.latest.update(epoch=epoch, batch=batch, total_batches=total_batches, global_step=global_step)
        if global_step % 10 == 0 or batch == 1 or batch == total_batches:
            self.latest.update(loss_E=loss_E.detach().item(), loss_S=loss_S.detach().item(), loss_SA=loss_SA.detach().item())
            self.heartbeat("running")

    def heartbeat(self, status):
        payload = {**self.latest, "run_id": self.run_id, "status": status, "pid": self.pid,
                   "updated_at_utc": now(), "elapsed_seconds": time.perf_counter() - self.started,
                   "optimizers": self.optimizer_details(),
                   "scalers": {key: obj.get_scale() for key, obj in self.scalers.items()}}
        if self.device.type == "cuda":
            payload["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(self.device)
            payload["gpu_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(self.device)
        atomic_json(self.run_dir / "heartbeat.json", payload)

    def epoch_complete(self, epoch, global_step, best_criterions, args, current_time):
        self.latest.update(epoch=epoch, global_step=global_step)
        self.heartbeat("checkpointing")
        payload = {
            "format": FORMAT, "schema_version": 1, "run_id": self.run_id,
            "original_source_sha256": self.source_sha256, "monitored_source_sha256": self.script_sha256,
            "models": {key: obj.state_dict() for key, obj in self.models.items()},
            "optimizers": {key: obj.state_dict() for key, obj in self.optimizers.items()},
            "schedulers": {key: obj.state_dict() for key, obj in self.schedulers.items()},
            "scalers": {key: obj.state_dict() for key, obj in self.scalers.items()},
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else []},
            "best_criterions": best_criterions, "global_step": global_step, "epoch": epoch,
            "args": vars(args).copy(), "current_time": current_time, "saved_at_utc": now(),
        }
        target = self.run_dir / "latest_full.pth"
        temporary = self.run_dir / ".latest_full.{}.tmp".format(self.pid)
        try:
            with temporary.open("wb") as handle:
                torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(str(temporary), str(target))
        finally:
            if temporary.exists():
                temporary.unlink()
        self.latest.update(last_saved_epoch=epoch, latest_full=str(target))
        self.heartbeat("running")

    def complete(self, epoch, global_step):
        self.latest.update(epoch=epoch, global_step=global_step)
        self.metadata.update(status="complete", completed_at_utc=now(), completed_epoch=epoch,
                             global_step=global_step)
        atomic_json(self.run_dir / "run_metadata.json", self.metadata)
        atomic_json(self.run_dir / "completed.json", {"run_id": self.run_id, "pid": self.pid,
                    "epoch": epoch, "global_step": global_step, "completed_at_utc": now(),
                    "elapsed_seconds": time.perf_counter() - self.started})
        self.heartbeat("complete")

    def fail(self, error):
        if self.run_dir is None or not self.run_dir.is_dir() or not self.metadata:
            return
        details = {"run_id": self.run_id, "pid": self.pid, "failed_at_utc": now(),
                   "error": "{}: {}".format(type(error).__name__, error),
                   "traceback": traceback.format_exc(), **self.latest}
        self.metadata.update(status="failed", updated_at_utc=now(), error=details["error"])
        atomic_json(self.run_dir / "run_metadata.json", self.metadata)
        atomic_json(self.run_dir / "failed.json", details)
        self.heartbeat("failed")
