"""Check the actual GPU and execute a small CUDA/cuDNN operation."""

import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import torchvision


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    versions = {name: importlib.metadata.version(name) for name in (
        "numpy", "Pillow", "einops", "timm", "tqdm", "huggingface-hub", "safetensors"
    )}
    details = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "disk_free_gib": round(shutil.disk_usage(root).free / 1024 ** 3, 2),
        "packages": versions,
    }
    if shutil.which("nvidia-smi"):
        result = subprocess.run([
            "nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"
        ], capture_output=True, text=True, timeout=15)
        details["nvidia_smi"] = result.stdout.strip() if result.returncode == 0 else result.stderr.strip()
    if not details["cuda_available"]:
        details["success"] = False
        details["error"] = "No usable CUDA device; verify AutoDL GPU mode and installed Torch build"
        print(json.dumps(details, ensure_ascii=False, indent=2))
        return 2
    properties = torch.cuda.get_device_properties(0)
    details.update({
        "gpu": properties.name,
        "vram_gib": round(properties.total_memory / 1024 ** 3, 2),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch_arch_targets": torch.cuda.get_arch_list(),
    })
    try:
        with torch.no_grad():
            image = torch.rand(1, 3, 32, 32, device="cuda")
            result = torch.nn.Conv2d(3, 8, 3).cuda()(image)
            torch.cuda.synchronize()
            assert torch.isfinite(result).all().item()
        details["cuda_convolution_shape"] = list(result.shape)
        details["success"] = True
    except Exception as exc:
        details.update({"success": False, "error": str(exc)})
    print(json.dumps(details, ensure_ascii=False, indent=2))
    return 0 if details["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
