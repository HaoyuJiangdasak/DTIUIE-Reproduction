#!/usr/bin/env python3
"""Run the supplied checkpoint with the original test preprocessing, without training."""

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time
from datetime import datetime, timezone


PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".ppm", ".bmp"}
OFFICIAL_CHECKPOINT_SHA256 = "e9449842c0f81211f72bceacba63a7d1d2c085c920e00901e26e2e40dea0c7b6"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "DTIUIE checkpoint inference at 256x256. Explicitly choose model_S "
            "(official test) or model_SA (training prior); these weights differ."
        )
    )
    parser.add_argument("--guidance", required=True, choices=("model_S", "model_SA"))
    parser.add_argument("--trusted-checkpoint-sha256", required=True, help="Explicit SHA256 of this run own checkpoint")
    parser.add_argument(
        "--checkpoint", type=Path,
        default=PROJECT_ROOT / "checkpoints" / "dtiuie_ckpt.pth",
    )
    parser.add_argument(
        "--input-dir", type=Path,
        default=PROJECT_ROOT / "dataset" / "test" / "images",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None, help="Process the first N sorted images.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--no-png", action="store_true",
        help="Skip the additional lossless PNG copies (JPEGs are always written).",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be a positive integer")
    return args


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output_names(path):
    # Preserve the official JPG/JPEG basename; other supported inputs get JPEG copies.
    jpeg_name = path.name if path.suffix.lower() in {".jpg", ".jpeg"} else path.stem + ".jpg"
    return jpeg_name, path.stem + ".png"


def validate_paths(args):
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.input_dir = args.input_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.checkpoint.is_file():
        raise ValueError("Checkpoint does not exist: {}".format(args.checkpoint))
    if not args.input_dir.is_dir():
        raise ValueError("Input directory does not exist: {}".format(args.input_dir))
    # Avoid incorporating this run's own output files into a later input traversal.
    if args.output_dir == args.input_dir or args.input_dir in args.output_dir.parents:
        raise ValueError("Output directory must be outside the input directory")
    if args.output_dir.exists():
        if not args.output_dir.is_dir() or any(args.output_dir.iterdir()):
            raise ValueError("Refusing to overwrite a non-empty output directory: {}".format(args.output_dir))
    paths = sorted(
        p for p in args.input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise ValueError("No supported images were found in {}".format(args.input_dir))
    if args.limit is not None:
        paths = paths[:args.limit]
    for name_index, label in ((0, "JPEG"), (1, "PNG")):
        if label == "PNG" and args.no_png:
            continue
        names = [output_names(p)[name_index].casefold() for p in paths]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate {} output basenames; input filenames must be unique".format(label))
    return paths


def write_manifest(path, manifest):
    # Keep a completed-image record even if a later image fails.
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def main():
    args = parse_args()
    paths = validate_paths(args)

    # Import after argument parsing so --help needs only Python's standard library.
    sys.path.insert(0, str(PROJECT_ROOT))
    import numpy as np
    from PIL import Image
    import torch
    import torch.nn.functional as F
    import torchvision
    from torchvision.transforms.functional import to_tensor
    from models.enhancement.model_dtiuie import DTIUIE
    from models.segmentation.vgg_unet import VGG16Unet

    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Only CPU and CUDA devices are supported")
    gpu = None
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but torch.cuda.is_available() is False")
        index = device.index if device.index is not None else 0
        if index >= torch.cuda.device_count():
            raise ValueError("Requested CUDA device does not exist")
        device = torch.device("cuda", index)
        torch.cuda.set_device(device)
        properties = torch.cuda.get_device_properties(device)
        gpu = {
            "name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "total_memory_bytes": properties.total_memory,
        }

    start = time.perf_counter()
    checkpoint_hash = sha256_file(args.checkpoint)
    # Torch 1.13's restricted reader cannot parse the floating scheduler fields in
    # this full checkpoint (Unsupported operand 71). Only permit its legacy reader
    # for the exact reviewed official file; modern Torch uses the restricted reader.
    restricted_reader = not torch.__version__.split("+")[0].startswith("1.13.")
    if checkpoint_hash != args.trusted_checkpoint_sha256:
        raise ValueError("Trained checkpoint does not match its declared SHA256")
    checkpoint = torch.load(
        str(args.checkpoint), map_location="cpu", weights_only=restricted_reader
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected the official checkpoint dictionary")
    for key in ("model_E", args.guidance):
        if key not in checkpoint:
            raise ValueError("Checkpoint is missing required key: {}".format(key))
    enhancer = DTIUIE()
    # All backbone weights are supplied by the checkpoint: never download ImageNet weights.
    guidance = VGG16Unet(n_channels=3, n_classes=8, pretrained=False)
    enhancer.load_state_dict(checkpoint["model_E"], strict=True)
    guidance.load_state_dict(checkpoint[args.guidance], strict=True)
    epoch = checkpoint.get("epoch")
    if torch.is_tensor(epoch):
        epoch = epoch.item()
    if epoch is not None and not isinstance(epoch, (int, float, str, bool)):
        epoch = str(epoch)
    del checkpoint
    enhancer = enhancer.to(device).eval()
    guidance = guidance.to(device).eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jpeg_dir = args.output_dir / "jpeg"
    jpeg_dir.mkdir(exist_ok=False)
    png_dir = args.output_dir / "png"
    if not args.no_png:
        png_dir.mkdir(exist_ok=False)
    manifest_path = args.output_dir / "run_manifest.json"
    manifest = {
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "packages": {name: importlib.metadata.version(name) for name in (
            "numpy", "Pillow", "timm", "einops"
        )},
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device": str(device),
        "gpu": gpu,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_load_weights_only": restricted_reader,
        "checkpoint_epoch": epoch,
        "guidance": args.guidance,
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "limit": args.limit,
        "expected_count": len(paths),
        "completed_count": 0,
        "input_color": "RGB",
        "input_range": [0.0, 1.0],
        "input_normalization": "PIL RGB to torchvision to_tensor; no mean/std",
        "preprocessing": "bicubic align to multiples of 32 (area cap 1088x1920), then bicubic to 256x256; align_corners=False",
        "output_size_hw": [256, 256],
        "postprocessing": "original model normalize_img; original bicubic 256x256; multiply by 255 then uint8 truncation",
        "jpeg_encoding": "PIL JPEG defaults; original JPG/JPEG basename preserved",
        "additional_png": not args.no_png,
        "model_E_parameters": sum(p.numel() for p in enhancer.parameters()),
        "guidance_parameters": sum(p.numel() for p in guidance.parameters()),
        "precision": {
            "model_dtype": str(next(enhancer.parameters()).dtype),
            "autocast": False,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        },
        "images": [],
    }
    write_manifest(manifest_path, manifest)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    inference_start = time.perf_counter()

    try:
        with torch.no_grad():
            for count, path in enumerate(paths, 1):
                image_start = time.perf_counter()
                with Image.open(path) as pil_image:
                    rgb = pil_image.convert("RGB")
                    original_w, original_h = rgb.size
                    image = to_tensor(rgb).unsqueeze(0).to(device)
                aligned_h = (original_h + 31) // 32 * 32
                aligned_w = (original_w + 31) // 32 * 32
                if original_h * original_w > 1088 * 1920:
                    aligned_h, aligned_w = 1088, 1920
                image = F.interpolate(image, size=(aligned_h, aligned_w), mode="bicubic", align_corners=False)
                image = F.interpolate(image, size=(256, 256), mode="bicubic", align_corners=False)
                _, features = guidance(image, return_feats=True)
                output = enhancer(image, features)
                # The official script also interpolates its 256-square output to 256-square.
                output = F.interpolate(output, size=(256, 256), mode="bicubic", align_corners=False)
                expected_shape = (1, 3, 256, 256)
                if tuple(output.shape) != expected_shape:
                    raise ValueError("Unexpected output shape for {}: {}".format(path.name, tuple(output.shape)))
                if not torch.isfinite(output).all().item():
                    raise ValueError("Non-finite output values for {}".format(path.name))
                minimum = output.min().item()
                maximum = output.max().item()
                # Never silently clamp or otherwise change model outputs before uint8 conversion.
                if minimum < 0.0 or maximum > 1.0:
                    raise ValueError("Output range would wrap on uint8 conversion for {}: [{}, {}]".format(path.name, minimum, maximum))
                if not math.isfinite(minimum) or not math.isfinite(maximum):
                    raise ValueError("Non-finite output range for {}".format(path.name))
                pixels = (output[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                saved = Image.fromarray(pixels)
                jpeg_name, png_name = output_names(path)
                # Defaults match Image.save(path_to_jpg) in underwater_test.py.
                saved.save(jpeg_dir / jpeg_name, format="JPEG")
                if not args.no_png:
                    saved.save(png_dir / png_name, format="PNG")
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                record = {
                    "input": str(path),
                    "input_size_hw": [original_h, original_w],
                    "aligned_size_hw": [aligned_h, aligned_w],
                    "output_shape_nchw": list(output.shape),
                    "output_min": minimum,
                    "output_max": maximum,
                    "output_finite": True,
                    "output_range_valid": True,
                    "jpeg": "jpeg/" + jpeg_name,
                    "png": None if args.no_png else "png/" + png_name,
                    "elapsed_seconds": time.perf_counter() - image_start,
                }
                manifest["images"].append(record)
                manifest["completed_count"] = count
                write_manifest(manifest_path, manifest)
                print("[{}/{}] {} -> 256x256; range [{:.6f}, {:.6f}]".format(count, len(paths), path.name, minimum, maximum), flush=True)
                del image, features, output
        manifest["status"] = "complete"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = "{}: {}".format(type(error).__name__, error)
        raise
    finally:
        manifest["inference_seconds"] = time.perf_counter() - inference_start
        manifest["total_seconds"] = time.perf_counter() - start
        if device.type == "cuda":
            manifest["gpu"]["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            manifest["gpu"]["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        write_manifest(manifest_path, manifest)
    print("Completed {} images. Manifest: {}".format(len(paths), manifest_path), flush=True)


if __name__ == "__main__":
    main()
