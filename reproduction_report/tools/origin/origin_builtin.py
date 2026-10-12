"""Run in Origin's built-in Python to create editable worksheets and graphs.

This script is supplied as an import/rebuild fallback. It has NOT been executed
in Origin on the current machine: Origin's registered executable is missing.
Once Origin works, use scripts/plot_origin.py to launch an isolated instance or
run this file in a NEW Origin session. The inputs are actual archived records.
"""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import originpro as op

OUT = Path(__file__).resolve().parent


def load(name):
    with (OUT / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def numeric_sheet(name, rows, specifications):
    sheet = op.new_sheet("w", lname=name)
    sheet.cols = len(specifications)
    for column, (field, long_name, unit) in enumerate(specifications):
        values = [float(row[field]) if row.get(field, "") != "" else float("nan")
                  for row in rows]
        sheet.from_list(column, values, lname=long_name, units=unit,
                        comments="Derived directly from archived DTIUIE logs; missing values retained.")
    sheet.cols_axis("xy", 0, len(specifications) - 1)
    return sheet


def graph(name, worksheet, ycol, xlabel, ylabel, color, scatter=False):
    page = op.new_graph(lname=name, template="origin")
    layer = page[0]
    plot = layer.add_plot(worksheet, ycol, 0, type="s" if scatter else "l")
    plot.color = color
    if scatter:
        plot.symbol_size = 4
    else:
        plot.line_width = 2
    layer.rescale()
    layer.lt_exec(f'label -xb "{xlabel}"; label -yl "{ylabel}";')
    image_path = OUT / f"{name}.png"
    exported = page.save_fig(str(image_path), width=1800)
    if not exported or not image_path.is_file() or image_path.stat().st_size == 0:
        raise RuntimeError(f"Origin failed to export {name}")
    return name


def main():
    epochs = load("epochs.csv")
    gpu = [row for row in load("gpu_samples.csv")
           if row.get("elapsed_hours") and row.get("gpu_util_pct")]
    gpu.sort(key=lambda row: float(row["elapsed_hours"]))
    op.new()
    ew = numeric_sheet("Training_100_epochs", epochs, [
        ("epoch", "Epoch", ""),
        ("ssim", "Validation SSIM", ""),
        ("loss_E", "Mean enhancement loss E", ""),
        ("loss_S", "Mean guidance segmentation loss S", ""),
        ("loss_SA_mean", "Mean guidance autoencoder loss SA", ""),
        ("psnr", "Validation PSNR", "dB"),
        ("miou", "Validation mean IoU", ""),
    ])
    gw = numeric_sheet("GPU_recorded_snapshots", gpu, [
        ("elapsed_hours", "Elapsed training time", "h"),
        ("gpu_util_pct", "GPU utilization", "%"),
        ("vram_used_mib", "VRAM used", "MiB"),
        ("temp_c", "GPU temperature", "deg C"),
    ])
    names = [
        graph("Validation_SSIM", ew, 1, "Epoch", "Validation SSIM", "#176A89"),
        graph("Enhancement_loss_E", ew, 2, "Epoch", "Mean enhancement loss E", "#176A89"),
        graph("Guidance_segmentation_loss_S", ew, 3, "Epoch", "Mean guidance segmentation loss S", "#B36A34"),
        graph("Guidance_autoencoder_loss_SA", ew, 4, "Epoch", "Mean guidance autoencoder loss SA", "#7A5DA8"),
        graph("GPU_utilization_samples", gw, 1, "Elapsed training time (h)", "GPU utilization (%)", "#176A89", True),
        graph("GPU_VRAM_samples", gw, 2, "Elapsed training time (h)", "VRAM used (MiB)", "#B36A34", True),
        graph("GPU_temperature_samples", gw, 3, "Elapsed training time (h)", "GPU temperature (deg C)", "#7A5DA8", True),
    ]
    project = OUT / "DTIUIE_training_and_GPU.opju"
    op.save(str(project))
    if not project.is_file() or project.stat().st_size == 0:
        raise RuntimeError("Origin project save failed")
    (OUT / "origin_export_success.json").write_text(json.dumps({
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": str(project), "graphs": names,
        "epoch_rows": len(epochs), "actual_gpu_samples": len(gpu),
        "gpu_interpolation": False,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
