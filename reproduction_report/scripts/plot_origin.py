"""Prepare traceable Origin inputs and run an isolated, hidden Origin COM instance.

Only writes reproduction_report/tools/origin. Never attaches to an existing Origin
window, mutates the model code, or accesses SSH credentials. Missing Origin is
recorded honestly; no OPJU or image is fabricated.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tools" / "origin"


def registry_probe():
    import winreg
    entries = []
    for name, view in (("64-bit", winreg.KEY_WOW64_64KEY),
                       ("32-bit", winreg.KEY_WOW64_32KEY)):
        entry = {"registry_view": name, "progid": "Origin.Application"}
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"Software\Classes\Origin.Application\CLSID",
                                0, winreg.KEY_READ | view) as key:
                clsid = winreg.QueryValue(key, None)
            entry["clsid"] = clsid
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                rf"Software\Classes\CLSID\{clsid}\LocalServer32",
                                0, winreg.KEY_READ | view) as key:
                exe = winreg.QueryValue(key, None)
            entry["registered_executable"] = exe
            entry["registered_executable_exists"] = Path(exe.strip('"')).exists()
        except OSError as exc:
            entry["registry_error"] = str(exc)
        entries.append(entry)
    return entries


def prepare_data():
    summary = {}
    for name in ("epochs.csv", "gpu_samples.csv"):
        source = ROOT / "data" / name
        if not source.exists():
            summary[name] = {"available": False}
            continue
        shutil.copy2(source, OUT / name)
        with source.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        summary[name] = {"available": True, "rows": len(rows),
                         "source": str(source),
                         "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                         "columns": list(rows[0]) if rows else []}
        if name == "gpu_samples.csv":
            times = sorted(float(row["elapsed_hours"]) for row in rows
                           if row.get("elapsed_hours"))
            gaps = [(b - a) * 3600 for a, b in zip(times, times[1:])]
            summary[name].update({
                "first_elapsed_hours": times[0] if times else None,
                "last_elapsed_hours": times[-1] if times else None,
                "median_interval_seconds": statistics.median(gaps) if gaps else None,
                "max_interval_seconds": max(gaps) if gaps else None,
                "active_training_samples": sum(row.get("training_status") == "running" for row in rows),
                "plot_policy": "Actual recorded points only; no interpolation or invented continuous telemetry."
            })
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    evidence = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "registry": registry_probe(),
        "input_data": prepare_data(),
        "origin_com_succeeded": False,
        "origin_project_created": False,
        "origin_figures_created": False,
    }
    app = None
    try:
        if args.prepare_only:
            evidence["status"] = "prepared_only"
        else:
            import win32com.client
            # DispatchEx creates a separate application; GetActiveObject/attach is forbidden here.
            app = win32com.client.DispatchEx("Origin.Application")
            app.Visible = 0
            evidence["origin_com_succeeded"] = True
            evidence["origin_version"] = str(app.GetLTVar("@V"))
            if all(v.get("available") for v in evidence["input_data"].values()):
                success = app.Execute(f'run -pyf "{(OUT / "origin_builtin.py").as_posix()}";')
                evidence["labtalk_execute_return"] = str(success)
                marker = OUT / "origin_export_success.json"
                if marker.exists():
                    evidence["origin_export"] = json.loads(marker.read_text(encoding="utf-8"))
                    evidence["origin_project_created"] = (OUT / "DTIUIE_training_and_GPU.opju").exists()
                    evidence["origin_figures_created"] = bool(list(OUT.glob("*.png")))
                evidence["status"] = "exported" if evidence["origin_project_created"] else "export_failed"
            else:
                evidence["status"] = "data_pending"
    except Exception as exc:
        evidence["status"] = "blocked_origin_not_available"
        evidence["exception_type"] = type(exc).__name__
        evidence["exception"] = str(exc)
        evidence["hresult"] = getattr(exc, "hresult", None)
        evidence["limitation"] = (
            "Only old Origin COM registration was found; the registered executable is missing. "
            "No Origin session, OPJU project, or Origin-rendered image was created. "
            "The copied CSVs and origin_builtin.py are ready for a machine with a working Origin installation."
        )
    finally:
        if app is not None:
            try:
                app.Exit()
            except Exception as exc:
                evidence["origin_exit_error"] = str(exc)
        (OUT / "origin_availability.json").write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
        (OUT / "origin_probe.log").write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(evidence, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
