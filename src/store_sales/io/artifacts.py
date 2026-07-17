"""Run artifact persistence: config, metrics, environment, metadata."""

from __future__ import annotations

import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml

from store_sales.io.logging import get_logger

logger = get_logger(__name__)

_PACKAGE_PROBE = (
    "pandas",
    "numpy",
    "scikit-learn",
    "lightgbm",
    "catboost",
    "xgboost",
    "optuna",
    "pyarrow",
    "PyYAML",
)


def collect_environment() -> dict[str, Any]:
    """Capture interpreter, package versions, platform, optional CUDA flag."""
    packages: dict[str, str | None] = {}
    for name in _PACKAGE_PROBE:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None

    cuda_available: bool | None = None
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        # torch is optional; leave null when unavailable
        cuda_available = None

    return {
        "python_version": sys.version,
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "cuda_available": cuda_available,
    }


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except Exception:
        return None
    return None


def save_run_dir(
    *,
    outputs_root: Path | str,
    run_id: str,
    config: dict[str, Any],
    metrics: dict[str, Any],
    seed: int,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a reproducible run directory under ``outputs_root / runs / run_id``.

    Always writes: ``config.yaml``, ``metrics.json``, ``environment.json``,
    ``run_metadata.json``. Optional ``extra`` payloads are written as JSON
    files named by key (e.g. ``fold_metrics`` → ``fold_metrics.json``).
    """
    run_dir = Path(outputs_root) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "config.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, default_flow_style=False)

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8"
    )

    env = collect_environment()
    (run_dir / "environment.json").write_text(
        json.dumps(env, indent=2), encoding="utf-8"
    )

    meta = {
        "seed": seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "hostname": socket.gethostname(),
        "run_id": run_id,
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    if extra:
        for key, value in extra.items():
            path = run_dir / f"{key}.json"
            path.write_text(
                json.dumps(value, indent=2, default=str), encoding="utf-8"
            )

    logger.info("Saved run artifacts to %s", run_dir)
    return run_dir
