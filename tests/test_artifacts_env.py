from pathlib import Path

from store_sales.io.artifacts import save_run_dir


def test_save_run_dir_writes_environment_json(tmp_path: Path):
    run_dir = save_run_dir(
        outputs_root=tmp_path,
        run_id="unit_run",
        config={"seed": 42, "model": {"name": "last_value"}},
        metrics={"mean_rmsle": 1.23},
        seed=42,
    )
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "environment.json").exists()
    assert (run_dir / "run_metadata.json").exists()
    assert (run_dir / "config.yaml").exists()
