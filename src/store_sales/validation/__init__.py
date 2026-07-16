"""Validation utilities (walk-forward fold builders, re-exports)."""

from store_sales.data.split import build_expanding_folds, save_fold_manifests

__all__ = ["build_expanding_folds", "save_fold_manifests"]
