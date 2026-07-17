"""Point-in-time feature builders and group registry."""

from store_sales.features.registry import (
    FEATURE_GROUPS,
    build_feature_matrix,
    list_group_ablation_configs,
)

__all__ = [
    "FEATURE_GROUPS",
    "build_feature_matrix",
    "list_group_ablation_configs",
]
