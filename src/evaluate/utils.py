"""
Utility functions for watermark evaluation.

This module provides utility functions for evaluation, including statistics
calculation for watermark evaluation results.
"""

import pandas as pd
import logging
from typing import Dict, Optional


def calculate_statistics(
    df: pd.DataFrame, logger: Optional[logging.Logger] = None
) -> Dict:
    """
    Calculate FNR and FPR statistics from evaluation results.

    Args:
        df: DataFrame with evaluation results. Must contain columns:
            - "attack_type": "removal" or "forgery"
            - "detected": bool
            - "perturbation": str
            - "aggregation": str
        logger: Optional logger

    Returns:
        Dictionary with statistics:
        {
            "overall": {
                "FNR": float,
                "FNR_count": int,
                "removal_total": int,
                "FPR": float,
                "FPR_count": int,
                "forgery_total": int
            },
            "by_perturbation": {
                "perturbation_name": {
                    "FNR": float,
                    "FPR": float,
                    ...
                }
            },
            "by_aggregation": {
                "aggregation_name": {
                    "FNR": float,
                    "FPR": float,
                    ...
                }
            }
        }
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    stats = {
        "overall": {},
        "by_perturbation": {},
        "by_aggregation": {},
    }

    # Overall statistics
    removal_df = df[df["attack_type"] == "removal"]
    forgery_df = df[df["attack_type"] == "forgery"]

    if len(removal_df) > 0:
        # FNR: False Negative Rate (워터마크가 있는데 검출 실패)
        not_detected = ~removal_df["detected"]
        fnr = not_detected.sum() / len(removal_df)
        stats["overall"]["FNR"] = float(fnr)
        stats["overall"]["FNR_count"] = int(not_detected.sum())
        stats["overall"]["removal_total"] = int(len(removal_df))

    if len(forgery_df) > 0:
        # FPR: False Positive Rate (워터마크가 없는데 검출됨)
        detected = forgery_df["detected"]
        fpr = detected.sum() / len(forgery_df)
        stats["overall"]["FPR"] = float(fpr)
        stats["overall"]["FPR_count"] = int(detected.sum())
        stats["overall"]["forgery_total"] = int(len(forgery_df))

    # Statistics by perturbation (if perturbation column exists)
    if "perturbation" in df.columns:
        for pert in df["perturbation"].unique():
            pert_df = df[df["perturbation"] == pert]
            pert_removal = pert_df[pert_df["attack_type"] == "removal"]
            pert_forgery = pert_df[pert_df["attack_type"] == "forgery"]

            pert_stats = {}
            if len(pert_removal) > 0:
                not_detected = ~pert_removal["detected"]
                fnr = not_detected.sum() / len(pert_removal)
                pert_stats["FNR"] = float(fnr)
                pert_stats["FNR_count"] = int(not_detected.sum())
                pert_stats["removal_total"] = int(len(pert_removal))

            if len(pert_forgery) > 0:
                detected = pert_forgery["detected"]
                fpr = detected.sum() / len(pert_forgery)
                pert_stats["FPR"] = float(fpr)
                pert_stats["FPR_count"] = int(detected.sum())
                pert_stats["forgery_total"] = int(len(pert_forgery))

            stats["by_perturbation"][pert] = pert_stats

    # Statistics by aggregation
    for agg in df["aggregation"].unique():
        agg_df = df[df["aggregation"] == agg]
        agg_removal = agg_df[agg_df["attack_type"] == "removal"]
        agg_forgery = agg_df[agg_df["attack_type"] == "forgery"]

        agg_stats = {}
        if len(agg_removal) > 0:
            not_detected = ~agg_removal["detected"]
            fnr = not_detected.sum() / len(agg_removal)
            agg_stats["FNR"] = float(fnr)
            agg_stats["FNR_count"] = int(not_detected.sum())
            agg_stats["removal_total"] = int(len(agg_removal))

        if len(agg_forgery) > 0:
            detected = agg_forgery["detected"]
            fpr = detected.sum() / len(agg_forgery)
            agg_stats["FPR"] = float(fpr)
            agg_stats["FPR_count"] = int(detected.sum())
            agg_stats["forgery_total"] = int(len(agg_forgery))

        stats["by_aggregation"][agg] = agg_stats

    return stats
