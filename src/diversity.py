"""
===============================================================================
diversity.py

Ecological Diversity Analysis Utilities

Provides functions for calculating alpha diversity metrics commonly used in
community ecology.

Implemented metrics
-------------------
- Species Richness
- Shannon Diversity Index
- Simpson Diversity Index (1 - D)
- Pielou's Evenness

Author: Peter Ugege
Project: Omo Forest Ecological Informatics Project
===============================================================================
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

__version__ = "1.0.0"

__all__ = [
    "species_richness",
    "shannon_index",
    "simpson_index",
    "pielou_evenness",
    "calculate_alpha_diversity",
    "summarize_alpha_diversity",
    "export_alpha_diversity",
]

def species_richness(abundances: np.ndarray) -> int:
    """
    Calculate species richness.

    Parameters
    ----------
    abundances : np.ndarray
        Vector of species abundances for a single plot.

    Returns
    -------
    int
        Number of species present.
    """

    abundances = np.asarray(abundances)

    return int(np.sum(abundances > 0))
    
def shannon_index(abundances: np.ndarray) -> float:
    """
    Calculate Shannon diversity index (H').

    Parameters
    ----------
    abundances : np.ndarray
        Species abundances.

    Returns
    -------
    float
        Shannon diversity index.
    """

    abundances = np.asarray(abundances, dtype=float)

    total = abundances.sum()

    if total == 0:
        return np.nan

    proportions = abundances[abundances > 0] / total

    return float(-np.sum(proportions * np.log(proportions)))
    
def simpson_index(abundances: np.ndarray) -> float:
    """
    Calculate Simpson diversity index (1 - D).

    Parameters
    ----------
    abundances : np.ndarray
        Species abundances.

    Returns
    -------
    float
        Simpson diversity index.
    """

    abundances = np.asarray(abundances, dtype=float)

    total = abundances.sum()

    if total == 0:
        return np.nan

    proportions = abundances / total

    return float(1 - np.sum(proportions ** 2))
    
def pielou_evenness(
    richness: int,
    shannon: float
) -> float:
    """
    Calculate Pielou's evenness.

    Parameters
    ----------
    richness : int
        Species richness.

    shannon : float
        Shannon diversity index.

    Returns
    -------
    float
        Pielou's evenness.
    """

    if richness <= 1:
        return np.nan

    return float(shannon / np.log(richness))
    
def calculate_alpha_diversity(
    community_matrix: pd.DataFrame,
    plot_metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Calculate alpha diversity metrics for every plot.

    Parameters
    ----------
    community_matrix : pd.DataFrame
        Community matrix with plots as rows and species as columns.
    plot_metadata : pd.DataFrame, optional
        Plot metadata to merge with diversity metrics.

    Returns
    -------
    pd.DataFrame
        DataFrame containing:

        Plot_ID
        Richness
        Shannon
        Simpson
        Pielou

        plus metadata columns if supplied.
    """

    if community_matrix.empty:
        raise ValueError("Community matrix is empty.")

    results = []

    for plot_id, row in community_matrix.iterrows():

        abundances = row.to_numpy(dtype=float)

        richness = species_richness(abundances)
        shannon = shannon_index(abundances)
        simpson = simpson_index(abundances)
        evenness = pielou_evenness(richness, shannon)

        results.append(
            {
                "Plot_ID": plot_id,
                "Richness": richness,
                "Shannon": shannon,
                "Simpson": simpson,
                "Pielou": evenness,
            }
        )

    alpha_df = pd.DataFrame(results)

    if plot_metadata is not None:

        alpha_df = alpha_df.merge(
            plot_metadata,
            on="Plot_ID",
            how="left"
        )

    return alpha_df
    
def summarize_alpha_diversity(
    alpha_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Generate summary statistics for alpha diversity metrics.

    Parameters
    ----------
    alpha_df : pd.DataFrame
        Output from calculate_alpha_diversity().

    Returns
    -------
    dict
        Dictionary containing summary tables:
            - Overall
            - Zone
            - Habitat
    """

    metrics = [
        "Richness",
        "Shannon",
        "Simpson",
        "Pielou",
    ]

    summaries = {}

    # ----------------------------------------------------------
    # Overall summary
    # ----------------------------------------------------------

    overall = (
        alpha_df[metrics]
        .agg(["mean", "std", "min", "max"])
        .T
        .round(4)
    )

    summaries["Overall"] = overall

    # ----------------------------------------------------------
    # By Zone
    # ----------------------------------------------------------

    if "Zone" in alpha_df.columns:

        zone_summary = (
            alpha_df
            .groupby("Zone")[metrics]
            .agg(["mean", "std"])
            .round(4)
        )

        summaries["Zone"] = zone_summary

    # ----------------------------------------------------------
    # By Habitat
    # ----------------------------------------------------------

    if "Habitat" in alpha_df.columns:

        habitat_summary = (
            alpha_df
            .groupby("Habitat")[metrics]
            .agg(["mean", "std"])
            .round(4)
        )

        summaries["Habitat"] = habitat_summary

    return summaries
    
def export_alpha_diversity(
    alpha_df: pd.DataFrame,
    summaries: dict[str, pd.DataFrame],
    output_file: Path,
) -> None:
    """
    Export alpha diversity results to Excel.

    Parameters
    ----------
    alpha_df : pd.DataFrame
        Alpha diversity metrics.

    summaries : dict
        Output from summarize_alpha_diversity().

    output_file : Path
        Destination Excel file.
    """

    output_file = Path(output_file)

    with pd.ExcelWriter(output_file) as writer:

        alpha_df.to_excel(
            writer,
            sheet_name="Alpha_Diversity",
            index=False
        )

        for sheet_name, table in summaries.items():

            table.to_excel(
                writer,
                sheet_name=sheet_name
            )

    print(f"✓ Alpha diversity exported to {output_file}")