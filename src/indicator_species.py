# =============================================================================
# OMO ECOLOGICAL INFORMATICS TOOLKIT
# =============================================================================
#
# Module:
#     indicator_species.py
#
# Description:
#     Indicator Species Analysis (IndVal) following the method of
#     Dufrêne & Legendre (1997).
#
#     This module identifies species that are characteristic of ecological
#     groups (e.g., habitats or clusters) using:
#
#         • Specificity (Positive Predictive Value)
#         • Fidelity (Sensitivity)
#         • Indicator Value (IndVal)
#         • Permutation-based significance testing
#
# Version:
#     1.0.0
#
# Authors:
#     Omo Ecological Informatics Toolkit
#
# References:
#
# Dufrêne, M., & Legendre, P. (1997).
# Species assemblages and indicator species: The need for a flexible
# asymmetrical approach.
# Ecological Monographs, 67(3), 345–366.
#
# Phipson, B., & Smyth, G. K. (2010).
# Permutation p-values Should Never Be Zero.
# Statistical Applications in Genetics and Molecular Biology.
#
# =============================================================================

from __future__ import annotations

import logging
import time
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .core import (
    AnalysisResult,
    validate_community_data,
)


# =============================================================================
# LOGGER
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# INTERNAL HELPER FUNCTIONS
# =============================================================================


def _calculate_specificity(
    species_values: np.ndarray,
    groups: np.ndarray,
) -> pd.Series:
    """
    Calculate the specificity (A component) of a species.

    Specificity measures the degree to which a species is restricted
    to a particular ecological group.

    Parameters
    ----------
    species_values : np.ndarray
        Species abundance across plots.

    groups : np.ndarray
        Group assignment for each plot.

    Returns
    -------
    pd.Series
        Specificity values for each group.
    """

    df = pd.DataFrame(
        {
            "Group": groups,
            "Abundance": species_values,
        }
    )

    group_mean = (
        df.groupby("Group")["Abundance"]
        .mean()
    )

    total = group_mean.sum()

    if total == 0:
        return group_mean * 0.0

    return group_mean / total


def _calculate_fidelity(
    species_values: np.ndarray,
    groups: np.ndarray,
) -> pd.Series:
    """
    Calculate the fidelity (B component).

    Fidelity measures the proportion of sampling plots within each
    ecological group where the species is present.

    Parameters
    ----------
    species_values : np.ndarray
        Species abundance across plots.

    groups : np.ndarray
        Group assignment for each plot.

    Returns
    -------
    pd.Series
        Fidelity values for each group.
    """

    df = pd.DataFrame(
        {
            "Group": groups,
            "Presence": species_values > 0,
        }
    )

    return (
        df.groupby("Group")["Presence"]
        .mean()
        .astype(float)
    )


def _calculate_indval(
    specificity: pd.Series,
    fidelity: pd.Series,
) -> pd.Series:
    """
    Calculate Indicator Values.

    Parameters
    ----------
    specificity : pd.Series
        Specificity values.

    fidelity : pd.Series
        Fidelity values.

    Returns
    -------
    pd.Series
        Indicator Values expressed as percentages.
    """

    return specificity * fidelity * 100.0


def _best_group(
    indval: pd.Series,
) -> Tuple[object, float]:
    """
    Identify the group with the highest Indicator Value.

    Parameters
    ----------
    indval : pd.Series
        Indicator Values by group.

    Returns
    -------
    tuple
        Best group and maximum Indicator Value.
    """

    best = indval.idxmax()

    return (
        best,
        float(indval.loc[best]),
    )


def _max_indval(
    species_values: np.ndarray,
    groups: np.ndarray,
) -> Tuple[object, float, float, float]:
    """
    Compute the maximum Indicator Value for a species.

    Returns
    -------
    tuple
        Best group, specificity, fidelity, and maximum IndVal.
    """

    specificity = _calculate_specificity(
        species_values,
        groups,
    )

    fidelity = _calculate_fidelity(
        species_values,
        groups,
    )

    indval = _calculate_indval(
        specificity,
        fidelity,
    )

    best_group, best_value = _best_group(
        indval
    )

    return (
        best_group,
        float(specificity.loc[best_group]),
        float(fidelity.loc[best_group]),
        float(best_value),
    )


# =============================================================================
# INTERNAL UTILITY FUNCTIONS
# =============================================================================


def _evaluate_all_species(
    community_matrix: pd.DataFrame,
    groups: np.ndarray,
) -> pd.DataFrame:
    """
    Calculate Indicator Values for all species.

    Parameters
    ----------
    community_matrix : pd.DataFrame
        Species abundance matrix.

    groups : np.ndarray
        Group assignments.

    Returns
    -------
    pd.DataFrame
        Indicator statistics for every species.
    """

    records = []

    for species in community_matrix.columns:

        species_values = (
            community_matrix[species]
            .to_numpy(dtype=float)
        )

        mean_abundance = species_values.mean()

        occurrence = (
            (species_values > 0).mean()
            * 100
        )

        (
            best_group,
            specificity,
            fidelity,
            indval,
        ) = _max_indval(
            species_values,
            groups,
        )

        group_mask = (
            groups == best_group
        )

        group_values = (
            species_values[
                group_mask
            ]
        )

        group_mean = (
            group_values.mean()
        )

        group_occurrence = (
            (group_values > 0).mean()
            * 100
        )

        records.append(
            {
                "Species": species,
                "Best Group": best_group,
                "Overall Mean Abundance": round(
                    mean_abundance,
                    3,
                ),
                "Group Mean Abundance": round(
                    group_mean,
                    3,
                ),
                "Overall Occurrence (%)": round(
                    occurrence,
                    1,
                ),
                "Group Occurrence (%)": round(
                    group_occurrence,
                    1,
                ),
                "Specificity": round(
                    specificity,
                    3,
                ),
                "Fidelity": round(
                    fidelity,
                    3,
                ),
                "IndVal": round(
                    indval,
                    3,
                ),
            }
        )

    return pd.DataFrame(
        records
    )


def _permutation_test(
    community_matrix: pd.DataFrame,
    groups: np.ndarray,
    observed: pd.DataFrame,
    n_permutations: int = 999,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Perform permutation tests for all species simultaneously.

    Parameters
    ----------
    community_matrix : pd.DataFrame
        Species abundance matrix.

    groups : np.ndarray
        Group assignments.

    observed : pd.DataFrame
        Observed Indicator Species results.

    n_permutations : int, default=999
        Number of permutations.

    rng : np.random.Generator, optional
        Random-number generator.

    Returns
    -------
    np.ndarray
        Permutation p-values.
    """

    if rng is None:
        rng = np.random.default_rng()

    observed_values = (
        observed["IndVal"]
        .to_numpy()
    )

    counts = np.zeros(
        len(observed_values),
        dtype=int,
    )

    for i in range(
        n_permutations
    ):

        shuffled = rng.permutation(
            groups
        )

        permuted = _evaluate_all_species(
            community_matrix,
            shuffled,
        )

        counts += (
            permuted["IndVal"]
            .to_numpy()
            >= observed_values
        )

        if (
            (i + 1) % 100 == 0
            or (i + 1) == n_permutations
        ):

            logger.info(
                "Completed permutation "
                f"{i + 1:,}/{n_permutations:,}"
            )

    return (
        counts + 1
    ) / (
        n_permutations + 1
    )


# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================


def run_indicator_species_analysis(
    community_matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    grouping: str = "Habitat",
    n_permutations: int = 999,
    alpha: float = 0.05,
    random_state: int = 42,
) -> AnalysisResult:
    """
    Perform Indicator Species Analysis (IndVal).

    Parameters
    ----------
    community_matrix : pd.DataFrame
        Community matrix with Plot_ID values as the index and
        species abundances as columns.

    metadata : pd.DataFrame
        Plot-level metadata containing Plot_ID and the selected
        grouping variable.

    grouping : str, default="Habitat"
        Metadata column defining ecological groups.

    n_permutations : int, default=999
        Number of permutations used for significance testing.

    alpha : float, default=0.05
        Significance level.

    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    AnalysisResult
        Indicator Species Analysis results.
    """

    logger.info(
        "=" * 70
    )

    logger.info(
        "Running Indicator Species Analysis"
    )

    logger.info(
        "=" * 70
    )

    start_time = time.perf_counter()

    # ==========================================================
    # Validate Inputs
    # ==========================================================

    validate_community_data(
        community_matrix,
        metadata,
    )

    if grouping not in metadata.columns:

        raise ValueError(
            f"Grouping column '{grouping}' "
            "not found in metadata."
        )

    if metadata[
        grouping
    ].isna().any():

        missing_groups = (
            metadata.loc[
                metadata[grouping].isna(),
                "Plot_ID",
            ]
            .tolist()
        )

        raise ValueError(
            f"Missing values detected in grouping column "
            f"'{grouping}' for Plot_ID values: "
            f"{missing_groups}"
        )

    # ==========================================================
    # Validate Parameters
    # ==========================================================

    if not isinstance(
        n_permutations,
        int,
    ):

        raise TypeError(
            "n_permutations must be an integer."
        )

    if n_permutations < 1:

        raise ValueError(
            "n_permutations must be at least 1."
        )

    if not isinstance(
        random_state,
        int,
    ):

        raise TypeError(
            "random_state must be an integer."
        )

    if not isinstance(
        alpha,
        (int, float),
    ):

        raise TypeError(
            "alpha must be numeric."
        )

    if not 0 < alpha < 1:

        raise ValueError(
            "alpha must be between 0 and 1."
        )

    # ==========================================================
    # Align Metadata to Community Matrix
    # ==========================================================

    metadata_aligned = (
        metadata
        .copy()
        .set_index("Plot_ID")
        .loc[
            community_matrix.index
        ]
    )

    groups = (
        metadata_aligned[
            grouping
        ]
        .to_numpy()
    )

    n_groups = pd.Series(
        groups
    ).nunique()

    if n_groups < 2:

        raise ValueError(
            "Indicator Species Analysis requires "
            "at least two ecological groups."
        )

    group_sizes = (
        metadata_aligned[
            grouping
        ]
        .value_counts()
        .to_dict()
    )

    rng = np.random.default_rng(
        random_state
    )

    # ==========================================================
    # Calculate Observed Indicator Values
    # ==========================================================

    observed = _evaluate_all_species(
        community_matrix,
        groups,
    )

    # ==========================================================
    # Permutation Test
    # ==========================================================

    logger.info(
        "Running permutation test..."
    )

    p_values = _permutation_test(
        community_matrix=community_matrix,
        groups=groups,
        observed=observed,
        n_permutations=n_permutations,
        rng=rng,
    )

    observed["p-value"] = np.round(
        p_values,
        4,
    )

    observed["Significant"] = (
        observed["p-value"]
        < alpha
    )

    # ==========================================================
    # Sort Indicator Species
    # ==========================================================

    observed = (
        observed
        .sort_values(
            by=[
                "Significant",
                "p-value",
                "IndVal",
            ],
            ascending=[
                False,
                True,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    # ==========================================================
    # Significant Indicator Species
    # ==========================================================

    significant_species = (
        observed.loc[
            observed["Significant"]
        ]
        .reset_index(
            drop=True
        )
    )

    # ==========================================================
    # Group Summary
    # ==========================================================

    if significant_species.empty:

        group_summary = pd.DataFrame(
            columns=[
                "Best Group",
                "Indicator Species",
            ]
        )

    else:

        group_summary = (
            significant_species
            .groupby(
                "Best Group"
            )
            .size()
            .rename(
                "Indicator Species"
            )
            .reset_index()
        )

    # ==========================================================
    # Runtime
    # ==========================================================

    runtime = (
        time.perf_counter()
        - start_time
    )

    # ==========================================================
    # Diagnostics
    # ==========================================================

    diagnostics = {
        "grouping": grouping,
        "n_groups": n_groups,
        "n_species":
            community_matrix.shape[1],
        "n_plots":
            community_matrix.shape[0],
        "group_sizes":
            group_sizes,
        "permutations":
            n_permutations,
        "alpha":
            alpha,
        "significant_species":
            len(significant_species),
        "runtime_seconds":
            round(runtime, 2),
    }

    # ==========================================================
    # Parameters
    # ==========================================================

    parameters = {
        "grouping":
            grouping,
        "n_permutations":
            n_permutations,
        "alpha":
            alpha,
        "random_state":
            random_state,
    }

    # ==========================================================
    # Logging
    # ==========================================================

    logger.info(
        "Indicator Species Analysis completed in "
        f"{runtime:.2f} seconds. "
        f"{len(significant_species)} significant "
        "indicator species identified."
    )

    # ==========================================================
    # Return AnalysisResult
    # ==========================================================

    return AnalysisResult(
        analysis="Indicator Species Analysis",
        version="1.0",
        parameters=parameters,
        results={
            "indicator_table":
                observed,
            "significant_species":
                significant_species,
            "group_summary":
                group_summary,
            "group_sizes":
                group_sizes,
        },
        diagnostics=diagnostics,
        model=None,
    )