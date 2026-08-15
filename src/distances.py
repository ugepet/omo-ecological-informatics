"""
===============================================================================
Omo Forest Ecological Informatics Project

Module:
    distances.py

Purpose:
    Distance matrix computations and validation for community ecology.

Version:
    1.0
===============================================================================
"""

import numpy as np
import pandas as pd

from scipy.spatial.distance import pdist, squareform

from .core import AnalysisResult


def validate_distance_matrix(
    distance_matrix: pd.DataFrame,
    atol: float = 1e-10,
) -> None:
    """
    Validate a community distance matrix.

    Parameters
    ----------
    distance_matrix : pd.DataFrame
        Square symmetric distance matrix with Plot_ID values
        stored in both the index and columns.

    atol : float, default=1e-10
        Absolute tolerance used when checking symmetry,
        non-negative values, and zero diagonal values.

    Raises
    ------
    TypeError
        If distance_matrix is not a pandas DataFrame.

    ValueError
        If the matrix is empty, non-square, misaligned,
        contains duplicate Plot_ID values, non-numeric values,
        non-finite values, negative distances, is asymmetric,
        or has a non-zero diagonal.
    """

    # ==========================================================
    # Input Validation
    # ==========================================================

    if not isinstance(distance_matrix, pd.DataFrame):

        raise TypeError(
            "distance_matrix must be a pandas DataFrame."
        )

    if distance_matrix.empty:

        raise ValueError(
            "Distance matrix is empty."
        )

    # ==========================================================
    # Shape Validation
    # ==========================================================

    if distance_matrix.shape[0] != distance_matrix.shape[1]:

        raise ValueError(
            "Distance matrix must be square."
        )

    # ==========================================================
    # Plot_ID Validation
    # ==========================================================

    if distance_matrix.index.has_duplicates:

        raise ValueError(
            "Duplicate Plot_ID values detected in "
            "distance-matrix index."
        )

    if distance_matrix.columns.has_duplicates:

        raise ValueError(
            "Duplicate Plot_ID values detected in "
            "distance-matrix columns."
        )

    if not distance_matrix.index.equals(
        distance_matrix.columns
    ):

        raise ValueError(
            "Distance-matrix index and columns must contain "
            "the same Plot_ID values in the same order."
        )

    # ==========================================================
    # Numeric Conversion
    # ==========================================================

    try:

        matrix = distance_matrix.to_numpy(
            dtype=float
        )

    except (TypeError, ValueError) as exc:

        raise ValueError(
            "Distance matrix must contain only numeric values."
        ) from exc

    # ==========================================================
    # Finite Values
    # ==========================================================

    if not np.isfinite(matrix).all():

        raise ValueError(
            "Distance matrix contains NaN or infinite values."
        )

    # ==========================================================
    # Non-negative Distances
    # ==========================================================

    if (matrix < -atol).any():

        raise ValueError(
            "Distance matrix contains negative distances."
        )

    # ==========================================================
    # Symmetry
    # ==========================================================

    if not np.allclose(
        matrix,
        matrix.T,
        atol=atol,
        rtol=0.0,
    ):

        raise ValueError(
            "Distance matrix must be symmetric."
        )

    # ==========================================================
    # Diagonal
    # ==========================================================

    if not np.allclose(
        np.diag(matrix),
        0.0,
        atol=atol,
        rtol=0.0,
    ):

        raise ValueError(
            "Distance matrix diagonal must contain zeros."
        )


def compute_distance_matrix(
    community_matrix: pd.DataFrame,
    metric: str = "braycurtis",
):
    """
    Compute a community distance matrix.

    Parameters
    ----------
    community_matrix : pd.DataFrame
        Community matrix with Plot_ID as index and species
        abundances as columns.

    metric : str, default="braycurtis"
        Distance metric supported by
        scipy.spatial.distance.pdist.

    Returns
    -------
    AnalysisResult
        AnalysisResult containing the computed distance matrix.

    Raises
    ------
    TypeError
        If community_matrix is not a pandas DataFrame.

    ValueError
        If the community matrix is empty, contains duplicate
        Plot_ID values, non-numeric values, non-finite values,
        or negative abundance values.
    """

    # ==========================================================
    # Input Validation
    # ==========================================================

    if not isinstance(community_matrix, pd.DataFrame):

        raise TypeError(
            "community_matrix must be a pandas DataFrame."
        )

    if community_matrix.empty:

        raise ValueError(
            "Community matrix is empty."
        )

    if community_matrix.shape[0] < 2:

        raise ValueError(
            "Community matrix must contain at least two plots."
        )

    if community_matrix.shape[1] < 1:

        raise ValueError(
            "Community matrix must contain at least one species."
        )

    # ==========================================================
    # Plot_ID Validation
    # ==========================================================

    if community_matrix.index.has_duplicates:

        raise ValueError(
            "Duplicate Plot_ID values detected in "
            "community-matrix index."
        )

    # ==========================================================
    # Numeric Conversion
    # ==========================================================

    try:

        matrix = community_matrix.to_numpy(
            dtype=float
        )

    except (TypeError, ValueError) as exc:

        raise ValueError(
            "Community matrix must contain only numeric "
            "species abundance values."
        ) from exc

    # ==========================================================
    # Finite Values
    # ==========================================================

    if not np.isfinite(matrix).all():

        raise ValueError(
            "Community matrix contains NaN or infinite values."
        )

    # ==========================================================
    # Non-negative Abundances
    # ==========================================================

    if (matrix < 0).any():

        raise ValueError(
            "Community matrix contains negative abundance values."
        )

    # ==========================================================
    # Compute Distance Matrix
    # ==========================================================

    distance_vector = pdist(
        matrix,
        metric=metric,
    )

    distance_matrix = pd.DataFrame(
        squareform(distance_vector),
        index=community_matrix.index.copy(),
        columns=community_matrix.index.copy(),
    )

    # ==========================================================
    # Validate Computed Distance Matrix
    # ==========================================================

    validate_distance_matrix(
        distance_matrix
    )

    # ==========================================================
    # Return Analysis Result
    # ==========================================================

    return AnalysisResult(
        analysis="Distance Matrix",
        version="1.0",
        parameters={
            "metric": metric,
        },
        results={
            "distance_matrix": distance_matrix,
        },
        diagnostics={
            "n_samples": community_matrix.shape[0],
            "n_species": community_matrix.shape[1],
        },
        model=None,
    )