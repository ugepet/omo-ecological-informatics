"""
===============================================================================
Omo Forest Ecological Informatics Project

Module:
    core.py

Purpose:
    Core result containers and shared validation utilities for the
    Omo Ecological Informatics Toolkit.

Version:
    1.0
===============================================================================
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class AnalysisResult:
    """
    Standard container for analytical outputs produced by the
    Omo Ecological Informatics Toolkit.

    The class provides a consistent interface for storing
    analysis parameters, results, diagnostics, and fitted
    models across analytical modules.

    Parameters
    ----------
    analysis : str
        Name of the analysis performed.

    version : str, default="1.0"
        Version of the analysis implementation.

    parameters : dict
        Parameters used to perform the analysis.

    results : dict
        Main analytical outputs.

    diagnostics : dict
        Diagnostic and quality-control information.

    model : Any, optional
        Fitted statistical or machine-learning model, where
        applicable.
    """

    analysis: str

    version: str = "1.0"

    parameters: Dict[str, Any] = field(
        default_factory=dict
    )

    results: Dict[str, Any] = field(
        default_factory=dict
    )

    diagnostics: Dict[str, Any] = field(
        default_factory=dict
    )

    model: Optional[Any] = None

    # ==========================================================
    # Convenience Methods
    # ==========================================================

    def keys(self):
        """
        Return the names of all stored result objects.

        Returns
        -------
        list
            Names of objects stored in results.
        """

        return list(
            self.results.keys()
        )

    def get(
        self,
        key: str,
        default: Any = None,
    ):
        """
        Retrieve a stored result object by name.

        Parameters
        ----------
        key : str
            Name of the stored result.

        default : Any, optional
            Value returned if the requested key does not exist.

        Returns
        -------
        Any
            Stored result object or the supplied default value.
        """

        return self.results.get(
            key,
            default,
        )

    def summary(self):
        """
        Print a concise summary of the analysis.
        """

        print(
            "=" * 70
        )

        print(
            f"Analysis : {self.analysis}"
        )

        print(
            f"Version  : {self.version}"
        )

        print(
            "=" * 70
        )

        # ------------------------------------------------------
        # Parameters
        # ------------------------------------------------------

        print(
            "\nParameters"
        )

        if self.parameters:

            for key, value in self.parameters.items():

                print(
                    f"  {key}: {value}"
                )

        else:

            print(
                "  None"
            )

        # ------------------------------------------------------
        # Results
        # ------------------------------------------------------

        print(
            "\nResults"
        )

        if self.results:

            for key in self.results:

                print(
                    f"  • {key}"
                )

        else:

            print(
                "  None"
            )

        # ------------------------------------------------------
        # Diagnostics
        # ------------------------------------------------------

        print(
            "\nDiagnostics"
        )

        if self.diagnostics:

            for key, value in self.diagnostics.items():

                print(
                    f"  {key}: {value}"
                )

        else:

            print(
                "  None"
            )


def validate_community_data(
    community_matrix: pd.DataFrame,
    metadata: Optional[pd.DataFrame] = None,
    plot_column: str = "Plot_ID",
) -> bool:
    """
    Validate a community matrix and optional plot metadata.

    The community matrix must contain plots as rows and species
    as columns, with Plot_ID values stored in the index.

    When metadata are supplied, Plot_ID values are validated by
    identifier rather than by row order.

    Parameters
    ----------
    community_matrix : pd.DataFrame
        Community matrix with Plot_ID values as the index and
        species abundances as columns.

    metadata : pd.DataFrame, optional
        Plot-level metadata containing the Plot_ID column.

    plot_column : str, default="Plot_ID"
        Name of the plot identifier column in metadata.

    Returns
    -------
    bool
        True when validation succeeds.

    Raises
    ------
    TypeError
        If community_matrix or metadata is not a pandas DataFrame.

    ValueError
        If the community matrix is empty, contains duplicate
        identifiers, invalid abundance values, or does not align
        with supplied metadata.
    """

    # ==========================================================
    # Community Matrix Type
    # ==========================================================

    if not isinstance(
        community_matrix,
        pd.DataFrame,
    ):

        raise TypeError(
            "community_matrix must be a pandas DataFrame."
        )

    # ==========================================================
    # Community Matrix Dimensions
    # ==========================================================

    if community_matrix.empty:

        raise ValueError(
            "community_matrix is empty."
        )

    if community_matrix.shape[0] < 1:

        raise ValueError(
            "community_matrix must contain at least one plot."
        )

    if community_matrix.shape[1] < 1:

        raise ValueError(
            "community_matrix must contain at least one species."
        )

    # ==========================================================
    # Plot_ID Index Validation
    # ==========================================================

    if community_matrix.index.has_duplicates:

        duplicate_plots = (
            community_matrix.index[
                community_matrix.index.duplicated()
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "Duplicate Plot_ID values detected in "
            "community_matrix index: "
            f"{duplicate_plots}"
        )

    if community_matrix.index.isna().any():

        raise ValueError(
            "community_matrix index contains missing Plot_ID "
            "values."
        )

    # ==========================================================
    # Numeric Community Data
    # ==========================================================

    try:

        matrix = community_matrix.to_numpy(
            dtype=float
        )

    except (TypeError, ValueError) as exc:

        raise ValueError(
            "community_matrix must contain only numeric "
            "species abundance values."
        ) from exc

    # ==========================================================
    # Finite Values
    # ==========================================================

    if not np.isfinite(
        matrix
    ).all():

        raise ValueError(
            "community_matrix contains NaN or infinite values."
        )

    # ==========================================================
    # Non-negative Abundances
    # ==========================================================

    if (
        matrix < 0
    ).any():

        raise ValueError(
            "community_matrix contains negative abundances."
        )

    # ==========================================================
    # Metadata Validation
    # ==========================================================

    if metadata is not None:

        if not isinstance(
            metadata,
            pd.DataFrame,
        ):

            raise TypeError(
                "metadata must be a pandas DataFrame."
            )

        if metadata.empty:

            raise ValueError(
                "metadata is empty."
            )

        if plot_column not in metadata.columns:

            raise ValueError(
                f"metadata must contain a "
                f"'{plot_column}' column."
            )

        # ------------------------------------------------------
        # Missing Plot IDs
        # ------------------------------------------------------

        if metadata[
            plot_column
        ].isna().any():

            raise ValueError(
                f"metadata contains missing "
                f"'{plot_column}' values."
            )

        # ------------------------------------------------------
        # Duplicate Plot IDs
        # ------------------------------------------------------

        if metadata[
            plot_column
        ].duplicated().any():

            duplicate_plots = (
                metadata.loc[
                    metadata[
                        plot_column
                    ].duplicated(
                        keep=False
                    ),
                    plot_column,
                ]
                .unique()
                .tolist()
            )

            raise ValueError(
                f"Duplicate '{plot_column}' values detected "
                f"in metadata: {duplicate_plots}"
            )

        # ------------------------------------------------------
        # Plot_ID Agreement
        # ------------------------------------------------------

        community_plots = set(
            community_matrix.index
        )

        metadata_plots = set(
            metadata[
                plot_column
            ]
        )

        missing_from_metadata = (
            community_plots
            - metadata_plots
        )

        missing_from_community = (
            metadata_plots
            - community_plots
        )

        if missing_from_metadata:

            raise ValueError(
                "The following Plot_ID values are present in "
                "community_matrix but missing from metadata: "
                f"{sorted(missing_from_metadata)}"
            )

        if missing_from_community:

            raise ValueError(
                "The following Plot_ID values are present in "
                "metadata but missing from community_matrix: "
                f"{sorted(missing_from_community)}"
            )

    # ==========================================================
    # Validation Successful
    # ==========================================================

    return True