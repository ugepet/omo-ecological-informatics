"""
===============================================================================
Omo Forest Ecological Informatics Project

Module:
    ordination.py

Purpose:
    Ordination, multivariate hypothesis testing, and hierarchical
    clustering for ecological community data.

Version:
    1.0

Main Analyses
-------------
- Non-metric Multidimensional Scaling (NMDS)
- PERMANOVA
- PERMDISP
- Principal Component Analysis (PCA)
- Hierarchical clustering
- Publication-quality ordination and dendrogram plots

Author:
    Peter Ugege et al.

===============================================================================
"""

# =============================================================================
# STANDARD LIBRARY
# =============================================================================

import time
import warnings

from datetime import datetime
from pathlib import Path
from typing import Optional, Union


# =============================================================================
# SCIENTIFIC COMPUTING
# =============================================================================

import numpy as np
import pandas as pd


# =============================================================================
# SCIPY
# =============================================================================

from scipy.cluster.hierarchy import (
    cophenet,
    dendrogram,
    fcluster,
    linkage,
)

from scipy.spatial import (
    ConvexHull,
    QhullError,
)

from scipy.spatial.distance import squareform

from scipy.stats import chi2


# =============================================================================
# SCIKIT-LEARN
# =============================================================================

from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler


# =============================================================================
# SCIKIT-BIO
# =============================================================================

from skbio import DistanceMatrix

from skbio.stats.distance import (
    permanova,
    permdisp,
)


# =============================================================================
# MATPLOTLIB
# =============================================================================

import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.colors import to_hex
from matplotlib.patches import (
    Ellipse,
    Polygon,
)


# =============================================================================
# LOCAL IMPORTS
# =============================================================================

from .core import (
    AnalysisResult,
    validate_community_data,
)

from .distances import (
    compute_distance_matrix,
    validate_distance_matrix,
)

from .publication_graphics import categorical_palette

# =============================================================================
# NON-METRIC MULTIDIMENSIONAL SCALING (NMDS)
# =============================================================================
def run_nmds(
    community_matrix: pd.DataFrame,
    metadata: pd.DataFrame | None = None,
    metric: str = "braycurtis",
    n_components: int = 2,
    random_state: int = 42,
    max_iter: int = 300,
    n_init: int = 10,
) -> AnalysisResult:
    """
    Perform Non-metric Multidimensional Scaling (NMDS).

    Parameters
    ----------
    community_matrix : pd.DataFrame
        Community abundance matrix with plots as rows and
        species as columns.

    metadata : pd.DataFrame, optional
        Plot-level metadata containing Plot_ID.

    metric : str, default="braycurtis"
        Distance metric used to calculate community dissimilarity.

    n_components : int, default=2
        Number of NMDS dimensions.

    random_state : int, default=42
        Random seed for reproducibility.

    max_iter : int, default=300
        Maximum number of optimization iterations.

    n_init : int, default=10
        Number of independent NMDS initializations.

    Returns
    -------
    AnalysisResult
        Standardized NMDS result containing coordinates,
        distance matrix, diagnostics, parameters, and fitted model.
    """

    # ==========================================================
    # Parameter Validation
    # ==========================================================

    if not isinstance(n_components, int) or n_components < 1:

        raise ValueError(
            "n_components must be an integer greater than "
            "or equal to 1."
        )

    if not isinstance(max_iter, int) or max_iter < 1:

        raise ValueError(
            "max_iter must be an integer greater than "
            "or equal to 1."
        )

    if not isinstance(n_init, int) or n_init < 1:

        raise ValueError(
            "n_init must be an integer greater than "
            "or equal to 1."
        )

    if (
        random_state is not None
        and not isinstance(random_state, int)
    ):

        raise TypeError(
            "random_state must be an integer or None."
        )

    # ==========================================================
    # Community Data Validation
    # ==========================================================

    if metadata is not None:

        validate_community_data(
            community_matrix,
            metadata,
        )

    else:

        if not isinstance(
            community_matrix,
            pd.DataFrame,
        ):

            raise TypeError(
                "community_matrix must be a pandas DataFrame."
            )

        if community_matrix.empty:

            raise ValueError(
                "Community matrix is empty."
            )

        if community_matrix.index.has_duplicates:

            raise ValueError(
                "Duplicate Plot_ID values detected in "
                "community matrix index."
            )

        duplicate_species = (
            community_matrix.columns[
                community_matrix.columns.duplicated()
            ]
            .tolist()
        )

        if duplicate_species:

            raise ValueError(
                "Duplicate species columns detected in "
                f"community matrix: {duplicate_species}"
            )

        try:

            matrix = community_matrix.to_numpy(
                dtype=float
            )

        except (TypeError, ValueError) as exc:

            raise ValueError(
                "Community matrix must contain only numeric "
                "species abundance values."
            ) from exc

        if not np.isfinite(
            matrix
        ).all():

            raise ValueError(
                "Community matrix contains NaN or "
                "infinite values."
            )

        if (
            matrix < 0
        ).any():

            raise ValueError(
                "Negative abundance values detected in "
                "community matrix."
            )

    # ==========================================================
    # Zero-Abundance Samples
    # ==========================================================

    zero_abundance_plots = (
        community_matrix
        .sum(axis=1)
        .loc[lambda x: x == 0]
        .index
        .tolist()
    )

    if zero_abundance_plots:

        if metric.lower() == "braycurtis":

            raise ValueError(
                "NMDS using Bray-Curtis distance cannot safely "
                "proceed because the following plots have zero "
                "total abundance: "
                f"{zero_abundance_plots}. "
                "Investigate or remove these samples before "
                "running NMDS."
            )

        warnings.warn(
            "One or more plots contain zero total abundance: "
            f"{zero_abundance_plots}. "
            "Some distance metrics may not be appropriate "
            "for these samples.",
            UserWarning,
        )

    # ==========================================================
    # Distance Matrix
    # ==========================================================

    distance_result = compute_distance_matrix(
        community_matrix=community_matrix,
        metric=metric,
    )

    distance_matrix = (
        distance_result.results[
            "distance_matrix"
        ]
    )

    # ==========================================================
    # Validate Distance Matrix
    # ==========================================================

    validate_distance_matrix(
        distance_matrix
    )

    if distance_matrix.shape[0] < 2:

        raise ValueError(
            "NMDS requires at least two sampling plots."
        )

    if n_components >= distance_matrix.shape[0]:

        raise ValueError(
            "n_components must be smaller than the number "
            "of sampling plots."
        )

    # ==========================================================
    # NMDS
    # ==========================================================

    start_time = time.perf_counter()

    model = MDS(
        n_components=n_components,
        metric=False,
        dissimilarity="precomputed",
        random_state=random_state,
        max_iter=max_iter,
        n_init=n_init,
    )

    coordinates_array = model.fit_transform(
        distance_matrix.to_numpy(
            dtype=float
        )
    )

    runtime = (
        time.perf_counter()
        - start_time
    )

    # ==========================================================
    # Coordinate DataFrame
    # ==========================================================

    coordinate_columns = [
        f"NMDS{i + 1}"
        for i in range(n_components)
    ]

    coordinates = pd.DataFrame(
        coordinates_array,
        index=community_matrix.index.copy(),
        columns=coordinate_columns,
    )

    coordinates.index.name = (
        community_matrix.index.name
    )

    # ==========================================================
    # Diagnostics
    # ==========================================================

    diagnostics = {
        "stress":
            float(model.stress_),
        "iterations":
            getattr(
                model,
                "n_iter_",
                None,
            ),
        "runtime_seconds":
            runtime,
        "n_samples":
            community_matrix.shape[0],
        "n_species":
            community_matrix.shape[1],
        "distance_metric":
            metric,
        "timestamp":
            datetime.now().isoformat(
                timespec="seconds"
            ),
    }

    # ==========================================================
    # Return Standardized Result
    # ==========================================================

    return AnalysisResult(
        analysis="NMDS",
        version="1.0",
        parameters={
            "distance_metric": metric,
            "n_components": n_components,
            "random_state": random_state,
            "max_iter": max_iter,
            "n_init": n_init,
        },
        results={
            "coordinates": coordinates,
            "distance_matrix": distance_matrix,
        },
        diagnostics=diagnostics,
        model=model,
    )
# =============================================================================
# NMDS Plot Preparation
# =============================================================================

def _prepare_nmds_dataframe(
    coordinates: pd.DataFrame,
    metadata: pd.DataFrame,
    grouping: str | None = None,
    plot_column: str = "Plot_ID"
) -> pd.DataFrame:
    """
    Prepare NMDS coordinates and metadata for plotting.

    The function aligns metadata to NMDS coordinates using Plot_ID
    rather than relying on row order.

    Parameters
    ----------
    coordinates : pd.DataFrame
        NMDS coordinate dataframe. Plot_ID values must be stored
        in the index.

    metadata : pd.DataFrame
        Plot-level metadata containing the Plot_ID column.

    grouping : str, optional
        Metadata column used for grouping observations in plots.

    plot_column : str, default="Plot_ID"
        Name of the plot identifier column in metadata.

    Returns
    -------
    pd.DataFrame
        DataFrame containing NMDS coordinates and aligned metadata.

    Raises
    ------
    TypeError
        If coordinates or metadata is not a DataFrame.

    ValueError
        If required identifiers, coordinates, or grouping variables
        are missing or invalid.
    """

    # ==========================================================
    # Input Validation
    # ==========================================================

    if not isinstance(coordinates, pd.DataFrame):

        raise TypeError(
            "coordinates must be a pandas DataFrame."
        )

    if not isinstance(metadata, pd.DataFrame):

        raise TypeError(
            "metadata must be a pandas DataFrame."
        )

    if coordinates.empty:

        raise ValueError(
            "NMDS coordinates dataframe is empty."
        )

    if metadata.empty:

        raise ValueError(
            "Metadata dataframe is empty."
        )

    # ==========================================================
    # Coordinate Index Validation
    # ==========================================================

    if coordinates.index.has_duplicates:

        raise ValueError(
            "Duplicate Plot_ID values detected in NMDS "
            "coordinate index."
        )

    # ==========================================================
    # Metadata Plot_ID Validation
    # ==========================================================

    if plot_column not in metadata.columns:

        raise ValueError(
            f"Metadata must contain '{plot_column}' column."
        )

    if metadata[plot_column].duplicated().any():

        raise ValueError(
            f"Duplicate '{plot_column}' values detected "
            "in metadata."
        )

    # ==========================================================
    # Coordinate Column Validation
    # ==========================================================

    coordinate_columns = [
        column
        for column in coordinates.columns
        if str(column).startswith("NMDS")
    ]

    if not coordinate_columns:

        raise ValueError(
            "No NMDS coordinate columns were found. "
            "Expected columns such as 'NMDS1' and 'NMDS2'."
        )

    # ==========================================================
    # Metadata Copy
    # ==========================================================

    metadata_aligned = metadata.copy()

    # ==========================================================
    # Set Plot_ID as Metadata Index
    # ==========================================================

    metadata_aligned = metadata_aligned.set_index(
        plot_column
    )

    # ==========================================================
    # Verify Plot_ID Agreement
    # ==========================================================

    coordinate_plots = set(
        coordinates.index
    )

    metadata_plots = set(
        metadata_aligned.index
    )

    missing_from_metadata = (
        coordinate_plots - metadata_plots
    )

    missing_from_coordinates = (
        metadata_plots - coordinate_plots
    )

    if missing_from_metadata:

        raise ValueError(
            "The following Plot_ID values are present in "
            "NMDS coordinates but missing from metadata: "
            f"{sorted(missing_from_metadata)}"
        )

    if missing_from_coordinates:

        raise ValueError(
            "The following Plot_ID values are present in "
            "metadata but missing from NMDS coordinates: "
            f"{sorted(missing_from_coordinates)}"
        )

    # ==========================================================
    # Align Metadata to NMDS Coordinates
    # ==========================================================

    metadata_aligned = metadata_aligned.loc[
        coordinates.index
    ]

    # ==========================================================
    # Validate Grouping Variable
    # ==========================================================

    if grouping is not None:

        if grouping not in metadata_aligned.columns:

            raise ValueError(
                f"Grouping column '{grouping}' not found "
                "in metadata."
            )

        if metadata_aligned[grouping].isna().any():

            missing_groups = (
                metadata_aligned.index[
                    metadata_aligned[grouping].isna()
                ]
                .tolist()
            )

            raise ValueError(
                f"Missing values detected in grouping column "
                f"'{grouping}' for Plot_ID values: "
                f"{missing_groups}"
            )

    # ==========================================================
    # Combine Coordinates and Metadata
    # ==========================================================

    plotting_data = pd.concat(
        [
            coordinates.copy(),
            metadata_aligned
        ],
        axis=1
    )

    # ==========================================================
    # Final Alignment Check
    # ==========================================================

    if not plotting_data.index.equals(
        coordinates.index
    ):

        raise ValueError(
            "Final NMDS coordinate and metadata alignment "
            "failed."
        )

    # ==========================================================
    # Return Plotting Data
    # ==========================================================

    return plotting_data

# =============================================================================
# Colour Palettes
# =============================================================================

def _get_palette(
    groups,
    grouping: str | None = None,
):
    """
    Return a stable colour palette for ordination groups.

    Project-specific colours are used automatically for known
    ecological grouping variables such as Zone and Habitat.
    Other grouping variables receive deterministic fallback colours.

    Parameters
    ----------
    groups : iterable
        Category labels present in the ordination.

    grouping : str, optional
        Name of the metadata grouping variable.

    Returns
    -------
    dict
        Mapping of group labels to colours.
    """

    return categorical_palette(
        groups=groups,
        grouping=grouping,
    )
    
# =============================================================================
# Group Boundaries
# =============================================================================
def _draw_group_boundary(
    ax,
    x,
    y,
    colour,
    boundary="convex",
    ellipse_level=0.68,
):
    """
    Draw a descriptive boundary around an NMDS group.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib axes on which the boundary will be drawn.

    x, y : array-like
        Two-dimensional ordination coordinates for the group.

    colour : str
        Colour used for the boundary.

    boundary : str, default="convex"
        Boundary type.

        Options
        -------
        "none"
            Do not draw a boundary.

        "convex"
            Draw a convex hull around the observations.

        "ellipse"
            Draw a covariance ellipse representing the specified
            probability level under a bivariate normal approximation.

        "concave"
            Reserved for future implementation.

    ellipse_level : float, default=0.68
        Probability level used to scale the covariance ellipse.

        The default value of 0.68 represents approximately 68%
        probability under a bivariate normal approximation.

    Returns
    -------
    None

    Notes
    -----
    Boundaries are descriptive visualizations of group structure
    in ordination space. They should not automatically be interpreted
    as statistical confidence regions.

    Convex hulls require at least three observations.

    Covariance ellipses require at least three observations and a
    non-singular two-dimensional covariance matrix.
    """

    # ==========================================================
    # Validate Boundary Method
    # ==========================================================

    valid_boundaries = {
        "none",
        "convex",
        "ellipse",
        "concave",
    }

    if boundary not in valid_boundaries:

        raise ValueError(
            "boundary must be one of: "
            "'none', 'convex', 'ellipse', or 'concave'."
        )

    # ==========================================================
    # Validate Ellipse Level
    # ==========================================================

    if not (
        isinstance(ellipse_level, (int, float))
        and 0 < ellipse_level < 1
    ):

        raise ValueError(
            "ellipse_level must be a numeric value "
            "strictly between 0 and 1."
        )

    # ==========================================================
    # Convert Coordinates
    # ==========================================================

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # ==========================================================
    # Coordinate Validation
    # ==========================================================

    if x.ndim != 1 or y.ndim != 1:

        raise ValueError(
            "x and y coordinates must be one-dimensional."
        )

    if len(x) != len(y):

        raise ValueError(
            "x and y coordinates must contain the same "
            "number of observations."
        )

    if not (
        np.isfinite(x).all()
        and np.isfinite(y).all()
    ):

        raise ValueError(
            "x and y coordinates contain NaN or infinite values."
        )

    # ==========================================================
    # No Boundary
    # ==========================================================

    if boundary == "none":

        return

    # ==========================================================
    # Future Concave Boundary
    # ==========================================================

    if boundary == "concave":

        raise NotImplementedError(
            "Concave hulls will be added in version 2."
        )

    # ==========================================================
    # Minimum Observations
    # ==========================================================

    if len(x) < 3:

        return

    # ==========================================================
    # Convex Hull
    # ==========================================================

    if boundary == "convex":

        points = np.column_stack(
            (x, y)
        )

        try:

            hull = ConvexHull(points)

        except QhullError:

            # Degenerate or coincident coordinates
            return

        polygon = Polygon(

            points[hull.vertices],

            closed=True,

            fill=False,

            edgecolor=colour,

            linewidth=2,

            alpha=0.9,

            zorder=2

        )

        ax.add_patch(polygon)

        return

    # ==========================================================
    # Covariance Ellipse
    # ==========================================================

    if boundary == "ellipse":

        cov = np.cov(
            x,
            y
        )

        # ------------------------------------------------------
        # Check covariance matrix
        # ------------------------------------------------------

        if cov.shape != (2, 2):

            return

        if not np.isfinite(cov).all():

            return

        if np.linalg.matrix_rank(cov) < 2:

            return

        # ------------------------------------------------------
        # Eigen decomposition
        # ------------------------------------------------------

        vals, vecs = np.linalg.eigh(
            cov
        )

        # ------------------------------------------------------
        # Sort eigenvalues from largest to smallest
        # ------------------------------------------------------

        order = vals.argsort()[::-1]

        vals = vals[order]

        vecs = vecs[:, order]

        # ------------------------------------------------------
        # Protect against numerical problems
        # ------------------------------------------------------

        if np.any(vals < 0):

            return

        # ------------------------------------------------------
        # Ellipse orientation
        # ------------------------------------------------------

        theta = np.degrees(
            np.arctan2(
                vecs[1, 0],
                vecs[0, 0]
            )
        )

        # ------------------------------------------------------
        # Probability scaling
        # ------------------------------------------------------

        scale = np.sqrt(
            chi2.ppf(
                ellipse_level,
                df=2
            )
        )

        width = (
            2
            * scale
            * np.sqrt(vals[0])
        )

        height = (
            2
            * scale
            * np.sqrt(vals[1])
        )

        # ------------------------------------------------------
        # Draw ellipse
        # ------------------------------------------------------

        ellipse = Ellipse(

            (
                x.mean(),
                y.mean()
            ),

            width,

            height,

            angle=theta,

            fill=True,

            facecolor=colour,

            edgecolor=colour,

            alpha=0.10,

            linewidth=2.2,

            zorder=2

        )

        ax.add_patch(
            ellipse
        )

        return
# =============================================================================
# Draw NMDS Groups
# =============================================================================

def _draw_groups(
    ax,
    plot_df,
    grouping,
    palette=None,
    boundary="convex",
    show_centroids=True,
    show_spiders=True,
):
    """
    Draw NMDS groups, centroids, spider lines, and group boundaries.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib axes on which the groups will be drawn.

    plot_df : pd.DataFrame
        DataFrame containing NMDS coordinates and grouping information.

    grouping : str
        Column containing group labels.

    palette : dict, optional
        Mapping of group names to colours. If None, a palette is
        generated automatically.

    boundary : str, default="convex"
        Group boundary method.

        Options
        -------
        "none"
            No group boundary.

        "convex"
            Convex hull.

        "ellipse"
            Covariance ellipse.

        "concave"
            Reserved for future implementation.

    show_centroids : bool, default=True
        Whether to display group centroids and labels.

    show_spiders : bool, default=True
        Whether to draw lines from observations to their group
        centroid.

    Returns
    -------
    dict[str, tuple[float, float]]
        Dictionary containing the NMDS-space centroid coordinates
        for each group.
    """

    # ==========================================================
    # Input Validation
    # ==========================================================

    if not isinstance(plot_df, pd.DataFrame):

        raise TypeError(
            "plot_df must be a pandas DataFrame."
        )

    if plot_df.empty:

        raise ValueError(
            "plot_df is empty."
        )

    if grouping not in plot_df.columns:

        raise ValueError(
            f"Grouping column '{grouping}' not found "
            "in plot_df."
        )

    required_coordinates = {
        "NMDS1",
        "NMDS2",
    }

    missing_coordinates = (
        required_coordinates
        - set(plot_df.columns)
    )

    if missing_coordinates:

        raise ValueError(
            "Missing NMDS coordinate columns: "
            f"{sorted(missing_coordinates)}"
        )

    # ==========================================================
    # Validate Group Values
    # ==========================================================

    if plot_df[grouping].isna().any():

        raise ValueError(
            f"Missing values detected in grouping column "
            f"'{grouping}'."
        )

    groups = sorted(
        plot_df[grouping].unique(),
        key=str
    )

    if not groups:

        raise ValueError(
            f"No groups found in '{grouping}'."
        )

    # ==========================================================
    # Prepare Palette
    # ==========================================================

    if palette is None:

        palette = _get_palette(groups)

    else:

        if not isinstance(palette, dict):

            raise TypeError(
                "palette must be a dictionary mapping group "
                "names to colours."
            )

        missing_colours = [
            group_name
            for group_name in groups
            if group_name not in palette
        ]

        if missing_colours:

            raise ValueError(
                "Palette is missing colours for groups: "
                f"{missing_colours}"
            )

    # ==========================================================
    # Calculate Y-axis Label Offset
    # ==========================================================

    y_range = (
        plot_df["NMDS2"].max()
        - plot_df["NMDS2"].min()
    )

    if y_range == 0:

        y_offset = 0.05

    else:

        y_offset = 0.02 * y_range

    # ==========================================================
    # Group Plotting
    # ==========================================================

    centroids = {}

    for group_name in groups:

        group = plot_df.loc[
            plot_df[grouping] == group_name
        ].copy()

        if group.empty:
            continue

        colour = palette[group_name]

        # ------------------------------------------------------
        # Scatter Points
        # ------------------------------------------------------

        ax.scatter(

            group["NMDS1"],

            group["NMDS2"],

            s=85,

            color=colour,

            edgecolor="black",

            linewidth=0.5,

            alpha=0.80,

            label=str(group_name),

            zorder=4,

        )

        # ------------------------------------------------------
        # Group Boundary
        # ------------------------------------------------------

        _draw_group_boundary(

            ax=ax,

            x=group["NMDS1"],

            y=group["NMDS2"],

            colour=colour,

            boundary=boundary,

        )

        # ------------------------------------------------------
        # Group Centroid
        # ------------------------------------------------------

        cx = group["NMDS1"].mean()

        cy = group["NMDS2"].mean()

        centroids[group_name] = (
            float(cx),
            float(cy)
        )

        # ------------------------------------------------------
        # Spider Lines
        # ------------------------------------------------------

        if show_spiders:

            for _, row in group.iterrows():

                ax.plot(

                    [row["NMDS1"], cx],

                    [row["NMDS2"], cy],

                    color=colour,

                    linewidth=0.30,

                    alpha=0.08,

                    zorder=1,

                )

    # ==========================================================
    # Plot Centroids
    # ==========================================================

    if show_centroids:

        for group_name, (cx, cy) in centroids.items():

            ax.scatter(

                cx,

                cy,

                marker="X",

                s=200,

                color="black",

                edgecolor="white",

                linewidth=1.3,

                zorder=6,

            )

            ax.text(

                cx,

                cy + y_offset,

                str(group_name),

                fontsize=12,

                fontweight="bold",

                ha="center",

                va="bottom",

                zorder=7,

            )

    return centroids
# =============================================================================
# NMDS Plotting
# =============================================================================

def plot_nmds(
    nmds_result: AnalysisResult,
    metadata: pd.DataFrame,
    grouping: str = "Habitat",
    boundary: str = "ellipse",
    show_centroids: bool = True,
    show_spiders: bool = True,
    palette: dict | None = None,
    annotate: bool = True,
    permanova: AnalysisResult | None = None,
    figsize: tuple = (9, 8),
    title: str | None = None,
    save_path=None,
    dpi: int = 600,
):
    """
    Create a publication-quality NMDS ordination plot.

    Parameters
    ----------
    nmds_result : AnalysisResult
        Output from run_nmds().

    metadata : DataFrame
        Plot metadata.

    grouping : str
        Metadata column used for colouring groups.

    boundary : {"none", "convex", "ellipse"}

    show_centroids : bool
        Draw group centroids.

    show_spiders : bool
        Draw spider lines from observations to centroids.

    palette : dict or None
        Optional colour palette.

    annotate : bool
        Display stress and summary statistics.

    permanova : AnalysisResult or None
        Optional PERMANOVA results to include in annotation.

    figsize : tuple

    title : str or None

    save_path : str or Path

    dpi : int

    Returns
    -------
    fig, ax
    """

    # ------------------------------------------------------------
    # Validate boundary
    # ------------------------------------------------------------

    valid = {"none", "convex", "ellipse"}

    if boundary not in valid:
        raise ValueError(
            f"boundary must be one of {valid}"
        )

    # ------------------------------------------------------------
    # Validate NMDS result
    # ------------------------------------------------------------

    if "coordinates" not in nmds_result.results:
        raise ValueError(
            "nmds_result does not contain NMDS coordinates."
        )

    # ------------------------------------------------------------
    # Prepare plotting dataframe
    # ------------------------------------------------------------

    plot_df = _prepare_nmds_dataframe(
        coordinates=nmds_result.results["coordinates"],
        metadata=metadata,
        grouping=grouping,
    )

    # ------------------------------------------------------------
    # Colour palette
    # ------------------------------------------------------------

    if palette is None:
        palette = _get_palette(
            plot_df[grouping].unique(),
            grouping=grouping,
        )

    # ------------------------------------------------------------
    # Create figure
    # ------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=figsize
    )

    # ------------------------------------------------------------
    # Draw groups
    # ------------------------------------------------------------

    _draw_groups(
        ax=ax,
        plot_df=plot_df,
        grouping=grouping,
        palette=palette,
        boundary=boundary,
        show_centroids=show_centroids,
        show_spiders=show_spiders,
    )

    # ------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------

    ax.set_xlabel(
        "NMDS1",
        fontsize=13,
        fontweight="bold"
    )

    ax.set_ylabel(
        "NMDS2",
        fontsize=13,
        fontweight="bold"
    )

    if title is None:
        title = (
            f"NMDS Ordination of Woody Plant Communities by {grouping}"
        )

    ax.set_title(
        title,
        fontsize=15,
        fontweight="bold",
        pad=20,
    )

    # ------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------

    ax.legend(
        title=grouping,
        bbox_to_anchor=(1.03, 1.00),
        loc="upper left",
        frameon=True,
    )

    # ------------------------------------------------------------
    # Statistics annotation
    # ------------------------------------------------------------

    if annotate:

        d = nmds_result.diagnostics

        annotation = (
            "Distance : Bray–Curtis\n"
            f"Stress   : {d['stress']:.3f}\n"
        )

        # --------------------------------------------------------
        # PERMANOVA (optional)
        # --------------------------------------------------------

        if permanova is not None:

            if "permanova" not in permanova.results:
                raise ValueError(
                    "permanova result does not contain "
                    "'permanova' statistics."
                )

            p = permanova.results["permanova"]

            annotation += (
                f"Pseudo-F : {p['test statistic']:.2f}\n"
                f"p-value  : {p['p-value']:.4f}\n"
            )

        annotation += (
            f"\nPlots    : {d['n_samples']}"
            f"\nSpecies  : {d['n_species']}"
        )

        ax.text(
            0.98,
            0.02,
            annotation,
            transform=ax.transAxes,
            fontsize=10,
            ha="right",
            va="bottom",
            bbox=dict(
                facecolor="white",
                edgecolor="gray",
                alpha=0.90,
                boxstyle="round",
            ),
        )

    # ------------------------------------------------------------
    # Styling
    # ------------------------------------------------------------

    ax.grid(
        linestyle="--",
        linewidth=0.4,
        alpha=0.20,
    )

    ax.set_axisbelow(True)

    fig.tight_layout()

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    if save_path is not None:

        fig.savefig(
            save_path,
            dpi=dpi,
            bbox_inches="tight",
        )

    # ------------------------------------------------------------
    # Return figure
    # ------------------------------------------------------------

    return fig, ax
    
# =============================================================================
# PERMANOVA
# =============================================================================

def run_permanova(
    distance_matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    grouping_column: str,
    permutations: int = 999,
    random_state: int = 42,
):
    """
    Perform PERMANOVA on a precomputed community distance matrix.

    Metadata are aligned to the distance matrix using Plot_ID
    rather than relying on row order.

    Parameters
    ----------
    distance_matrix : pd.DataFrame
        Square symmetric distance matrix with Plot_ID values
        stored in both the index and columns.

    metadata : pd.DataFrame
        Plot-level metadata containing Plot_ID and the selected
        grouping variable.

    grouping_column : str
        Metadata column defining the groups to compare.

    permutations : int, default=999
        Number of permutations used to assess statistical
        significance.

    random_state : int, default=42
        Random seed used for reproducibility.

    Returns
    -------
    AnalysisResult
        AnalysisResult containing the PERMANOVA statistics,
        analysis parameters, and diagnostic information.

    Raises
    ------
    TypeError
        If metadata is not a pandas DataFrame.

    ValueError
        If metadata are empty, required columns are missing,
        Plot_ID values are duplicated or misaligned, grouping
        values are missing, fewer than two groups are present,
        or permutations is invalid.
    """

    start = time.perf_counter()

    # ==========================================================
    # Validate Distance Matrix
    # ==========================================================

    validate_distance_matrix(
        distance_matrix
    )

    # ==========================================================
    # Metadata Validation
    # ==========================================================

    if not isinstance(metadata, pd.DataFrame):

        raise TypeError(
            "metadata must be a pandas DataFrame."
        )

    if metadata.empty:

        raise ValueError(
            "Metadata dataframe is empty."
        )

    if "Plot_ID" not in metadata.columns:

        raise ValueError(
            "Metadata must contain a 'Plot_ID' column."
        )

    if metadata["Plot_ID"].duplicated().any():

        raise ValueError(
            "Duplicate Plot_ID values detected in metadata."
        )

    if grouping_column not in metadata.columns:

        raise ValueError(
            f"Grouping column '{grouping_column}' "
            "not found in metadata."
        )

    # ==========================================================
    # Parameter Validation
    # ==========================================================

    if not isinstance(permutations, int):

        raise TypeError(
            "permutations must be an integer."
        )

    if permutations < 1:

        raise ValueError(
            "permutations must be at least 1."
        )

    if not isinstance(random_state, int):

        raise TypeError(
            "random_state must be an integer."
        )

    # ==========================================================
    # Align Metadata to Distance Matrix
    # ==========================================================

    metadata_aligned = (
        metadata
        .copy()
        .set_index("Plot_ID")
    )

    distance_plots = set(
        distance_matrix.index
    )

    metadata_plots = set(
        metadata_aligned.index
    )

    missing_from_metadata = (
        distance_plots - metadata_plots
    )

    missing_from_distance = (
        metadata_plots - distance_plots
    )

    if missing_from_metadata:

        raise ValueError(
            "The following Plot_ID values are present in the "
            "distance matrix but missing from metadata: "
            f"{sorted(missing_from_metadata)}"
        )

    if missing_from_distance:

        raise ValueError(
            "The following Plot_ID values are present in "
            "metadata but missing from the distance matrix: "
            f"{sorted(missing_from_distance)}"
        )

    metadata_aligned = metadata_aligned.loc[
        distance_matrix.index
    ]

    # ==========================================================
    # Validate Grouping Variable
    # ==========================================================

    if metadata_aligned[
        grouping_column
    ].isna().any():

        missing_groups = (
            metadata_aligned.index[
                metadata_aligned[
                    grouping_column
                ].isna()
            ]
            .tolist()
        )

        raise ValueError(
            f"Missing values detected in grouping column "
            f"'{grouping_column}' for Plot_ID values: "
            f"{missing_groups}"
        )

    n_groups = metadata_aligned[
        grouping_column
    ].nunique()

    if n_groups < 2:

        raise ValueError(
            "PERMANOVA requires at least two groups."
        )

    # ==========================================================
    # Group Sizes
    # ==========================================================

    group_sizes = (
        metadata_aligned[
            grouping_column
        ]
        .value_counts()
        .to_dict()
    )

    # ==========================================================
    # Build scikit-bio Distance Matrix
    # ==========================================================

    matrix = np.ascontiguousarray(
        distance_matrix.to_numpy(
            dtype=float
        )
    )

    dm = DistanceMatrix(
        matrix,
        ids=distance_matrix.index.astype(str),
    )

    grouping = (
        metadata_aligned[
            grouping_column
        ]
        .to_numpy()
    )

    # ==========================================================
    # Reproducibility
    # ==========================================================

    np.random.seed(
        random_state
    )

    # ==========================================================
    # Run PERMANOVA
    # ==========================================================

    permanova_result = permanova(
        distance_matrix=dm,
        grouping=grouping,
        permutations=permutations,
    )

    # ==========================================================
    # Runtime
    # ==========================================================

    runtime = (
        time.perf_counter()
        - start
    )

    # ==========================================================
    # Return Analysis Result
    # ==========================================================

    return AnalysisResult(
        analysis="PERMANOVA",
        version="1.0",
        parameters={
            "grouping_column": grouping_column,
            "permutations": permutations,
            "random_state": random_state,
        },
        results={
            "permanova": permanova_result,
            "group_sizes": group_sizes,
        },
        diagnostics={
            "runtime_seconds": runtime,
            "n_samples": len(
                metadata_aligned
            ),
            "n_groups": n_groups,
            "timestamp": datetime.now().isoformat(
                timespec="seconds"
            ),
        },
        model=permanova_result,
    )
# =============================================================================
# PERMDISP (Homogeneity of Multivariate Dispersion)
# =============================================================================

def run_permdisp(
    community_matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    grouping_column: str,
    distance_metric: str = "braycurtis",
    permutations: int = 999,
    random_state: int = 42,
    test: str = "median",
) -> AnalysisResult:
    """
    Test homogeneity of multivariate dispersions (PERMDISP).

    PERMDISP evaluates whether multivariate dispersion differs
    among predefined groups using a community distance matrix.

    Parameters
    ----------
    community_matrix : pd.DataFrame
        Community matrix with Plot_ID values stored in the index
        and species abundances stored in columns.

    metadata : pd.DataFrame
        Plot-level metadata containing Plot_ID and the selected
        grouping variable.

    grouping_column : str
        Metadata column defining the groups to compare.

    distance_metric : str, default="braycurtis"
        Distance metric used to calculate ecological dissimilarity.

    permutations : int, default=999
        Number of permutations used to assess statistical
        significance.

    random_state : int, default=42
        Random seed used for reproducibility.

    test : {"median", "centroid"}, default="median"
        Location estimator used by PERMDISP. The spatial median
        provides a more robust default, while "centroid" uses
        group centroids.

    Returns
    -------
    AnalysisResult
        AnalysisResult containing the PERMDISP statistics,
        distance matrix, analysis parameters, and diagnostics.

    Raises
    ------
    TypeError
        If metadata is not a pandas DataFrame or parameters have
        invalid types.

    ValueError
        If required metadata fields are missing, Plot_ID values
        are duplicated or misaligned, grouping values are missing,
        fewer than two groups are present, or analysis parameters
        are invalid.
    """

    start = time.perf_counter()

    # ==========================================================
    # Validate Community Data
    # ==========================================================

    validate_community_data(
        community_matrix,
        metadata,
    )

    # ==========================================================
    # Metadata Validation
    # ==========================================================

    if not isinstance(metadata, pd.DataFrame):

        raise TypeError(
            "metadata must be a pandas DataFrame."
        )

    if metadata.empty:

        raise ValueError(
            "Metadata dataframe is empty."
        )

    if "Plot_ID" not in metadata.columns:

        raise ValueError(
            "Metadata must contain a 'Plot_ID' column."
        )

    if metadata["Plot_ID"].duplicated().any():

        raise ValueError(
            "Duplicate Plot_ID values detected in metadata."
        )

    if grouping_column not in metadata.columns:

        raise ValueError(
            f"Grouping column '{grouping_column}' "
            "not found in metadata."
        )

    # ==========================================================
    # Parameter Validation
    # ==========================================================

    if not isinstance(permutations, int):

        raise TypeError(
            "permutations must be an integer."
        )

    if permutations < 1:

        raise ValueError(
            "permutations must be at least 1."
        )

    if not isinstance(random_state, int):

        raise TypeError(
            "random_state must be an integer."
        )

    valid_tests = {
        "median",
        "centroid",
    }

    if test not in valid_tests:

        raise ValueError(
            f"test must be one of {valid_tests}."
        )

    # ==========================================================
    # Compute Distance Matrix
    # ==========================================================

    distance_result = compute_distance_matrix(
        community_matrix,
        metric=distance_metric,
    )

    distance_matrix = (
        distance_result.results[
            "distance_matrix"
        ]
    )

    validate_distance_matrix(
        distance_matrix
    )

    # ==========================================================
    # Align Metadata to Distance Matrix
    # ==========================================================

    metadata_aligned = (
        metadata
        .copy()
        .set_index("Plot_ID")
    )

    distance_plots = set(
        distance_matrix.index
    )

    metadata_plots = set(
        metadata_aligned.index
    )

    missing_from_metadata = (
        distance_plots - metadata_plots
    )

    missing_from_distance = (
        metadata_plots - distance_plots
    )

    if missing_from_metadata:

        raise ValueError(
            "The following Plot_ID values are present in the "
            "distance matrix but missing from metadata: "
            f"{sorted(missing_from_metadata)}"
        )

    if missing_from_distance:

        raise ValueError(
            "The following Plot_ID values are present in "
            "metadata but missing from the distance matrix: "
            f"{sorted(missing_from_distance)}"
        )

    metadata_aligned = metadata_aligned.loc[
        distance_matrix.index
    ]

    # ==========================================================
    # Validate Grouping Variable
    # ==========================================================

    if metadata_aligned[
        grouping_column
    ].isna().any():

        missing_groups = (
            metadata_aligned.index[
                metadata_aligned[
                    grouping_column
                ].isna()
            ]
            .tolist()
        )

        raise ValueError(
            f"Missing values detected in grouping column "
            f"'{grouping_column}' for Plot_ID values: "
            f"{missing_groups}"
        )

    n_groups = metadata_aligned[
        grouping_column
    ].nunique()

    if n_groups < 2:

        raise ValueError(
            "PERMDISP requires at least two groups."
        )

    # ==========================================================
    # Group Sizes
    # ==========================================================

    group_sizes = (
        metadata_aligned[
            grouping_column
        ]
        .value_counts()
        .to_dict()
    )

    # ==========================================================
    # Build scikit-bio Distance Matrix
    # ==========================================================

    matrix = np.ascontiguousarray(
        distance_matrix.to_numpy(
            dtype=float
        )
    )

    dm = DistanceMatrix(
        matrix,
        ids=distance_matrix.index.astype(str),
    )

    grouping = (
        metadata_aligned[
            grouping_column
        ]
        .to_numpy()
    )

    # ==========================================================
    # Run PERMDISP
    # ==========================================================

    permdisp_result = permdisp(
        distmat=dm,
        grouping=grouping,
        test=test,
        permutations=permutations,
        method="eigh",
        seed=random_state,
    )

    # ==========================================================
    # Runtime
    # ==========================================================

    runtime = (
        time.perf_counter()
        - start
    )

    # ==========================================================
    # Return Analysis Result
    # ==========================================================

    return AnalysisResult(
        analysis="PERMDISP",
        version="1.0",
        parameters={
            "grouping_column": grouping_column,
            "distance_metric": distance_metric,
            "permutations": permutations,
            "random_state": random_state,
            "test": test,
        },
        results={
            "permdisp": permdisp_result,
            "distance_matrix": distance_matrix,
            "group_sizes": group_sizes,
        },
        diagnostics={
            "runtime_seconds": runtime,
            "n_samples": len(
                metadata_aligned
            ),
            "n_groups": n_groups,
            "timestamp": datetime.now().isoformat(
                timespec="seconds"
            ),
        },
        model=permdisp_result,
    )
    
def run_pca(
    community_matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    grouping_column: Optional[str] = None,
    transform: str = "hellinger",
    scale_data: bool = False,
    n_components: Optional[int] = None,
) -> AnalysisResult:
    """
    Perform Principal Component Analysis (PCA) on community data.

    Parameters
    ----------
    community_matrix : pd.DataFrame
        Community matrix with plots as rows, species as columns,
        and Plot_ID values stored in the index.

    metadata : pd.DataFrame
        Plot-level metadata containing Plot_ID.

    grouping_column : str, optional
        Metadata column describing plot groups, such as Habitat
        or Zone.

    transform : {"hellinger", "relative", "log", "none"},
        default="hellinger"
        Data transformation applied before PCA.

    scale_data : bool, default=False
        Whether to standardize species variables to unit variance
        before PCA.

    n_components : int, optional
        Number of principal components to retain. If None, all
        possible components are retained.

    Returns
    -------
    AnalysisResult
        AnalysisResult containing PCA scores, species coefficients,
        species loadings, variance summaries, transformed data,
        diagnostics, and fitted PCA model.

    Raises
    ------
    TypeError
        If n_components is not an integer when supplied.

    ValueError
        If transformation is invalid, n_components is outside the
        allowable range, or grouping metadata are invalid.
    """

    # ==========================================================
    # Validate Community Data
    # ==========================================================

    validate_community_data(
        community_matrix,
        metadata,
    )

    # ==========================================================
    # Transformation Validation
    # ==========================================================

    valid_transforms = {
        "hellinger",
        "relative",
        "log",
        "none",
    }

    if not isinstance(transform, str):

        raise TypeError(
            "transform must be a string."
        )

    transform = transform.lower()

    if transform not in valid_transforms:

        raise ValueError(
            "transform must be one of "
            "['hellinger', 'relative', 'log', 'none']."
        )

    # ==========================================================
    # Grouping Validation
    # ==========================================================

    metadata_aligned = (
        metadata
        .copy()
        .set_index("Plot_ID")
        .loc[community_matrix.index]
    )

    if grouping_column is not None:

        if grouping_column not in metadata_aligned.columns:

            raise ValueError(
                f"Grouping column '{grouping_column}' "
                "not found in metadata."
            )

        if metadata_aligned[
            grouping_column
        ].isna().any():

            missing_groups = (
                metadata_aligned.index[
                    metadata_aligned[
                        grouping_column
                    ].isna()
                ]
                .tolist()
            )

            raise ValueError(
                f"Missing values detected in grouping column "
                f"'{grouping_column}' for Plot_ID values: "
                f"{missing_groups}"
            )

    # ==========================================================
    # Prepare Numeric Community Matrix
    # ==========================================================

    X = community_matrix.astype(
        float
    ).copy()

    # ==========================================================
    # Data Transformation
    # ==========================================================

    if transform == "hellinger":

        row_totals = X.sum(
            axis=1
        )

        relative = X.div(
            row_totals.replace(
                0,
                np.nan,
            ),
            axis=0,
        )

        X = np.sqrt(
            relative
        ).fillna(0)

    elif transform == "relative":

        row_totals = X.sum(
            axis=1
        )

        X = (
            X.div(
                row_totals.replace(
                    0,
                    np.nan,
                ),
                axis=0,
            )
            .fillna(0)
        )

    elif transform == "log":

        X = np.log1p(
            X
        )

    # transform == "none" requires no change

    # ==========================================================
    # Validate Number of Components
    # ==========================================================

    max_components = min(
        X.shape[0],
        X.shape[1],
    )

    if n_components is not None:

        if not isinstance(
            n_components,
            int,
        ):

            raise TypeError(
                "n_components must be an integer or None."
            )

        if n_components < 1:

            raise ValueError(
                "n_components must be at least 1."
            )

        if n_components > max_components:

            raise ValueError(
                "n_components cannot exceed "
                f"{max_components} for the supplied data."
            )

    # ==========================================================
    # Optional Scaling
    # ==========================================================

    scaler = None

    if scale_data:

        scaler = StandardScaler()

        X = pd.DataFrame(
            scaler.fit_transform(
                X
            ),
            index=X.index,
            columns=X.columns,
        )

    # ==========================================================
    # PCA
    # ==========================================================

    pca = PCA(
        n_components=n_components
    )

    scores = pca.fit_transform(
        X
    )

    # ==========================================================
    # PCA Score Columns
    # ==========================================================

    score_columns = [
        f"PC{i + 1}"
        for i in range(
            scores.shape[1]
        )
    ]

    # ==========================================================
    # Plot Scores
    # ==========================================================

    scores_df = pd.DataFrame(
        scores,
        index=X.index,
        columns=score_columns,
    )

    # ==========================================================
    # Attach Grouping Information
    # ==========================================================

    if grouping_column is not None:

        scores_df[
            grouping_column
        ] = metadata_aligned.loc[
            scores_df.index,
            grouping_column,
        ].values

    # ==========================================================
    # PCA Component Coefficients
    # ==========================================================

    components_df = pd.DataFrame(
        pca.components_.T,
        index=X.columns,
        columns=score_columns,
    )

    # ==========================================================
    # Species Loadings
    # ==========================================================

    loadings_df = pd.DataFrame(
        pca.components_.T
        * np.sqrt(
            pca.explained_variance_
        ),
        index=X.columns,
        columns=score_columns,
    )

    # ==========================================================
    # Variance Summary
    # ==========================================================

    variance_percent = (
        pca.explained_variance_ratio_
        * 100
    )

    cumulative_percent = np.cumsum(
        variance_percent
    )

    variance_summary = pd.DataFrame(
        {
            "Component": score_columns,
            "Eigenvalue":
                pca.explained_variance_,
            "Variance (%)":
                variance_percent,
            "Cumulative (%)":
                cumulative_percent,
        }
    )

    # ==========================================================
    # Return Analysis Result
    # ==========================================================

    return AnalysisResult(
        analysis="PCA",
        version="1.0",
        parameters={
            "transformation": transform,
            "scaled": scale_data,
            "n_components": n_components,
            "grouping_column":
                grouping_column,
        },
        results={
            "scores": scores_df,
            "components": components_df,
            "loadings": loadings_df,
            "variance_summary":
                variance_summary,
            "transformed_data":
                X.copy(),
            "explained_variance":
                pca.explained_variance_,
            "explained_variance_ratio":
                pca.explained_variance_ratio_,
            "cumulative_variance":
                np.cumsum(
                    pca.explained_variance_ratio_
                ),
        },
        diagnostics={
            "n_samples": X.shape[0],
            "n_species": X.shape[1],
            "n_components":
                scores.shape[1],
            "transformation":
                transform,
            "scaling":
                scale_data,
        },
        model=pca,
    )

def plot_pca(
    pca_result: AnalysisResult,
    metadata: pd.DataFrame,
    grouping: str,
    components: tuple = (1, 2),
    show_species: bool = True,
    top_species: int = 10,
    show_centroids: bool = True,
    boundary: str = "ellipse",
    ellipse_level: float = 0.68,
    figsize: tuple = (10, 8),
    alpha: float = 0.80,
    point_size: int = 70,
    centroid_size: int = 250,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 600,
):
    """
    Create a publication-quality PCA biplot.

    Parameters
    ----------
    pca_result : AnalysisResult
        Result returned by run_pca().

    metadata : pd.DataFrame
        Plot-level metadata containing Plot_ID.

    grouping : str
        Metadata column used for colouring ecological groups.

    components : tuple, default=(1, 2)
        Principal components to display.

    show_species : bool, default=True
        Whether to display species loading vectors.

    top_species : int, default=10
        Number of species with the largest loading magnitudes
        to display.

    show_centroids : bool, default=True
        Whether to display group centroids.

    boundary : {"ellipse", "convex", "none"}, default="ellipse"
        Type of descriptive group boundary.

    ellipse_level : float, default=0.68
        Probability level used for descriptive covariance ellipses.

    figsize : tuple, default=(10, 8)
        Figure size.

    alpha : float, default=0.80
        Transparency of sample points.

    point_size : int, default=70
        Sample-point size.

    centroid_size : int, default=250
        Group-centroid marker size.

    save_path : str or Path, optional
        Path used to save the figure.

    dpi : int, default=600
        Resolution used when saving the figure.

    Returns
    -------
    tuple
        Matplotlib figure and axes objects.
    """

    from scipy.spatial import ConvexHull
    from scipy.stats import chi2

    # ==========================================================
    # Validate PCA Result
    # ==========================================================

    required_results = {
        "scores",
        "loadings",
        "variance_summary",
    }

    missing_results = (
        required_results
        - set(pca_result.results)
    )

    if missing_results:

        raise ValueError(
            "pca_result is missing required results: "
            f"{sorted(missing_results)}"
        )

    scores = (
        pca_result.results["scores"]
        .copy()
    )

    loadings = (
        pca_result.results["loadings"]
        .copy()
    )

    variance = (
        pca_result.results["variance_summary"]
        .copy()
    )

    # ==========================================================
    # Validate Components
    # ==========================================================

    if (
        not isinstance(components, tuple)
        or len(components) != 2
    ):

        raise ValueError(
            "components must be a tuple containing exactly "
            "two component numbers."
        )

    if not all(
        isinstance(component, int)
        and component >= 1
        for component in components
    ):

        raise ValueError(
            "PCA component numbers must be positive integers."
        )

    if components[0] == components[1]:

        raise ValueError(
            "PCA components must be different."
        )

    pcx = f"PC{components[0]}"
    pcy = f"PC{components[1]}"

    for pc in (pcx, pcy):

        if pc not in scores.columns:

            raise ValueError(
                f"{pc} is not available in PCA scores."
            )

        if pc not in loadings.columns:

            raise ValueError(
                f"{pc} is not available in PCA loadings."
            )

    # ==========================================================
    # Validate Boundary
    # ==========================================================

    valid_boundaries = {
        "ellipse",
        "convex",
        "none",
    }

    if boundary not in valid_boundaries:

        raise ValueError(
            f"boundary must be one of {valid_boundaries}."
        )

    # ==========================================================
    # Validate Ellipse Level
    # ==========================================================

    if not 0 < ellipse_level < 1:

        raise ValueError(
            "ellipse_level must be between 0 and 1."
        )

    # ==========================================================
    # Validate Species Count
    # ==========================================================

    if not isinstance(top_species, int):

        raise TypeError(
            "top_species must be an integer."
        )

    if top_species < 1:

        raise ValueError(
            "top_species must be at least 1."
        )

    # ==========================================================
    # Align Grouping Information
    # ==========================================================

    if grouping not in scores.columns:

        if "Plot_ID" not in metadata.columns:

            raise ValueError(
                "Metadata must contain a 'Plot_ID' column."
            )

        if metadata["Plot_ID"].duplicated().any():

            raise ValueError(
                "Duplicate Plot_ID values detected in metadata."
            )

        if grouping not in metadata.columns:

            raise ValueError(
                f"Grouping column '{grouping}' "
                "not found in metadata."
            )

        metadata_lookup = (
            metadata
            .copy()
            .set_index("Plot_ID")
        )

        missing_plots = (
            set(scores.index)
            - set(metadata_lookup.index)
        )

        if missing_plots:

            raise ValueError(
                "The following PCA Plot_ID values are missing "
                "from metadata: "
                f"{sorted(missing_plots)}"
            )

        scores[grouping] = (
            metadata_lookup.loc[
                scores.index,
                grouping,
            ]
            .values
        )

    if scores[grouping].isna().any():

        missing_groups = (
            scores.index[
                scores[grouping].isna()
            ]
            .tolist()
        )

        raise ValueError(
            f"Missing values detected in grouping column "
            f"'{grouping}' for Plot_ID values: "
            f"{missing_groups}"
        )

    # ==========================================================
    # Colour Palette
    # ==========================================================

    palette = _get_palette(
        scores[grouping].unique(),
        grouping=grouping,
    )

    # ==========================================================
    # Create Figure
    # ==========================================================

    fig, ax = plt.subplots(
        figsize=figsize
    )

    # ==========================================================
    # Plot Sample Groups
    # ==========================================================

    for grp in scores[grouping].unique():

        subset = scores.loc[
            scores[grouping] == grp
        ]

        ax.scatter(
            subset[pcx],
            subset[pcy],
            s=point_size,
            color=palette[grp],
            edgecolor="black",
            linewidth=0.5,
            alpha=alpha,
            label=grp,
            zorder=3,
        )

        # ------------------------------------------------------
        # Group Centroid
        # ------------------------------------------------------

        cx = subset[pcx].mean()
        cy = subset[pcy].mean()

        if show_centroids:

            ax.scatter(
                cx,
                cy,
                s=centroid_size,
                marker="X",
                color=palette[grp],
                edgecolor="black",
                linewidth=1.5,
                zorder=5,
            )

            ax.annotate(
                str(grp),
                xy=(cx, cy),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=10,
                fontweight="bold",
            )

        # ------------------------------------------------------
        # Ellipse Boundary
        # ------------------------------------------------------

        if (
            boundary == "ellipse"
            and len(subset) >= 3
        ):

            points = subset[
                [pcx, pcy]
            ].to_numpy()

            cov = np.cov(
                points.T
            )

            eigenvalues, eigenvectors = (
                np.linalg.eigh(cov)
            )

            order = (
                eigenvalues
                .argsort()[::-1]
            )

            eigenvalues = (
                eigenvalues[order]
            )

            eigenvectors = (
                eigenvectors[:, order]
            )

            if np.all(
                eigenvalues >= 0
            ):

                theta = np.degrees(
                    np.arctan2(
                        eigenvectors[1, 0],
                        eigenvectors[0, 0],
                    )
                )

                scale_factor = np.sqrt(
                    chi2.ppf(
                        ellipse_level,
                        df=2,
                    )
                )

                width = (
                    2
                    * scale_factor
                    * np.sqrt(
                        eigenvalues[0]
                    )
                )

                height = (
                    2
                    * scale_factor
                    * np.sqrt(
                        eigenvalues[1]
                    )
                )

                ellipse = Ellipse(
                    xy=(cx, cy),
                    width=width,
                    height=height,
                    angle=theta,
                    facecolor=palette[grp],
                    edgecolor=palette[grp],
                    alpha=0.10,
                    linewidth=2,
                    zorder=1,
                )

                ax.add_patch(
                    ellipse
                )

        # ------------------------------------------------------
        # Convex Hull Boundary
        # ------------------------------------------------------

        elif (
            boundary == "convex"
            and len(subset) >= 3
        ):

            points = subset[
                [pcx, pcy]
            ].to_numpy()

            try:

                hull = ConvexHull(
                    points
                )

            except Exception:

                hull = None

            if hull is not None:

                hull_points = points[
                    hull.vertices
                ]

                hull_points = np.vstack(
                    [
                        hull_points,
                        hull_points[0],
                    ]
                )

                ax.plot(
                    hull_points[:, 0],
                    hull_points[:, 1],
                    color=palette[grp],
                    linewidth=2,
                    alpha=0.8,
                    zorder=2,
                )

    # ==========================================================
    # Species Loading Vectors
    # ==========================================================

    if show_species:

        magnitude = np.sqrt(
            loadings[pcx] ** 2
            + loadings[pcy] ** 2
        )

        n_species_to_show = min(
            top_species,
            len(loadings),
        )

        selected_species = (
            magnitude
            .nlargest(
                n_species_to_show
            )
            .index
        )

        # ------------------------------------------------------
        # Dynamic Arrow Scaling
        # ------------------------------------------------------

        score_x_range = (
            scores[pcx].max()
            - scores[pcx].min()
        )

        score_y_range = (
            scores[pcy].max()
            - scores[pcy].min()
        )

        loading_x_max = (
            loadings.loc[
                selected_species,
                pcx,
            ]
            .abs()
            .max()
        )

        loading_y_max = (
            loadings.loc[
                selected_species,
                pcy,
            ]
            .abs()
            .max()
        )

        scale_candidates = []

        if loading_x_max > 0:

            scale_candidates.append(
                0.35
                * score_x_range
                / loading_x_max
            )

        if loading_y_max > 0:

            scale_candidates.append(
                0.35
                * score_y_range
                / loading_y_max
            )

        arrow_scale = (
            min(scale_candidates)
            if scale_candidates
            else 1.0
        )

        for species in selected_species:

            x = (
                loadings.loc[
                    species,
                    pcx,
                ]
                * arrow_scale
            )

            y = (
                loadings.loc[
                    species,
                    pcy,
                ]
                * arrow_scale
            )

            ax.annotate(
                "",
                xy=(x, y),
                xytext=(0, 0),
                arrowprops=dict(
                    arrowstyle="->",
                    color="dimgray",
                    linewidth=1.2,
                    alpha=0.75,
                ),
                zorder=2,
            )

            ax.text(
                x * 1.08,
                y * 1.08,
                str(species),
                fontsize=8,
                ha="center",
                va="center",
            )

    # ==========================================================
    # Variance Explained
    # ==========================================================

    pcx_variance = variance.loc[
        variance["Component"] == pcx,
        "Variance (%)",
    ]

    pcy_variance = variance.loc[
        variance["Component"] == pcy,
        "Variance (%)",
    ]

    if pcx_variance.empty or pcy_variance.empty:

        raise ValueError(
            "Variance summary does not contain the "
            "requested PCA components."
        )

    pcx_var = pcx_variance.iloc[0]
    pcy_var = pcy_variance.iloc[0]

    # ==========================================================
    # Axis Labels
    # ==========================================================

    ax.set_xlabel(
        f"{pcx} ({pcx_var:.1f}%)",
        fontsize=12,
        fontweight="bold",
    )

    ax.set_ylabel(
        f"{pcy} ({pcy_var:.1f}%)",
        fontsize=12,
        fontweight="bold",
    )

    # ==========================================================
    # Title
    # ==========================================================

    ax.set_title(
        "Principal Component Analysis of "
        f"Woody Plant Communities by {grouping}",
        fontsize=15,
        fontweight="bold",
        pad=15,
    )

    # ==========================================================
    # Styling
    # ==========================================================

    ax.axhline(
        0,
        linewidth=0.7,
        color="gray",
        alpha=0.5,
    )

    ax.axvline(
        0,
        linewidth=0.7,
        color="gray",
        alpha=0.5,
    )

    ax.grid(
        linestyle="--",
        alpha=0.25,
    )

    ax.set_axisbelow(True)

    ax.legend(
        frameon=True,
        title=grouping,
        bbox_to_anchor=(1.03, 1.00),
        loc="upper left",
    )

    fig.tight_layout()

    # ==========================================================
    # Save Figure
    # ==========================================================

    if save_path is not None:

        save_path = Path(
            save_path
        )

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            save_path,
            dpi=dpi,
            bbox_inches="tight",
        )

    # ==========================================================
    # Return Figure
    # ==========================================================

    return fig, ax
    
def run_hierarchical_clustering(
    community_matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    distance_metric: str = "braycurtis",
    linkage_method: str = "average",
    transform: str = "hellinger",
    n_clusters: int = 3,
) -> AnalysisResult:
    """
    Perform hierarchical clustering on community composition.

    Parameters
    ----------
    community_matrix : pd.DataFrame
        Community matrix with Plot_ID values as the index and
        species abundances as columns.

    metadata : pd.DataFrame
        Plot-level metadata containing Plot_ID and optional
        ecological grouping variables.

    distance_metric : str, default="braycurtis"
        Distance metric used to quantify ecological dissimilarity.

    linkage_method : str, default="average"
        Hierarchical clustering linkage method.
        "average" corresponds to UPGMA and is appropriate for
        Bray-Curtis community dissimilarities.

    transform : {"hellinger", "relative", "log", "none"},
        default="hellinger"
        Transformation applied before calculating distances.

    n_clusters : int, default=3
        Number of clusters used to derive cluster membership.

    Returns
    -------
    AnalysisResult
        AnalysisResult containing the linkage matrix, cluster
        membership, summary tables, distance matrix, transformed
        data, and clustering diagnostics.
    """

    # ==========================================================
    # Validate Community Data
    # ==========================================================

    validate_community_data(
        community_matrix,
        metadata,
    )

    # ==========================================================
    # Transformation Validation
    # ==========================================================

    if not isinstance(transform, str):

        raise TypeError(
            "transform must be a string."
        )

    transform = transform.lower()

    valid_transforms = {
        "hellinger",
        "relative",
        "log",
        "none",
    }

    if transform not in valid_transforms:

        raise ValueError(
            "transform must be one of "
            "['hellinger', 'relative', 'log', 'none']."
        )

    # ==========================================================
    # Linkage Validation
    # ==========================================================

    if not isinstance(linkage_method, str):

        raise TypeError(
            "linkage_method must be a string."
        )

    linkage_method = linkage_method.lower()

    valid_linkages = {
        "single",
        "complete",
        "average",
        "weighted",
        "centroid",
        "median",
        "ward",
    }

    if linkage_method not in valid_linkages:

        raise ValueError(
            f"linkage_method must be one of "
            f"{sorted(valid_linkages)}."
        )

    # Methods with Euclidean-distance assumptions
    euclidean_only_methods = {
        "ward",
        "centroid",
        "median",
    }

    if (
        linkage_method in euclidean_only_methods
        and distance_metric.lower() != "euclidean"
    ):

        raise ValueError(
            f"linkage_method='{linkage_method}' requires "
            "Euclidean distances. Use distance_metric='euclidean' "
            "or choose a linkage method such as 'average'."
        )

    # ==========================================================
    # Cluster Number Validation
    # ==========================================================

    if not isinstance(n_clusters, int):

        raise TypeError(
            "n_clusters must be an integer."
        )

    if n_clusters < 2:

        raise ValueError(
            "n_clusters must be at least 2."
        )

    if n_clusters > community_matrix.shape[0]:

        raise ValueError(
            "n_clusters cannot exceed the number of plots."
        )

    # ==========================================================
    # Prepare Numeric Community Matrix
    # ==========================================================

    X = community_matrix.astype(
        float
    ).copy()

    # ==========================================================
    # Data Transformation
    # ==========================================================

    if transform == "hellinger":

        row_totals = X.sum(
            axis=1
        )

        relative = X.div(
            row_totals.replace(
                0,
                np.nan,
            ),
            axis=0,
        )

        X = np.sqrt(
            relative
        ).fillna(0)

    elif transform == "relative":

        row_totals = X.sum(
            axis=1
        )

        X = (
            X.div(
                row_totals.replace(
                    0,
                    np.nan,
                ),
                axis=0,
            )
            .fillna(0)
        )

    elif transform == "log":

        X = np.log1p(
            X
        )

    # transform == "none" requires no change

    # ==========================================================
    # Compute Distance Matrix
    # ==========================================================

    distance_result = compute_distance_matrix(
        X,
        metric=distance_metric,
    )

    distance_matrix = (
        distance_result.results[
            "distance_matrix"
        ]
    )

    validate_distance_matrix(
        distance_matrix
    )

    # ==========================================================
    # Condensed Distance Vector
    # ==========================================================

    distance_vector = squareform(
        distance_matrix.to_numpy(
            dtype=float
        ),
        checks=False,
    )

    # ==========================================================
    # Hierarchical Clustering
    # ==========================================================

    linkage_matrix = linkage(
        distance_vector,
        method=linkage_method,
    )

    # ==========================================================
    # Cophenetic Correlation
    # ==========================================================

    cophenetic_corr, _ = cophenet(
        linkage_matrix,
        distance_vector,
    )

    # ==========================================================
    # Cluster Membership
    # ==========================================================

    clusters = fcluster(
        linkage_matrix,
        t=n_clusters,
        criterion="maxclust",
    )

    cluster_df = pd.DataFrame(
        {
            "Cluster": clusters,
        },
        index=community_matrix.index,
    )

    cluster_df.index.name = "Plot_ID"

    # ==========================================================
    # Align Metadata
    # ==========================================================

    metadata_lookup = (
        metadata
        .copy()
        .set_index("Plot_ID")
    )

    metadata_lookup = metadata_lookup.loc[
        cluster_df.index
    ]

    cluster_df = cluster_df.join(
        metadata_lookup
    )

    # ==========================================================
    # Cluster Summary
    # ==========================================================

    cluster_summary = (
        cluster_df
        .groupby("Cluster")
        .size()
        .rename("Number_of_Plots")
        .to_frame()
    )

    # ==========================================================
    # Habitat Summary
    # ==========================================================

    habitat_summary = None
    habitat_percent = None

    if "Habitat" in cluster_df.columns:

        habitat_summary = pd.crosstab(
            cluster_df["Cluster"],
            cluster_df["Habitat"],
        )

        habitat_percent = (
            habitat_summary
            .div(
                habitat_summary.sum(
                    axis=1
                ),
                axis=0,
            )
            .mul(100)
            .round(1)
        )

    # ==========================================================
    # Zone Summary
    # ==========================================================

    zone_summary = None
    zone_percent = None

    if "Zone" in cluster_df.columns:

        zone_summary = pd.crosstab(
            cluster_df["Cluster"],
            cluster_df["Zone"],
        )

        zone_percent = (
            zone_summary
            .div(
                zone_summary.sum(
                    axis=1
                ),
                axis=0,
            )
            .mul(100)
            .round(1)
        )

    # ==========================================================
    # Cluster Sizes
    # ==========================================================

    cluster_sizes = (
        cluster_df["Cluster"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    # ==========================================================
    # Diagnostics
    # ==========================================================

    diagnostics = {
        "n_samples": X.shape[0],
        "n_species": X.shape[1],
        "distance_metric": distance_metric,
        "linkage_method": linkage_method,
        "transformation": transform,
        "cophenetic_correlation":
            float(cophenetic_corr),
        "n_clusters": int(n_clusters),
    }

    # ==========================================================
    # Results
    # ==========================================================

    results = {
        "linkage_matrix":
            linkage_matrix,
        "cluster_membership":
            cluster_df,
        "cluster_summary":
            cluster_summary,
        "cluster_sizes":
            cluster_sizes,
        "habitat_summary":
            habitat_summary,
        "habitat_percent":
            habitat_percent,
        "zone_summary":
            zone_summary,
        "zone_percent":
            zone_percent,
        "distance_matrix":
            distance_matrix,
        "distance_vector":
            distance_vector,
        "transformed_data":
            X.copy(),
    }

    # ==========================================================
    # Return AnalysisResult
    # ==========================================================

    return AnalysisResult(
        analysis="Hierarchical Clustering",
        version="1.0",
        parameters={
            "distance_metric":
                distance_metric,
            "linkage_method":
                linkage_method,
            "transform":
                transform,
            "n_clusters":
                n_clusters,
        },
        results=results,
        diagnostics=diagnostics,
        model=linkage_matrix,
    )
    
def plot_dendrogram(
    cluster_result: AnalysisResult,
    metadata: pd.DataFrame,
    grouping: str = "Habitat",
    figsize: tuple = (14, 8),
    color_labels: bool = True,
    show_cut_line: bool = True,
    cut_clusters: int | None = None,
    title: str | None = None,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 600,
):
    """
    Plot a publication-quality hierarchical clustering dendrogram.

    Parameters
    ----------
    cluster_result : AnalysisResult
        Output returned by run_hierarchical_clustering().

    metadata : pd.DataFrame
        Plot-level metadata containing Plot_ID and the selected
        grouping variable.

    grouping : str, default="Habitat"
        Metadata column used to colour dendrogram leaf labels.

    figsize : tuple, default=(14, 8)
        Figure size.

    color_labels : bool, default=True
        Whether to colour leaf labels according to grouping.

    show_cut_line : bool, default=True
        Whether to draw a horizontal line representing the
        selected number of clusters.

    cut_clusters : int, optional
        Number of clusters represented by the horizontal cut line.
        If None, the number used in run_hierarchical_clustering()
        is used.

    title : str, optional
        Figure title. If None, a title is generated from the
        clustering method.

    save_path : str or Path, optional
        Path used to save the figure.

    dpi : int, default=600
        Resolution used when saving the figure.

    Returns
    -------
    tuple
        Matplotlib figure and axes objects.
    """

    # ==========================================================
    # Validate Cluster Result
    # ==========================================================

    if not isinstance(cluster_result, AnalysisResult):

        raise TypeError(
            "cluster_result must be an AnalysisResult returned "
            "by run_hierarchical_clustering()."
        )

    required_results = {
        "linkage_matrix",
        "cluster_membership",
    }

    missing_results = (
        required_results
        - set(cluster_result.results)
    )

    if missing_results:

        raise ValueError(
            "cluster_result is missing required results: "
            f"{sorted(missing_results)}"
        )

    linkage_matrix = (
        cluster_result.results[
            "linkage_matrix"
        ]
    )

    cluster_membership = (
        cluster_result.results[
            "cluster_membership"
        ]
    )

    # ==========================================================
    # Validate Metadata
    # ==========================================================

    if not isinstance(metadata, pd.DataFrame):

        raise TypeError(
            "metadata must be a pandas DataFrame."
        )

    if metadata.empty:

        raise ValueError(
            "Metadata dataframe is empty."
        )

    if "Plot_ID" not in metadata.columns:

        raise ValueError(
            "Metadata must contain a 'Plot_ID' column."
        )

    if metadata["Plot_ID"].duplicated().any():

        raise ValueError(
            "Duplicate Plot_ID values detected in metadata."
        )

    if grouping not in metadata.columns:

        raise ValueError(
            f"Grouping column '{grouping}' "
            "not found in metadata."
        )

    # ==========================================================
    # Recover Plot Order Used for Clustering
    # ==========================================================

    plot_order = (
        cluster_membership.index
        .tolist()
    )

    metadata_lookup = (
        metadata
        .copy()
        .set_index("Plot_ID")
    )

    missing_plots = (
        set(plot_order)
        - set(metadata_lookup.index)
    )

    if missing_plots:

        raise ValueError(
            "The following clustered Plot_ID values are missing "
            "from metadata: "
            f"{sorted(missing_plots)}"
        )

    metadata_aligned = metadata_lookup.loc[
        plot_order
    ]

    if metadata_aligned[grouping].isna().any():

        missing_groups = (
            metadata_aligned.index[
                metadata_aligned[
                    grouping
                ].isna()
            ]
            .tolist()
        )

        raise ValueError(
            f"Missing values detected in grouping column "
            f"'{grouping}' for Plot_ID values: "
            f"{missing_groups}"
        )

    # ==========================================================
    # Clustering Parameters
    # ==========================================================

    distance_metric = (
        cluster_result.parameters.get(
            "distance_metric",
            "distance",
        )
    )

    linkage_method = (
        cluster_result.parameters.get(
            "linkage_method",
            "hierarchical",
        )
    )

    fitted_clusters = (
        cluster_result.parameters.get(
            "n_clusters",
            None,
        )
    )

    if cut_clusters is None:

        cut_clusters = fitted_clusters

    # ==========================================================
    # Validate Cut Clusters
    # ==========================================================

    n_samples = len(
        plot_order
    )

    if show_cut_line:

        if cut_clusters is None:

            raise ValueError(
                "cut_clusters must be supplied when the "
                "cluster result does not contain n_clusters."
            )

        if not isinstance(
            cut_clusters,
            int,
        ):

            raise TypeError(
                "cut_clusters must be an integer."
            )

        if cut_clusters < 2:

            raise ValueError(
                "cut_clusters must be at least 2."
            )

        if cut_clusters >= n_samples:

            raise ValueError(
                "cut_clusters must be smaller than the "
                "number of plots."
            )

    # ==========================================================
    # Colour Palette
    # ==========================================================

    palette = _get_palette(
        metadata_aligned[
            grouping
        ].unique()
    )

    # ==========================================================
    # Create Figure
    # ==========================================================

    fig, ax = plt.subplots(
        figsize=figsize
    )

    # ==========================================================
    # Draw Dendrogram
    # ==========================================================

    dend = dendrogram(
        linkage_matrix,
        labels=[
            str(plot_id)
            for plot_id in plot_order
        ],
        leaf_rotation=90,
        leaf_font_size=8,
        color_threshold=0,
        above_threshold_color="black",
        ax=ax,
    )

    # ==========================================================
    # Colour Leaf Labels
    # ==========================================================

    if color_labels:

        for tick in ax.get_xmajorticklabels():

            plot_id = tick.get_text()

            if plot_id not in metadata_lookup.index.astype(str):

                continue

            # Metadata index may not originally be string type
            matches = (
                metadata_lookup.index.astype(str)
                == plot_id
            )

            original_plot_id = (
                metadata_lookup.index[
                    matches
                ][0]
            )

            group = metadata_lookup.loc[
                original_plot_id,
                grouping,
            ]

            tick.set_color(
                palette[group]
            )

            tick.set_fontweight(
                "bold"
            )

    # ==========================================================
    # Draw Cluster Cut Line
    # ==========================================================

    if show_cut_line:

        heights = linkage_matrix[
            :, 2
        ]

        # For k clusters, the cut lies between the merge that
        # leaves k clusters and the next merge that produces k - 1.
        lower_height = heights[
            -cut_clusters
        ]

        upper_height = heights[
            -cut_clusters + 1
        ]

        threshold = (
            lower_height
            + upper_height
        ) / 2

        ax.axhline(
            threshold,
            linestyle="--",
            linewidth=1.5,
            color="red",
            label=f"{cut_clusters} clusters",
        )

    # ==========================================================
    # Title
    # ==========================================================

    if title is None:

        method_label = (
            "UPGMA"
            if linkage_method == "average"
            else linkage_method.title()
        )

        title = (
            f"Hierarchical Cluster Analysis ({method_label})"
        )

    ax.set_title(
        title,
        fontsize=16,
        fontweight="bold",
        pad=18,
    )

    # ==========================================================
    # Axis Labels
    # ==========================================================

    metric_label = (
        str(distance_metric)
        .replace("_", " ")
        .replace("-", " ")
        .title()
    )

    if str(distance_metric).lower() == "braycurtis":

        metric_label = "Bray–Curtis"

    ax.set_ylabel(
        f"{metric_label} Dissimilarity",
        fontsize=12,
    )

    ax.set_xlabel(
        "Sampling Plots",
        fontsize=12,
    )

    # ==========================================================
    # Styling
    # ==========================================================

    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)

    if show_cut_line:

        ax.legend(
            frameon=False,
            loc="upper right",
        )

    fig.tight_layout()

    # ==========================================================
    # Save Figure
    # ==========================================================

    if save_path is not None:

        save_path = Path(
            save_path
        )

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            save_path,
            dpi=dpi,
            bbox_inches="tight",
        )

    # ==========================================================
    # Return Figure
    # ==========================================================

    return fig, ax