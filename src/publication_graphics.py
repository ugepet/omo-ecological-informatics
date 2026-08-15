
"""
publication_graphics.py
=======================

Publication-quality plotting utilities for the
Omo Forest Ecological Informatics Project.

Version: 0.1
"""

from pathlib import Path
import string

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
from scipy.spatial import ConvexHull

# ---------------------------------------------------------------------
# Stable Project Color Palettes
# ---------------------------------------------------------------------

# Okabe-Ito-inspired, colour-blind-friendly palette.
#
# These mappings are intentionally fixed so that ecological categories
# retain the same colours across all figures produced by the toolkit.

ZONE_COLORS = {
    "Core": "#0072B2",
    "Buffer": "#009E73",
    "Transition": "#E69F00",
}


HABITAT_COLORS = {
    "Major River": "#0072B2",
    "Stream": "#56B4E9",
    "Upland": "#009E73",
}


# Generic colour-blind-friendly fallback palette for grouping variables
# that do not yet have a project-specific colour mapping.

FALLBACK_COLORS = [
    "#0072B2",
    "#009E73",
    "#E69F00",
    "#56B4E9",
    "#D55E00",
    "#CC79A7",
    "#F0E442",
    "#000000",
]


def zone_palette(zones):
    """
    Return stable colours for ecological zones.

    Known Omo Forest zones always receive their predefined project
    colours. Unknown categories are shown in grey.

    Parameters
    ----------
    zones : iterable
        Zone labels present in the dataset.

    Returns
    -------
    dict
        Mapping of zone labels to colours.
    """

    return {
        zone: ZONE_COLORS.get(
            zone,
            "#808080",
        )
        for zone in sorted(set(zones))
    }


def habitat_palette(habitats):
    """
    Return stable colours for habitat categories.

    Known Omo Forest habitats always receive their predefined project
    colours. Unknown categories are shown in grey.

    Parameters
    ----------
    habitats : iterable
        Habitat labels present in the dataset.

    Returns
    -------
    dict
        Mapping of habitat labels to colours.
    """

    return {
        habitat: HABITAT_COLORS.get(
            habitat,
            "#808080",
        )
        for habitat in sorted(set(habitats))
    }


def categorical_palette(
    groups,
    grouping=None,
):
    """
    Return a stable colour palette for categorical grouping variables.

    Project-specific colours are used automatically for Zone and Habitat.
    Other grouping variables receive deterministic colours from the
    fallback palette.

    Parameters
    ----------
    groups : iterable
        Category labels present in the dataset.

    grouping : str, optional
        Name of the grouping variable.

    Returns
    -------
    dict
        Mapping of category labels to colours.
    """

    groups = sorted(
        set(groups)
    )

    if grouping is not None:

        grouping_key = (
            str(grouping)
            .strip()
            .lower()
        )

        if grouping_key == "zone":

            return zone_palette(
                groups
            )

        if grouping_key == "habitat":

            return habitat_palette(
                groups
            )

    return {
        group: FALLBACK_COLORS[
            i % len(FALLBACK_COLORS)
        ]
        for i, group in enumerate(groups)
    }

# ---------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------

def set_publication_theme():
    """Apply a publication-ready plotting theme."""
    sns.set_theme(style="white")

    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "figure.titlesize": 16,
        "legend.frameon": False,
        "axes.linewidth": 1.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "savefig.dpi": 300
    })


# ---------------------------------------------------------------------
# Figure creation
# ---------------------------------------------------------------------

def create_figure(nrows=1, ncols=1, figsize=(8,6), constrained=True):
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        constrained_layout=constrained
    )
    return fig, axes


# ---------------------------------------------------------------------
# Panel labels
# ---------------------------------------------------------------------

def add_panel_label(ax, label):
    ax.text(
        -0.08,
        1.04,
        f"({label})",
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="bottom"
    )


def label_panels(axes):
    axes = np.ravel(axes)
    for ax, lab in zip(axes, string.ascii_uppercase):
        add_panel_label(ax, lab)


# ---------------------------------------------------------------------
# Statistics box
# ---------------------------------------------------------------------

def add_statistics_box(ax, text, loc=(0.02,0.98)):
    ax.text(
        loc[0],
        loc[1],
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.85,
            edgecolor="gray"
        )
    )


# ---------------------------------------------------------------------
# Convex hull
# ---------------------------------------------------------------------

def draw_convex_hull(ax, x, y, color, alpha=0.18, linewidth=2):
    pts = np.column_stack([x, y])

    if pts.shape[0] < 3:
        return

    try:
        hull = ConvexHull(pts)
    except Exception:
        return

    hull_pts = pts[hull.vertices]

    ax.fill(
        hull_pts[:,0],
        hull_pts[:,1],
        color=color,
        alpha=alpha,
        zorder=0
    )

    closed = np.vstack([hull_pts, hull_pts[0]])

    ax.plot(
        closed[:,0],
        closed[:,1],
        color=color,
        linewidth=linewidth
    )


# ---------------------------------------------------------------------
# Centroids
# ---------------------------------------------------------------------

def draw_centroid(ax, x, y, color):
    ax.scatter(
        np.mean(x),
        np.mean(y),
        marker="P",
        s=220,
        color=color,
        edgecolor="white",
        linewidth=1.2,
        zorder=10
    )


# ---------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------

def style_legend(ax, **kwargs):
    leg = ax.legend(**kwargs)
    if leg is not None:
        leg.get_frame().set_linewidth(0)
    return leg


# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------

def save_publication_figure(fig, filename, output_dir):
    """
    Save PNG, PDF and SVG versions of a figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    filename : str (without extension)
    output_dir : Path or str
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for ext in ("png", "pdf", "svg"):
        fig.savefig(
            output_dir / f"{filename}.{ext}",
            dpi=300,
            bbox_inches="tight"
        )

# ==========================================================
# Alpha Diversity Plot
# ==========================================================

def plot_alpha_metric(
    data,
    metric,
    grouping,
    ax,
    palette=None
):
    """
    Create a publication-quality alpha diversity boxplot with
    jittered observations.

    Parameters
    ----------
    data : pandas.DataFrame
        Alpha diversity results.

    metric : str
        Diversity metric to plot.

    grouping : str
        Column used for grouping ('Zone' or 'Habitat').

    ax : matplotlib.axes.Axes
        Target axis.

    palette : dict or str, optional
        Color palette.
    """

    
    if palette is None:
        if grouping == "Zone":

            palette = zone_palette(
                data[grouping].unique()
            )
        elif grouping == "Habitat":
            
            palette = habitat_palette(
                data[grouping].unique()
            )
        else:
            palette = "Set2"

    sns.boxplot(
        data=data,
        x=grouping,
        y=metric,
        palette=palette,
        showfliers=False,
        width=0.6,
        linewidth=1.2,
        ax=ax
    )

    sns.stripplot(
        data=data,
        x=grouping,
        y=metric,
        color="black",
        size=4,
        jitter=0.18,
        alpha=0.6,
        ax=ax
    )

    ax.set_title(
        metric,
        fontsize=13,
        fontweight="bold"
    )

    ax.set_xlabel("")
    ax.set_ylabel(metric)

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.30
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
