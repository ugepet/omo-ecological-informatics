"""
===============================================================================
Omo Forest Ecological Informatics Project

Package:
    src

Purpose:
    Public interface for the Omo Ecological Informatics Toolkit.

    The package provides reusable tools for community ecology,
    biodiversity analysis, statistical testing, ordination,
    clustering, indicator species analysis, and publication-ready
    ecological outputs.

Version:
    1.0
===============================================================================
"""

# =============================================================================
# CORE
# =============================================================================

from .core import (
    AnalysisResult,
    validate_community_data,
)


# =============================================================================
# DISTANCES
# =============================================================================

from .distances import (
    compute_distance_matrix,
    validate_distance_matrix,
)


# =============================================================================
# DIVERSITY
# =============================================================================

from .diversity import (
    species_richness,
    shannon_index,
    simpson_index,
    pielou_evenness,
    calculate_alpha_diversity,
    summarize_alpha_diversity,
    export_alpha_diversity,
)


# =============================================================================
# STATISTICS
# =============================================================================

from .statistics import (
    StatisticalResult,
    check_assumptions,
    calculate_effect_size,
    run_posthoc_tests,
    compare_groups,
    summarize_statistics,
    export_statistics,
)


# =============================================================================
# ORDINATION AND MULTIVARIATE ANALYSIS
# =============================================================================

from .ordination import (
    run_nmds,
    plot_nmds,
    run_permanova,
    run_permdisp,
    run_pca,
    plot_pca,
    run_hierarchical_clustering,
    plot_dendrogram,
)


# =============================================================================
# INDICATOR SPECIES ANALYSIS
# =============================================================================

from .indicator_species import (
    run_indicator_species_analysis,
)


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [

    # Core
    "AnalysisResult",
    "validate_community_data",

    # Distances
    "compute_distance_matrix",
    "validate_distance_matrix",

    # Diversity
    "species_richness",
    "shannon_index",
    "simpson_index",
    "pielou_evenness",
    "calculate_alpha_diversity",
    "summarize_alpha_diversity",
    "export_alpha_diversity",

    # Statistics
    "StatisticalResult",
    "check_assumptions",
    "calculate_effect_size",
    "run_posthoc_tests",
    "compare_groups",
    "summarize_statistics",
    "export_statistics",

    # Ordination and multivariate analysis
    "run_nmds",
    "plot_nmds",
    "run_permanova",
    "run_permdisp",
    "run_pca",
    "plot_pca",
    "run_hierarchical_clustering",
    "plot_dendrogram",

    # Indicator species
    "run_indicator_species_analysis",
]


__version__ = "1.0.0"