"""
==========================================================
Statistical Analysis Utilities
==========================================================

This module provides reusable statistical functions for
community ecology analyses.

Functions include:

- Assumption testing
- Parametric and non-parametric group comparisons
- Effect size estimation
- Post-hoc multiple comparisons
- Statistical summary tables

Author: Peter Ugege

Project: Omo Forest Reserve Ecological Informatics

Version : 1.0

Last Updated : July 2026

Python : >=3.11

License : MIT
==========================================================
"""

# ======================================================
# Standard Library
# ======================================================

from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Any
from pathlib import Path
from datetime import datetime


# ======================================================
# Third-Party Libraries
# ======================================================

import numpy as np
import pandas as pd

from scipy.stats import (
    shapiro,
    levene,
    f_oneway,
    kruskal
)

from statsmodels.stats.multicomp import pairwise_tukeyhsd

import scikit_posthocs as sp

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning
)

# ==========================================================
# Statistical Result
# ==========================================================

@dataclass
class StatisticalResult:
    """
    Container for statistical comparison results.
    """

    response: str
    grouping: str

    test: str

    statistic: float

    p_value: float

    significant: bool
    
    alpha: float = 0.05

    effect_size: float | None = None

    effect_measure: str | None = None

    assumptions: dict[str, Any] | None = None

    posthoc: pd.DataFrame | None = None
    
    sample_size: int | None = None
    
    n_groups: int | None = None
    
    
# ==========================================================
# Assumption Testing
# ==========================================================
def check_assumptions(
    data: pd.DataFrame,
    response: str,
    group: str,
    alpha: float = 0.05
) -> dict[str, Any]:
    """
    Evaluate assumptions required for parametric statistical tests.

    This function performs:
    - Shapiro-Wilk normality test for each group
    - Levene's test for homogeneity of variance
    - Descriptive statistics for each group
    - Sample size validation for Shapiro-Wilk

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe.

    response : str
        Response (dependent) variable.

    group : str
        Grouping (independent) variable.

    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:

        - normal
        - normality_assessable
        - equal_variance
        - group_sizes
        - sample_size_check
        - group_statistics
        - shapiro
        - levene
    """

    # ==========================================================
    # Input Validation
    # ==========================================================

    if response not in data.columns:
        raise ValueError(
            f"Response variable '{response}' not found in the DataFrame."
        )

    if group not in data.columns:
        raise ValueError(
            f"Grouping variable '{group}' not found in the DataFrame."
        )

    if data.empty:
        raise ValueError(
            "Input DataFrame is empty."
        )

    if data[group].nunique() < 2:
        raise ValueError(
            f"Grouping variable '{group}' must contain at least two groups."
        )

    # ==========================================================
    # Remove Missing Values
    # ==========================================================

    analysis_data = data[[response, group]].dropna().copy()

    if analysis_data.empty:
        raise ValueError(
            "No valid observations remain after removing missing values."
        )

    # ==========================================================
    # Group Information
    # ==========================================================

    group_sizes = (
        analysis_data
        .groupby(group)
        .size()
        .to_dict()
    )

    grouped = [
        values.values
        for _, values in analysis_data.groupby(group)[response]
    ]

    # ==========================================================
    # Assumption Checks
    # ==========================================================

    shapiro_results = {}

    group_statistics = {}

    sample_size_check = {}

    normal = True

    normality_assessable = True

    # ------------------------------------------------------
    # Shapiro-Wilk Test
    # ------------------------------------------------------

    for name, values in analysis_data.groupby(group):

        clean = values[response]

        n = len(clean)

        # ----------------------------------------------
        # Sample Size Check
        # ----------------------------------------------

        sample_size_check[name] = {

            "n": n,

            "shapiro_valid": n >= 3

        }

        # ----------------------------------------------
        # Descriptive Statistics
        # ----------------------------------------------

        group_statistics[name] = {

            "n": n,

            "mean": clean.mean(),

            "std": clean.std(),

            "median": clean.median(),

            "min": clean.min(),

            "max": clean.max()

        }

        # ----------------------------------------------
        # Shapiro-Wilk Normality Test
        # ----------------------------------------------

        if n >= 3:

            stat, p = shapiro(clean)

        else:

            stat = np.nan
            p = np.nan

            normality_assessable = False

        shapiro_results[name] = {

            "W": stat,

            "p": p

        }

        if not np.isnan(p) and p < alpha:

            normal = False

    # ------------------------------------------------------
    # Levene's Test
    # ------------------------------------------------------

    lev_stat, lev_p = levene(*grouped)

    equal_variance = lev_p >= alpha

    # ------------------------------------------------------
    # Return Results
    # ------------------------------------------------------

    return {

        "normal": normal,

        "normality_assessable": normality_assessable,

        "equal_variance": equal_variance,

        "group_sizes": group_sizes,

        "sample_size_check": sample_size_check,

        "group_statistics": group_statistics,

        "shapiro": shapiro_results,

        "levene": {

            "Statistic": lev_stat,

            "p": lev_p

        }

    }
    
# ==========================================================
# Effect Size
# ==========================================================
def calculate_effect_size(
    data: pd.DataFrame,
    response: str,
    group: str,
    test: str
) -> dict[str, float | str | None]:
    """
    Calculate effect size for One-way ANOVA or Kruskal-Wallis test.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe.

    response : str
        Response (dependent) variable.

    group : str
        Grouping (independent) variable.

    test : str
        Statistical test used.
        Supported values:
        - "One-way ANOVA"
        - "Kruskal-Wallis"

    Returns
    -------
    dict
        Dictionary containing:

        - effect_size
        - measure
    """

    # ==========================================================
    # Input Validation
    # ==========================================================

    if response not in data.columns:
        raise ValueError(
            f"Response variable '{response}' not found in the DataFrame."
        )

    if group not in data.columns:
        raise ValueError(
            f"Grouping variable '{group}' not found in the DataFrame."
        )

    if data.empty:
        raise ValueError(
            "Input DataFrame is empty."
        )

    # ==========================================================
    # Remove Missing Values
    # ==========================================================

    analysis_data = data[[response, group]].dropna().copy()

    if analysis_data.empty:
        raise ValueError(
            "No valid observations remain after removing missing values."
        )

    # ==========================================================
    # Prepare Groups
    # ==========================================================

    groups = [
        values.values
        for _, values in analysis_data.groupby(group)[response]
    ]

    n = len(analysis_data)
    k = len(groups)

    # ==========================================================
    # One-way ANOVA
    # ==========================================================

    if test == "One-way ANOVA":

        overall_mean = analysis_data[response].mean()

        ss_between = sum(
            len(g) * (np.mean(g) - overall_mean) ** 2
            for g in groups
        )

        ss_total = np.sum(
            (analysis_data[response] - overall_mean) ** 2
        )

        eta2 = ss_between / ss_total if ss_total > 0 else np.nan

        return {

            "effect_size": eta2,

            "measure": "Eta Squared (η²)"

        }

    # ==========================================================
    # Kruskal-Wallis
    # ==========================================================

    elif test == "Kruskal-Wallis":

        h, _ = kruskal(*groups)

        if n > k:

            epsilon2 = (h - k + 1) / (n - k)

        else:

            epsilon2 = np.nan

        return {

            "effect_size": epsilon2,

            "measure": "Epsilon Squared (ε²)"

        }

    # ==========================================================
    # Unsupported Test
    # ==========================================================

    raise ValueError(

        f"Unsupported statistical test '{test}'. "

        "Supported tests are 'One-way ANOVA' and "

        "'Kruskal-Wallis'."

    )

# ==========================================================
# Post-hoc Multiple Comparisons
# ==========================================================

def run_posthoc_tests(
    data: pd.DataFrame,
    response: str,
    group: str,
    test: str,
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Run post-hoc multiple comparisons.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe.

    response : str
        Response (dependent) variable.

    group : str
        Grouping (independent) variable.

    test : str
        Statistical test used.

    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    pd.DataFrame
        Pairwise comparison results.
    """

    # ==========================================================
    # Input Validation
    # ==========================================================

    if response not in data.columns:
        raise ValueError(
            f"Response variable '{response}' not found."
        )

    if group not in data.columns:
        raise ValueError(
            f"Grouping variable '{group}' not found."
        )

    if data.empty:
        raise ValueError(
            "Input DataFrame is empty."
        )

    # ==========================================================
    # Remove Missing Values
    # ==========================================================

    analysis_data = data[[response, group]].dropna().copy()

    if analysis_data.empty:
        raise ValueError(
            "No valid observations remain after removing missing values."
        )

    # ==========================================================
    # Tukey HSD
    # ==========================================================

    if test == "One-way ANOVA":

        tukey = pairwise_tukeyhsd(
            endog=analysis_data[response],
            groups=analysis_data[group],
            alpha=alpha
        )

        results = pd.DataFrame(
            tukey.summary().data[1:],
            columns=tukey.summary().data[0]
        )

        results = results.rename(
            columns={
                "group1": "Group 1",
                "group2": "Group 2",
                "meandiff": "Mean Difference",
                "p-adj": "Adjusted p-value",
                "lower": "Lower CI",
                "upper": "Upper CI",
                "reject": "Significant"
            }
        )

        return results

    # ==========================================================
    # Dunn's Test
    # ==========================================================

    if test == "Kruskal-Wallis":

        return sp.posthoc_dunn(
            analysis_data,
            val_col=response,
            group_col=group,
            p_adjust="holm"
        )

    # ==========================================================
    # Unsupported Test
    # ==========================================================

    raise ValueError(

        f"Unsupported statistical test '{test}' for post-hoc analysis."

    )

# ==========================================================
# Group Comparison
# ==========================================================

def compare_groups(
    data: pd.DataFrame,
    response: str,
    group: str,
    alpha: float = 0.05
) -> StatisticalResult:
    """
    Compare a response variable among groups.

    Automatically selects the appropriate statistical test based on
    assessment of normality and homogeneity of variance.

    Current decision logic:

        • One-way ANOVA
            - Normal data
            - Equal variances

        • Kruskal-Wallis
            - Non-normal data
            - Unequal variances
            - Normality not assessable

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe.

    response : str
        Response (dependent) variable.

    group : str
        Grouping (independent) variable.

    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    StatisticalResult
        Structured statistical comparison result.
    """

    # ==========================================================
    # Assumption Testing
    # ==========================================================

    assumptions = check_assumptions(
        data=data,
        response=response,
        group=group,
        alpha=alpha
    )

    # ==========================================================
    # Remove Missing Values
    # ==========================================================

    analysis_data = data[[response, group]].dropna().copy()

    if analysis_data.empty:
        raise ValueError(
            "No valid observations remain after removing missing values."
        )

    grouped = [
        values.values
        for _, values in analysis_data.groupby(group)[response]
    ]

    # ==========================================================
    # Warn if Normality Could Not Be Assessed
    # ==========================================================

    if not assumptions["normality_assessable"]:

        warnings.warn(

            "Normality could not be assessed for one or more groups "
            "because sample sizes were fewer than three observations. "
            "A non-parametric statistical test will be used.",

            UserWarning

        )

    # ==========================================================
    # Select Statistical Test
    # ==========================================================

    if (
        assumptions["normality_assessable"]
        and assumptions["normal"]
        and assumptions["equal_variance"]
    ):

        statistic, pvalue = f_oneway(*grouped)

        test = "One-way ANOVA"

    else:

        statistic, pvalue = kruskal(*grouped)

        test = "Kruskal-Wallis"

    # ==========================================================
    # Effect Size
    # ==========================================================

    effect = calculate_effect_size(
        data=analysis_data,
        response=response,
        group=group,
        test=test
    )

    # ==========================================================
    # Post-hoc Analysis
    # ==========================================================

    posthoc = None

    if pvalue < alpha:

        posthoc = run_posthoc_tests(
            data=analysis_data,
            response=response,
            group=group,
            test=test,
            alpha=alpha
        )

    # ==========================================================
    # Return Results
    # ==========================================================

    return StatisticalResult(

        response=response,

        grouping=group,

        test=test,

        statistic=statistic,

        p_value=pvalue,

        significant=pvalue < alpha,

        effect_size=effect["effect_size"],

        effect_measure=effect["measure"],

        assumptions=assumptions,

        posthoc=posthoc

    )
    
# ==========================================================
# Statistical Summary Table
# ==========================================================

def summarize_statistics(
    data: pd.DataFrame,
    metrics: list[str],
    grouping: str,
    alpha: float = 0.05
) -> tuple[pd.DataFrame, dict]:
    """
    Run statistical comparisons for multiple metrics and
    return a publication-ready summary table.

    Parameters
    ----------
    data : DataFrame
        Dataset containing diversity metrics.

    metrics : list[str]
        Diversity metrics to analyse.

    grouping : str
        Grouping variable (e.g., Zone or Habitat).

    alpha : float
        Significance level.

    Returns
    -------
    summary_table : DataFrame

    results : dict
        Dictionary of StatisticalResult objects.
    """

    summary_rows = []

    results = {}

    for metric in metrics:

        result = compare_groups(
            data=data,
            response=metric,
            group=grouping,
            alpha=alpha
        )

        results[metric] = result

        summary_rows.append({

            "Metric": metric,

            "Grouping": grouping,

            "Test": result.test,

            "Statistic": round(result.statistic, 4),

            "P-value": round(result.p_value, 4),

            "Effect Size": (
                round(result.effect_size, 4)
                if result.effect_size is not None
                else np.nan
            ),

            "Effect Measure": result.effect_measure,

            "Significant": (
                "Yes"
                if result.significant
                else "No"
            )

        })

    summary_table = pd.DataFrame(summary_rows)

    return summary_table, results


# ==========================================================
# Export Statistical Results
# ==========================================================

def export_statistics(
    summary_table: pd.DataFrame,
    results: dict[str, StatisticalResult],
    output_path: str | Path,
    alpha: float = 0.05,
    include_posthoc: bool = True,
    include_assumptions: bool = True
) -> None:
    """
    Export statistical analysis results to an Excel workbook.

    Parameters
    ----------
    summary_table : pd.DataFrame
        Publication-ready summary table.

    results : dict[str, StatisticalResult]
        Dictionary returned by summarize_statistics().

    output_path : str or Path
        Output Excel filename.

    alpha : float, default=0.05
        Significance level used during statistical analysis.

    include_posthoc : bool, default=True
        Export post-hoc comparison tables.

    include_assumptions : bool, default=True
        Export assumption test diagnostics.

    Returns
    -------
    None
    """

    # ==========================================================
    # Input Validation
    # ==========================================================

    if summary_table.empty:
        raise ValueError(
            "Summary table is empty."
        )

    if not results:
        raise ValueError(
            "Results dictionary is empty."
        )

    # ==========================================================
    # Prepare Output Path
    # ==========================================================

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # ==========================================================
    # Create Workbook
    # ==========================================================

    with pd.ExcelWriter(
        output_path,
        engine="openpyxl"
    ) as writer:

        # ======================================================
        # Sheet 1 : Workbook Information
        # ======================================================

        metadata = pd.DataFrame({

            "Item": [

                "Created",

                "Module",

                "Number of Metrics",

                "Significance Level",

                "Generated By"

            ],

            "Value": [

                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

                "statistics.py",

                len(results),

                alpha,

                "Omo Forest Reserve Ecological Informatics Toolkit"

            ]

        })

        metadata.to_excel(

            writer,

            sheet_name="Workbook_Info",

            index=False

        )

        # ======================================================
        # Sheet 2 : Summary
        # ======================================================

        summary_table.to_excel(

            writer,

            sheet_name="Summary",

            index=False

        )

        # ======================================================
        # Sheet 3 : Assumption Tests
        # ======================================================

        if include_assumptions:

            assumption_rows = []

            for metric, result in results.items():

                assumptions = result.assumptions

                for grp, values in assumptions["shapiro"].items():

                    sample_info = assumptions.get(
                        "sample_size_check",
                        {}
                    ).get(grp, {})

                    assumption_rows.append({

                        "Metric":
                            metric,

                        "Grouping":
                            result.grouping,

                        "Group":
                            grp,

                        "Sample Size":
                            sample_info.get("n"),

                        "Shapiro Valid":
                            sample_info.get("shapiro_valid"),

                        "Normality Assessable":
                            assumptions.get(
                                "normality_assessable"
                            ),

                        "Shapiro W":
                            values["W"],

                        "Shapiro p":
                            values["p"],

                        "Normal":
                            assumptions["normal"],

                        "Levene Statistic":
                            assumptions["levene"]["Statistic"],

                        "Levene p":
                            assumptions["levene"]["p"],

                        "Equal Variance":
                            assumptions["equal_variance"]

                    })

            assumption_df = pd.DataFrame(
                assumption_rows
            )

            assumption_df.to_excel(

                writer,

                sheet_name="Assumptions",

                index=False

            )

        # ======================================================
        # Post-hoc Sheets
        # ======================================================

        if include_posthoc:

            for metric, result in results.items():

                if result.posthoc is None:
                    continue

                sheet_name = (
                    f"{metric}_Posthoc"
                    .replace(" ", "_")
                )[:31]

                result.posthoc.to_excel(

                    writer,

                    sheet_name=sheet_name,

                    index=True

                )

        # ======================================================
        # Improve Workbook Formatting
        # ======================================================

        for worksheet in writer.sheets.values():

            # Freeze header row
            worksheet.freeze_panes = "A2"

            # Auto-fit column widths
            for column_cells in worksheet.columns:

                max_length = max(

                    len(str(cell.value))
                    if cell.value is not None else 0

                    for cell in column_cells

                )

                column_letter = column_cells[0].column_letter

                worksheet.column_dimensions[
                    column_letter
                ].width = min(max_length + 2, 40)