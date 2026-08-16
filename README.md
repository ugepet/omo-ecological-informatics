# Omo Forest Ecological Informatics

An open-source ecological informatics project for the analysis of tropical forest
communities in the Omo Biosphere Reserve, Nigeria.

The project integrates ecological data engineering, biodiversity analysis,
multivariate community ecology, statistical inference, functional ecology,
landscape ecology, machine learning, and reproducible computational workflows.

---

## Project Overview

The Omo Forest Ecological Informatics Project transforms vegetation inventory
data into structured, analysis-ready ecological datasets and reproducible
analytical workflows.

The project is designed around two complementary components:

1. **Jupyter notebooks** for transparent scientific workflows, exploration,
   interpretation, visualization, and reporting.
2. **Reusable Python modules (`src`)** containing validated analytical functions
   that can be reused across notebooks and future ecological studies.

This architecture separates analytical implementation from scientific
interpretation and improves reproducibility, maintainability, and extensibility.

---

## Study Area

The study focuses on woody plant communities within the **Omo Biosphere Reserve,
Ogun State, Nigeria**.

Sampling covers ecological and management environments including:

- Core Zone
- Buffer Zone
- Transition Zone
- Major River habitats
- Stream habitats
- Upland forest habitats

---

## Current Dataset

The finalized and taxonomically harmonized community dataset currently contains:

- **90 sampling plots**
- **127 canonical woody plant species**
- **1,750 individual woody plants**
- Harmonized species taxonomy
- Plot-level ecological metadata
- Species-level metadata
- Plot × species community abundance matrix

Canonical processed datasets include:

```text
data/processed/
├── omo_community_matrix_harmonized.csv
├── omo_plot_metadata.csv
├── omo_species_metadata.csv
├── community_species_inventory.csv
├── community_harmonization_summary.csv
├── community_taxonomic_corrections.csv
├── community_taxonomic_review.csv
├── exact_duplicate_species.csv
└── genus_similarity_candidates.csv
```

---

## Repository Structure

```text
omo-ecological-informatics/

├── data/
│   ├── raw/
│   ├── processed/
│   ├── ml_ready/
│   └── metadata/
│
├── notebooks/
├── src/
├── tests/
├── results/
│   ├── figures/
│   └── tables/
│
├── docs/
├── manuscript/
├── figures/
├── outputs/
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Reusable Analytical Toolkit

The `src` package provides reusable functions for community ecological analysis.

Current modules include:

```text
src/
├── __init__.py
├── core.py
├── distances.py
├── diversity.py
├── statistics.py
├── ordination.py
├── indicator_species.py
└── publication_graphics.py
```

Major capabilities include:

- ecological data validation
- Bray-Curtis distance matrices
- species richness
- Shannon diversity
- Simpson diversity
- Pielou evenness
- statistical assumption testing
- group comparisons
- effect-size estimation
- post-hoc analysis
- NMDS
- PERMANOVA
- PERMDISP
- PCA
- hierarchical clustering
- indicator species analysis
- publication-quality ecological graphics
- structured analytical result objects

---

## Notebook Workflow

The project follows a staged ecological informatics workflow.

### Completed

- ✅ Notebook 01 — Data Compilation
- ✅ Notebook 02 — Taxonomic Standardization
- ✅ Notebook 03 — Taxonomic Harmonization
- ✅ Notebook 04 — Ecological Characterization
- ✅ Notebook 05 — Community Matrix Construction and Ecological Data Restructuring
- ✅ Notebook 06 — Advanced Community Ecology and Statistical Analysis

### Current Phase — Notebook 07: Functional Ecology

Notebook 07 has been modularized into a series of linked notebooks to maintain
reproducibility, clarity, and manageable analytical workflows:

- 🔄 Notebook 07A — Structural Data Reconstruction and Validation
- ⏳ Notebook 07B — Functional Trait Acquisition and Harmonization
- ⏳ Notebook 07C — Trait QC and Analytical Matrix Construction
- ⏳ Notebook 07D — Functional Composition and Community-Weighted Traits
- ⏳ Notebook 07E — Multidimensional Functional Diversity
- ⏳ Notebook 07F — Integrated Functional Ecology and Structure–Function Relationships
- ⏳ Notebook 07G — Functional Ecology Synthesis, Validation and Handover

### Planned

- ⏳ Notebook 08 — Landscape and Environmental Ecology
- ⏳ Notebook 09 — Ecological Informatics Integration

---

## Notebook 06 — Community Ecology

Notebook 06 provides the principal community ecological and multivariate
statistical workflow.

Analyses currently include:

- Alpha-diversity estimation
- Statistical comparison of diversity metrics
- Bray-Curtis community dissimilarity
- Non-metric Multidimensional Scaling (NMDS)
- PERMANOVA
- PERMDISP
- Principal Component Analysis (PCA)
- Hierarchical clustering
- Indicator Species Analysis
- Publication-ready figures and tables
- Integrated ecological results summary

The finalized workflow was verified through a clean Jupyter kernel restart and
successful top-to-bottom execution.

---

## Current Community-Ecology Results

The finalized Notebook 06 analysis uses:

- **90 plots**
- **127 canonical woody plant species**
- **1,750 individuals**

Selected analytical diagnostics include:

- NMDS stress: **0.237**
- Habitat PERMANOVA: **Pseudo-F = 5.5963, p = 0.001**
- Zone PERMANOVA: **Pseudo-F = 10.6543, p = 0.001**
- Habitat PERMDISP: **F = 3.6458, p = 0.020**
- Zone PERMDISP: **F = 5.9881, p = 0.004**
- PCA PC1 + PC2 variance: **23.24%**
- Hierarchical clustering cophenetic correlation: **0.9085**
- Significant habitat indicator species: **44**
- Significant zone indicator species: **42**
- Significant community-cluster indicator species: **37**

Because PERMDISP is significant for both habitat and management zone,
PERMANOVA results should be interpreted with appropriate consideration of
differences in multivariate dispersion.

---

## Notebook 07 — Functional Ecology

Notebook 07 extends the taxonomic and community-ecological foundation established
in Notebooks 01–06 into trait-based and structure–function ecology.

The functional ecology phase will integrate:

- taxonomically harmonized woody plant communities
- historical forest structural information
- species functional traits
- community-weighted mean traits
- multidimensional functional diversity
- habitat and management-zone comparisons
- functional characterization of data-derived plant communities
- taxonomic–functional relationships
- forest structure–function relationships

The modular 07A–07G design allows each analytical stage to produce validated
outputs that become inputs to the subsequent stage without unnecessarily
repeating upstream analyses.

---

## Installation

Clone the repository and install the Python dependencies:

```bash
git clone https://github.com/ugepet/omo-ecological-informatics.git
cd omo-ecological-informatics
pip install -r requirements.txt
```

Core dependencies include:

- NumPy
- pandas
- SciPy
- scikit-learn
- scikit-bio
- statsmodels
- scikit-posthocs
- Matplotlib
- seaborn
- openpyxl

---

## Reproducibility

The project emphasizes reproducible ecological analysis.

Analytical functions are implemented in reusable Python modules while notebooks
serve primarily as orchestration, visualization, interpretation, and reporting
layers.

Validated upstream products are preserved and consumed by downstream notebooks
rather than being unnecessarily reconstructed or recalculated.

The intended workflow is:

```text
Ecological field data
        ↓
Data compilation
        ↓
Taxonomic standardization and harmonization
        ↓
Ecological characterization
        ↓
Community matrix construction
        ↓
Community ecology and statistical analysis
        ↓
Functional ecology
   (Notebooks 07A–07G)
        ↓
Landscape and environmental ecology
        ↓
Ecological informatics integration
        ↓
Scientific interpretation and publication
```

---

## Future Development

Planned extensions include:

- Functional trait integration
- Community-weighted trait analysis
- Multidimensional functional diversity
- Forest structure–function analysis
- Landscape ecological analysis
- Environmental predictor integration where spatial linkage is defensible
- Machine-learning-based ecological classification
- Explainable AI
- Ecological prediction
- Conservation-oriented decision support

---

## License

This project is released under the **MIT License**.

---

## Author

**Peter E. Ugege**

Forestry Research Institute of Nigeria (FRIN)  
Ibadan, Nigeria

Research interests include ecological informatics, biodiversity informatics,
GeoAI, machine learning, deep learning, and computational environmental science.