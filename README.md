# Omo Forest Ecological Informatics

An open-source ecological informatics project for the analysis of tropical forest
communities in the Omo Biosphere Reserve, Nigeria.

The project integrates ecological data engineering, biodiversity analysis,
multivariate community ecology, statistical inference, spatial ecology,
functional ecology, machine learning, and reproducible computational workflows.

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
- Major river habitats
- Stream habitats
- Upland forest habitats

---

## Current Dataset

The finalized community dataset currently contains:

- **90 sampling plots**
- **133 woody plant species**
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

### Next Phase

- 🔄 Notebook 07 — Functional Ecology
- ⏳ Notebook 08 — Spatial Ecology
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
- **133 woody plant species**

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
        ↓
Spatial ecology
        ↓
Ecological informatics integration
        ↓
Scientific interpretation and publication
```

---

## Future Development

Planned extensions include:

- Functional diversity analysis
- Species trait integration
- Spatial community ecology
- Environmental predictor integration
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