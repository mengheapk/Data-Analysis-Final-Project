# Predicting Coffee Quality Tier without Cupping

A multiclass logistic regression pipeline that classifies Arabica coffee samples into Commercial, Premium, or Specialty quality tiers using production-side features alone — no sensory cupping data required. The project quantifies how much predictive value cupping provides over production data, and demonstrates that a single sensory feature (Flavor) nearly doubles the predictive performance of 21 production features combined.

Final project for **Data Analysis 1 (INF-604)** at the American University of Phnom Penh (AUPP), Spring 2026.

**Team:** Khan Mengheap, Taing Muyleang
**Lecturer:** Dr. HAS Sothea

---

## Motivation

Coffee quality grading is traditionally done through cupping — sensory evaluation by certified Q-graders. Cupping is expensive, slow, and not always available, especially for smallholder farmers and early-stage buyers. This project investigates whether quality tier can be predicted from production-side information alone (origin, altitude, processing method, variety, defects), and how much predictive power is sacrificed by skipping the sensory evaluation step.

## Dataset

[Coffee Quality Institute (CQI) Arabica Quality Database](https://github.com/jldbc/coffee-quality-database) — 1,311 professionally evaluated Arabica coffee samples from 34 countries (2009–2018), scraped by LeDoux (2018) from the CQI's public database.

After preprocessing: **1,251 samples × 21 features**.

The target variable `quality_tier` was derived from `total_cup_points` using SCA grading thresholds:
- **Commercial** (< 80 points): 13.9%
- **Premium** (80 – 84 points): 78.0%
- **Specialty** (≥ 85 points): 8.1%

The severe class imbalance shaped several modeling decisions.

## Key Preprocessing Decisions

The project's main engineering contribution is in preprocessing. Highlights:

**Custom altitude parser.** The dataset's pre-processed altitude column contained values up to 190,164 m (Mount Everest is 8,849 m). Investigation revealed two distinct bugs: decimal-stripping errors in the dataset's own parser, and producer-side reporting errors. A custom regex-based parser was built to handle mixed unit formats, range notation, and unit conversion, with country-median imputation for missing values.

**Harvest year parser.** The `harvest_year` column had 46 unique values across 5 formats: plain years, ranges, two-digit codes, quarter notation, and multilingual date text. A two-stage regex parser recovered 1,251 of 1,308 rows; 57 unparseable entries were dropped.

**Hidden missingness in moisture.** The `moisture` column reported zero null values but contained 229 rows with `moisture = 0` — physically impossible. These were treated as hidden missingness and imputed with the global median.

**Categorical grouping.** Reduced cardinality using domain taxonomies: `variety` from 29 → 10 groups (World Coffee Research), `country_of_origin` from 34 → 5 regions (ICO classifications).

## Methodology

- **Algorithm:** Multiclass logistic regression (multinomial / softmax)
- **Split:** 80/20 stratified, `random_state=42`
- **Primary metric:** Macro-F1 (chosen for fairness across imbalanced classes)
- **Class imbalance handling:** `class_weight='balanced'` in Model A

Four model variants were compared to isolate specific design questions:

| Model | Features | Class weighting | Purpose |
|-------|----------|-----------------|---------|
| Dummy | None (predicts Premium) | — | Baseline floor |
| Model A — Unbalanced | 21 production features | None | Naive production model |
| Model A — Balanced | 21 production features | `balanced` | Imbalance-corrected production model |
| Model B — Flavor only | 1 sensory feature | None | Sensory benchmark |

## Results

| Model | Accuracy | Macro F1 | Precision | Recall |
|-------|----------|----------|-----------|--------|
| Dummy | 0.781 | 0.292 | 0.260 | 0.333 |
| Model A — Unbalanced | 0.785 | 0.409 | 0.615 | 0.397 |
| Model A — Balanced | 0.526 | **0.463** | 0.475 | 0.644 |
| Model B — Flavor only | **0.912** | **0.832** | 0.909 | 0.788 |

**Headline finding:** A single sensory feature (Flavor) nearly doubles the predictive performance of 21 production features combined (Macro-F1: 0.83 vs 0.46). This quantifies the predictive value of expert sensory evaluation and supports the conclusion that cupping captures information not available from production data alone.

## Repository Structure

├── final_project.ipynb       # Main Jupyter notebook with all analysis
├── final_project_report.pdf  # Full written report (Sections I–VIII)
├── data/                     # Raw and processed datasets
├── figures/                  # All charts and visualizations
└── README.md                 # This file

## Tools

Python 3, pandas, NumPy, scikit-learn, matplotlib, seaborn, SciPy, Jupyter, LaTeX (for the report).

## Future Work

- Hybrid model combining production features with 1–2 sensory scores to map the cost-quality tradeoff
- Tree-based models (Random Forest, XGBoost) to test whether non-linear patterns hinted at in Spearman correlation translate to better performance
- K-fold cross-validation for more robust performance estimates
- Application of similar methodology to other agricultural commodities with quality grading systems

## References

Key references are listed in the full report. Notable sources:
- LeDoux, J. (2018). *coffee-quality-database*. [GitHub](https://github.com/jldbc/coffee-quality-database).
- Specialty Coffee Association (2005). *SCAA Cupping Protocols*.
- Wintgens, J. N. (2009). *Coffee: Growing, Processing, Sustainable Production*.

## License

MIT — feel free to use this code for learning or further research.
