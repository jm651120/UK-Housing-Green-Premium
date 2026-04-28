# The Green Premium in the UK Housing Market
### Evidence from Open Government Data and Machine Learning

> **Master's Thesis — NOVA IMS, April 2026**  
> Author: João Marques · [joaomarques2003@sapo.pt](mailto:joaoluismarques2003@gmail.com)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Champion-XGBoost-orange?logo=xgboost)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CRISP-ML](https://img.shields.io/badge/Methodology-CRISP--ML-blueviolet)](https://ml-ops.org/content/crisp-ml)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Key Findings](#key-findings)
3. [Repository Structure](#repository-structure)
4. [Tech Stack](#tech-stack)
5. [Methodology — CRISP-ML Pipeline](#methodology--crisp-ml-pipeline)
6. [Models & Results](#models--results)
7. [Explainability — TreeSHAP Analysis](#explainability--treeshap-analysis)
8. [Data Sources](#data-sources)
9. [How to Run](#how-to-run)
10. [Citation](#citation)

---

## Project Overview

This repository contains the full machine-learning pipeline underpinning the Master's thesis *"The Green Premium in the UK Housing Market: Evidence from Open Government Data and Machine Learning."*

The study quantifies the **Green Premium** — the price uplift commanded by high-energy-efficiency homes (EPC Band A/B) — and the **Brown Discount** — the price penalty suffered by low-efficiency stock (EPC Band F/G) — across three study regions: **Greater London**, **Greater Manchester**, and **Cornwall**, covering **514,345 post-filter residential transactions** from 2021 to 2025.

The research is motivated by the UK Government's **Net Zero 2050** commitment and the £300 billion+ retrofitting challenge facing the national housing stock. By translating EPC ratings into precise pound-sterling valuations via machine learning and SHAP explainability, this project provides evidence-based guidance for buyers, valuers, lenders, and policymakers navigating the energy-efficiency transition.

---

## Key Findings

| Finding | Value |
|---|---|
| **Green Premium** (EPC Band A vs. Band D baseline) | **+£13,506** |
| **Brown Discount** (EPC Band G vs. Band D baseline) | **−£12,017** |
| **Historical Forgiveness Effect** — Band G rural heritage (Cornwall) | **+£41,317 SHAP contribution** |
| **Location Price Squeeze** — Band A in Greater London | Compressed premium despite highest Brown Discount nationally |
| **Champion Model R²** | **0.8249** |
| **Champion Model MAE** | **£60,868** |

Three novel spatial phenomena are identified and characterised:
- **Green Premium** — statistically significant EPC-linked value uplift
- **Brown Discount** — penalisation of thermally inefficient stock, strongest in Greater London
- **Historical Forgiveness Effect** — pre-1900 heritage properties in rural contexts where character overrides energy inefficiency in buyer valuations

---

## Repository Structure

```
UK-Housing-Green-Premium/
│
├── data/
│   ├── raw/                        # Source downloads (not tracked by Git)
│   │   ├── land_registry_ppd/      # HM Land Registry Price Paid Data (.csv)
│   │   ├── epc_domestic/           # MHCLG Domestic EPC certificates (.csv)
│   │   └── ons_postcode/           # ONS Postcode Directory (.csv)
│   └── processed/
│       ├── merged_clean.parquet    # Post-merge, post-filter dataset (514,345 rows)
│       └── approved_features_53.json  # Canonical 53-feature list (cross-model parity)
│
├── notebooks/
│   ├── 01_data_ingestion.ipynb     # Phase 1: data loading, merging, deduplication
│   ├── 02_eda.ipynb                # Exploratory data analysis & spatial maps
│   ├── 03_feature_engineering.ipynb # OHE, IQR outlier removal, feature construction
│   ├── 04_feature_screening.ipynb  # LightGBM split-frequency screening → 53 features
│   ├── 05_ols_baseline.ipynb       # OLS with postcode-outcode fixed effects
│   ├── 06_random_forest.ipynb      # RF with Optuna Bayesian optimisation (30 trials)
│   ├── 07_lightgbm.ipynb           # LightGBM with Optuna Bayesian optimisation
│   ├── 08_xgboost.ipynb            # XGBoost (champion) with Optuna Bayesian optimisation
│   └── 09_shap_analysis.ipynb      # TreeSHAP global + regional explainability
│
├── outputs/
│   ├── figures/                    # All thesis figures (matplotlib / seaborn / plotly)
│   ├── models/                     # Serialised model artefacts (.pkl / .json)
│   └── results/
│       └── model_comparison.csv    # R², MAE, RMSE, MAPE across all four models
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Tech Stack

| Category | Library / Tool |
|---|---|
| **Data Manipulation** | `pandas`, `numpy` |
| **Machine Learning** | `scikit-learn`, `xgboost`, `lightgbm` |
| **Statistical Modelling** | `statsmodels` |
| **Hyperparameter Optimisation** | `optuna` (TPE sampler, 30 trials per algorithm) |
| **Explainability** | `shap` (TreeExplainer / TreeSHAP) |
| **Visualisation** | `matplotlib`, `seaborn`, `plotly`, `folium` |
| **Serialisation** | `json` |
| **Acceleration** | CUDA / GPU (LightGBM feature-screening pass) |
| **Environment** | Python 3.10+, Jupyter Notebook |

---

## Methodology — CRISP-ML Pipeline

This project strictly follows the **CRISP-ML** (Cross-Industry Standard Process for Machine Learning) framework across three phases:

### Phase 1 — Data Understanding & Preparation

- **Sources merged**: HM Land Registry PPD × MHCLG Domestic EPC × ONS Postcode Directory
- **Linkage key**: full postcode string (exact match)
- **Temporal filter**: 2021–2025 transactions only
- **Study regions**: Greater London, Greater Manchester, Cornwall
- **Outlier removal**: IQR method — upper fences at Q3 + 1.5×IQR applied to `Price` and `Total_Floor_Area` *(Tukey, 1977; Rousseeuw & Hubert, 2011)*
- **Post-filter N**: **514,345 transactions**
- **EPC ordinal encoding**: A=7, B=6, C=5, D=4, E=3, F=2, G=1
- **Categorical encoding**: One-Hot Encoding with `drop_first=True`; 27 Construction Age Band dummies retained
- **Train/test split**: 80/20 stratified random split (`random_state=42`)

### Phase 2 — Feature Engineering & Screening

- Raw feature space constructed from merged dataset
- **LightGBM split-frequency screening**: 500 estimators, GPU acceleration, `random_state=42`
- Features with zero split contribution removed
- **Canonical 53-feature space** persisted to `approved_features_53.json`
- All downstream models (RF, XGBoost) load this artefact to enforce **cross-model feature parity**
- **OLS spatial treatment**: postcode-outcode dummy fixed effects (avoids linearity violation with continuous coordinates)
- **ML spatial treatment**: exact latitude/longitude passed as features for non-linear spatial boundary learning

### Phase 3 — Modelling, Optimisation & Evaluation

- **Four models trained**: OLS baseline, Random Forest, LightGBM, XGBoost
- **Bayesian hyperparameter optimisation**: Optuna framework, TPE sampler, `seed=42`, 30 trials per algorithm
- **Evaluation metrics**: R², MAE (£), RMSE (£), MAPE (%)
- **Explainability**: TreeSHAP via `shap.TreeExplainer`; 50,000-observation random subsample from test set; SHAP values denominated in £ sterling

---

## Models & Results

| Model | R² | MAE (£) | RMSE (£) | MAPE (%) |
|---|---|---|---|---|
| OLS Baseline | 0.7458 | £79,449 | £114,402 | 24.82% |
| Random Forest | 0.8187 | £62,202 | £96,610 | 18.16% |
| LightGBM | 0.8233 | £61,545 | £95,387 | 17.96% |
| **XGBoost (Champion)** | **0.8249** | **£60,868** | **£94,954** | **17.67%** |

> XGBoost was selected as the champion model on all four metrics following Bayesian optimisation. Its predictions serve as the basis for all SHAP-based decompositions reported in Chapter 5 of the thesis.

---

## Explainability — TreeSHAP Analysis

Global and regional SHAP analyses were conducted on a **50,000-observation random subsample** of the XGBoost test-set predictions. SHAP values are expressed in **£ sterling**, permitting direct interpretation as marginal price contributions.

**Key SHAP insights:**

- **`Current_Energy_Efficiency_Score`** is the dominant EPC-linked predictor globally; SHAP dependence plots reveal a non-linear, monotonically increasing relationship with house price.
- **`latitude` / `longitude`** rank among the top global drivers, confirming that spatial location is the principal source of price variance — and validating the model's ability to learn non-linear regional boundaries.
- The **Historical Forgiveness Effect** is identified through regional SHAP waterfall decompositions: Band G pre-1900 detached properties in Cornwall exhibit a **+£41,317 positive SHAP contribution** from EPC rating, driven by heritage buyer preferences overriding thermodynamic inefficiency penalties.
- The **Location Price Squeeze** in Greater London manifests as a compression of achievable Green Premia despite the city imposing the nation's most severe Brown Discount, explained by extreme baseline land rents that diminish the marginal valuation impact of efficiency improvements.

---

## Data Sources

| Dataset | Provider | Access |
|---|---|---|
| Price Paid Data (LR-PPD) | HM Land Registry | [Open Government Licence](https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads) |
| Domestic EPC Certificates | MHCLG / Open Data Communities | [Open Government Licence](https://epc.opendatacommunities.org/) |
| ONS Postcode Directory | Office for National Statistics | [Open Geography Portal](https://geoportal.statistics.gov.uk/) |

> All three datasets are released under the **Open Government Licence v3.0**. Raw data files are **not tracked in this repository** due to size constraints; download instructions are provided in `notebooks/01_data_ingestion.ipynb`.

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/jm651120/UK-Housing-Green-Premium.git
cd UK-Housing-Green-Premium
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download raw data

Follow the download instructions in `notebooks/01_data_ingestion.ipynb` to retrieve the three open-government datasets and place them in `data/raw/`.

### 4. Run the pipeline sequentially

```bash
jupyter notebook
```

Execute notebooks `01` through `09` in order. Each notebook reads from and writes to the `data/processed/` and `outputs/` directories.

> **GPU note**: Notebook `04_feature_screening.ipynb` uses GPU-accelerated LightGBM. If no CUDA device is available, change `device='gpu'` to `device='cpu'` in the LGBMRegressor constructor — runtime will increase significantly.

### 5. Reproduce champion-model SHAP analysis

Open `notebooks/09_shap_analysis.ipynb`. The notebook loads the serialised XGBoost model from `outputs/models/` and the 50,000-observation SHAP subsample, then reproduces all global and regional explainability figures.

---

## Citation

If you use this code or build upon this research, please cite:

```bibtex
@mastersthesis{marques2026green,
  author    = {João Marques},
  title     = {The Green Premium in the UK Housing Market: Evidence from Open Government Data and Machine Learning},
  school    = {NOVA Information Management School (NOVA IMS)},
  year      = {2026},
  month     = {April},
  address   = {Lisbon, Portugal}
}
```

---

<p align="center">
  <em>Built with open government data · NOVA IMS · 2026</em>
</p>
