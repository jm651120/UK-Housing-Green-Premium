# Decoding the UK Housing Market: ML, XAI, and the "Green Premium" 🏡🌱

This repository contains the full data science pipeline for my Master's Thesis in Data Science and Advanced Analytics at NOVA Information Management School.

## 📌 Project Overview
The primary objective of this research is to move beyond traditional Hedonic Pricing Models (OLS) by deploying advanced non-linear Machine Learning algorithms to decode the complex dynamics of the UK real estate market (2021-2025). The core focus is quantifying the financial value of Energy Performance Certificates (EPC) — the so-called **"Green Premium"** and **"Brown Discount"**.

By combining eXtreme Gradient Boosting (XGBoost) with Explainable AI (SHAP), this project transitions from "black-box" prediction to transparent microeconomic interpretation, proving that the valuation of sustainability is highly elastic and conditioned by spatial tension and historical architecture.

## 📊 Data Sources (Open Government Data)
The pipeline integrates three massive open-source datasets, comprising millions of records:
* **HM Land Registry Price Paid Data (LR-PPD):** Exact transaction prices and legal property features.
* **Energy Performance Certificates (EPC):** Structural characteristics, floor area, and energy ratings.
* **Office for National Statistics (ONS):** Exact geographic coordinates (Latitude/Longitude) mapped via Postcode Directory.

## 🛠️ Pipeline Architecture
The project is structured into sequential Jupyter Notebooks/Colab environments:
1. `01_Data_Integration_and_Preprocessing`: Cross-referencing LR-PPD and EPC data, executing strict geographical filtering (London, Manchester, Cornwall), and spatial coordinate merging.
2. `02_EDA_and_Modelling`: Exploratory Data Analysis, temporal/spatial profiling, and correlation mapping.
3. `03_Hedonic_Pricing_Model`: Establishing the linear baseline using Ordinary Least Squares (OLS) with Outcode Fixed Effects.
4. `04_ML_LightGBM`: Gradient Boosting tree implementation and Optuna hyperparameter optimization.
5. `05_ML_Random_Forest`: Bagging architecture implementation and performance evaluation.
6. `06_ML_XGBoost_and_SHAP`: Training the ultimate predictive champion (XGBoost) and deploying SHAP (Game Theory) to extract local interpretability and exact £ impacts of the EPC ratings.
7. `07_Model_Comparison`: Cross-evaluating global metrics ($R^2$, MAE, RMSE, MAPE).

## 🚀 Key Results
* **Predictive Supremacy:** The XGBoost model achieved an $R^2$ of 0.8247, reducing the average transaction error by ~£18,595 compared to the linear baseline.
* **The Green Premium:** The SHAP analysis quantified an average premium of **+£12,515** for Band A properties and a severe "Brown Discount" of **-£12,734** for Band G properties.
* **Spatial Heterogeneity:** The model proved that hyper-tensioned markets (e.g., Greater London) impose extreme penalties for inefficiency, while historical properties (built before 1930) experience a "historical forgiveness" effect, mitigating the Brown Discount.

## ⚙️ Tech Stack
* **Language:** Python 3
* **Machine Learning:** Scikit-Learn, XGBoost, LightGBM
* **Optimization:** Optuna (Bayesian Search)
* **Explainable AI (XAI):** SHAP
* **Data Manipulation & Viz:** Pandas, NumPy, Matplotlib, Seaborn

---
*Developed by João Marques*
