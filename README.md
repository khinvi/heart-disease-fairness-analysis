# Heart Disease Prediction Fairness Analysis

This repository contains a comprehensive fairness analysis of machine learning models for heart disease prediction, using the 2020 BRFSS Heart Disease Dataset from the CDC's Behavioral Risk Factor Surveillance System.

## Project Overview

The analysis focuses on gender-based fairness in heart disease prediction models, examining how different models and fairness interventions affect prediction equity between male and female groups.

### Key Components:
- **Fairness Metrics**: Accuracy Parity, Demographic Parity, and Equalized Odds
- **Models Evaluated**: Logistic Regression, Random Forest, XGBoost
- **Fairness Interventions**: 
  - Pre-processing: Massaging, Reweighting, Sampling
  - In-processing: Adversarial Debiasing, Reweighting during model training
  - Post-processing: Threshold adjustment for TPR, accuracy, and demographic parity

## Dataset

This study uses the 2020 BRFSS Heart Disease Dataset from the CDC's Behavioral Risk Factor Surveillance System (BRFSS), which collects 400,000+ annual interviews on health indicators across the U.S.

Dataset: [Kaggle Link](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease/data)

## Key Findings

- Significant gender-based disparities exist in all baseline models, with female subjects consistently receiving higher accuracy rates
- Adversarial Debiasing significantly improved fairness metrics with minimal impact on overall model performance
- Demographic distribution in the dataset shows gender and racial imbalances that affect model fairness
- Feature importance analysis reveals different predictive factors between genders

## Repository Structure

- `notebooks/`: Jupyter notebooks with the full analysis workflow
- `src/`: Python modules for data preprocessing, model training, and fairness evaluation
- `data/`: Instructions for obtaining and preparing the dataset
- `visualizations/`: Generated charts and visual results

## Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running the Analysis
1. Download the dataset from Kaggle using the link above
2. Place the dataset files in the `data/` directory
3. Run the notebooks in the following order:
   - `1_exploratory_analysis.ipynb`
   - `2_baseline_models.ipynb`
   - `3_fairness_interventions.ipynb`

## References

1. M. H. Shahrezaei, R. Loughran, and K. M. Daid, "Pre-processing Techniques to Mitigate Against Algorithmic Bias," 2023 31st Irish Conference on Artificial Intelligence and Cognitive Science (AICS), Letterkenny, Ireland, 2023, pp. 1-4, doi: 10.1109/AICS60730.2023.10470759