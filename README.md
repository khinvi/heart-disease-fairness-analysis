# Heart Disease Prediction Fairness Analysis

This repository contains a comprehensive fairness analysis of machine learning models for heart disease prediction, using the 2020 BRFSS Heart Disease Dataset from the CDC's Behavioral Risk Factor Surveillance System.

## Project Overview

The analysis focuses on gender-based fairness in heart disease prediction models, examining how different models and fairness interventions affect prediction equity between male and female groups. If interested please take a look at our [Colab Jupyter Notebook](https://github.com/khinvi/heart-disease-fairness-analysis/blob/main/Assignment2_291J.ipynb) for further insights and analysis.

### Key Components:
- **Fairness Metrics**: Accuracy Parity, Demographic Parity, and Equalized Odds
- **Models Evaluated**: Logistic Regression, Random Forest
- **Fairness Interventions**: 
  - Pre-processing: Reweighting
  - Post-processing: Threshold adjustment for equalized odds

## Dataset

This study uses the 2020 BRFSS Heart Disease Dataset from the CDC's Behavioral Risk Factor Surveillance System (BRFSS), which collects 400,000+ annual interviews on health indicators across the U.S.

Dataset: [Kaggle Link](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease/data)

## Repository Structure

```
heart-disease-fairness-analysis/
├── data/                  # Data directory with README
├── src/                   # Source code modules
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── exploratory_analysis.py
│   ├── fairness_metrics.py
│   ├── model_training.py
│   └── fairness_interventions.py
├── visualizations/        # Generated visualizations
│   ├── exploratory/
│   ├── models/
│   └── fairness/
├── README.md
├── requirements.txt
├── main.py               # Main analysis script
└── LICENSE
```

## Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running the Analysis
1. Download the dataset from Kaggle using the link above
2. Place the dataset files in the `data/2020/` directory
3. Run the main script:
   ```bash
   python main.py
   ```
4. View the generated visualizations in the `visualizations/` directory

## Key Visualizations

### Exploratory Analysis
- Gender distribution in dataset
- Heart disease rates by gender
- Feature correlation matrix

### Model Performance
- Model comparison (accuracy, precision, recall)
- Feature importance visualization

### Fairness Analysis
- Fairness metrics across interventions
- Accuracy vs. fairness trade-off
- Intervention comparison

## References

M. H. Shahrezaei, R. Loughran, and K. M. Daid, "Pre-processing Techniques to Mitigate Against Algorithmic Bias," 2023 31st Irish Conference on Artificial Intelligence and Cognitive Science (AICS), Letterkenny, Ireland, 2023, pp. 1-4, doi: 10.1109/AICS60730.2023.10470759
