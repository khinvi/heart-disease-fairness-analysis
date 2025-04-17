# Dataset Information

## Dataset Source
This project uses the **2020 BRFSS Heart Disease Dataset** from the CDC's **Behavioral Risk Factor Surveillance System (BRFSS)**.

Dataset: [Kaggle Link](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease/data)

## How to Obtain the Dataset

1. Download the dataset from the Kaggle link above
2. Place the downloaded files in this directory, maintaining the following structure:
   ```
   data/
   ├── 2020/
   │   └── heart_2020_cleaned.csv
   ├── 2022/
   │   └── heart_2022_cleaned.csv
   └── README.md
   ```

## Dataset Description

The BRFSS dataset collects 400,000+ annual interviews on health indicators across the United States. The heart disease dataset contains key indicators related to heart disease risk factors.

### Key Features

- **Demographic Information**: Age, Sex, Race, etc.
- **Health Indicators**: BMI, Smoking, Alcohol Consumption, etc.
- **Medical History**: Stroke, Diabetic, Asthma, Kidney Disease, etc.
- **Physical Health**: Physical Activity, Sleep Time, etc.
- **Target Variable**: HeartDisease (Yes/No)

### Dataset Statistics

- **Total Instances**: ~320,000
- **Heart Disease Positive Cases**: ~27,000 (~8.5%)
- **Heart Disease Negative Cases**: ~292,000 (~91.5%)
- **Gender Distribution**: Female (~168,000), Male (~152,000)
- **Race Distribution**: White (~245,000), Black (~23,000), Hispanic (~27,000), Asian (~8,000), Other (~11,000)

## Data Preprocessing Notes

- The dataset is already cleaned and preprocessed in the Kaggle version
- No missing values are present in the dataset
- All categorical variables are represented as strings (need encoding for modeling)
- The dataset has moderate class imbalance (Negative:Positive ≈ 11:1)