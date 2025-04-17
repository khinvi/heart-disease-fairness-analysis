#!/usr/bin/env python3
"""
Simplified main script for heart disease fairness analysis.
Focuses on key analyses and essential visualizations.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Import project modules
from src import data_preprocessing
from src import exploratory_analysis
from src import fairness_metrics
from src import model_training
from src import fairness_interventions

# Set plot style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# Set random seed for reproducibility
np.random.seed(42)

def create_directories():
    """Create necessary directories for visualizations."""
    os.makedirs("visualizations/exploratory", exist_ok=True)
    os.makedirs("visualizations/models", exist_ok=True)
    os.makedirs("visualizations/fairness", exist_ok=True)

def main():
    """Main function to run the heart disease fairness analysis pipeline."""
    print("Heart Disease Fairness Analysis")
    print("=" * 30)
    
    # Create directories
    create_directories()
    
    # Step 1: Load and preprocess data
    print("\nStep 1: Loading and preprocessing data...")
    data_path = os.path.join("data", "2020", "heart_2020_cleaned.csv")
    
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}")
        print("Please download the dataset as per instructions in data/README.md")
        return
    
    df = data_preprocessing.load_data(data_path)
    print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns")
    
    # Encode categorical features
    df_encoded, encoders = data_preprocessing.encode_categorical_features(df)
    
    # Split by gender
    df_female = df[df["Sex"] == "Female"]
    df_male = df[df["Sex"] == "Male"]
    print(f"Female instances: {len(df_female)}")
    print(f"Male instances: {len(df_male)}")
    
    # Prepare features and target
    X = df_encoded.drop(columns=['HeartDisease'])
    y = df_encoded['HeartDisease']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = data_preprocessing.prepare_train_test_data(
        df_encoded, sensitive_attr="Sex", test_size=0.2, random_state=42
    )
    
    # Standardize features
    X_train_scaled, X_test_scaled, scaler = data_preprocessing.standardize_features(X_train, X_test)
    
    # Step 2: Create exploratory visualizations
    print("\nStep 2: Creating exploratory visualizations...")
    
    # Gender distribution plot
    fig = exploratory_analysis.plot_distribution(
        df, "Sex", title="Gender Distribution in Heart Disease Dataset"
    )
    fig.savefig(os.path.join("visualizations", "exploratory", "gender_distribution.png"))
    plt.close(fig)
    
    # Heart disease by gender
    fig = exploratory_analysis.plot_distribution_by_target(
        df, "Sex", target_column="HeartDisease", 
        title="Heart Disease by Gender"
    )
    fig.savefig(os.path.join("visualizations", "exploratory", "heart_disease_by_gender.png"))
    plt.close(fig)
    
    # Feature correlation matrix
    fig = exploratory_analysis.plot_correlated_features(
        df_encoded, target_column="HeartDisease", top_n=10
    )
    fig.savefig(os.path.join("visualizations", "exploratory", "feature_correlation.png"))
    plt.close(fig)
    
    # Step 3: Train and evaluate models
    print("\nStep 3: Training and evaluating models...")
    
    # Train baseline models
    models = model_training.train_baseline_models(
        X_train_scaled, y_train, model_types=['lr', 'rf']
    )
    
    # Evaluate models
    model_comparison = model_training.compare_models(models, X_test_scaled, y_test)
    print("Model Performance:")
    print(model_comparison)
    
    # Create model performance visualization
    plt.figure(figsize=(10, 6))
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
    bar_width = 0.25
    x = np.arange(len(metrics_to_plot))
    
    for i, (model_name, model) in enumerate(models.items()):
        metrics_values = [model_comparison[model_comparison['model'] == model_name][metric].values[0] 
                          for metric in metrics_to_plot]
        plt.bar(x + i*bar_width, metrics_values, width=bar_width, label=model_name)
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x + bar_width/2, metrics_to_plot)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("visualizations", "models", "model_performance.png"))
    plt.close()
    
    # Feature importance visualization
    feature_names = X_train.columns.tolist()
    importance_df = model_training.get_feature_importance(
        models['lr'], feature_names, model_type='lr'
    )
    
    fig = exploratory_analysis.plot_feature_importance(
        importance_df, title="Feature Importance (Logistic Regression)"
    )
    fig.savefig(os.path.join("visualizations", "models", "feature_importance.png"))
    plt.close(fig)
    
    # Step 4: Prepare gender-specific datasets for fairness evaluation
    print("\nStep 4: Preparing gender-specific datasets for fairness evaluation...")
    
    # Female
    df_female_encoded, _ = data_preprocessing.encode_categorical_features(df_female)
    X_female = df_female_encoded.drop(columns=['HeartDisease'])
    y_female = df_female_encoded['HeartDisease']
    X_female_train, X_female_test, y_female_train, y_female_test = train_test_split(
        X_female, y_female, test_size=0.2, random_state=42, stratify=y_female
    )
    X_female_train_scaled, X_female_test_scaled, _ = data_preprocessing.standardize_features(
        X_female_train, X_female_test
    )
    
    # Male
    df_male_encoded, _ = data_preprocessing.encode_categorical_features(df_male)
    X_male = df_male_encoded.drop(columns=['HeartDisease'])
    y_male = df_male_encoded['HeartDisease']
    X_male_train, X_male_test, y_male_train, y_male_test = train_test_split(
        X_male, y_male, test_size=0.2, random_state=42, stratify=y_male
    )
    X_male_train_scaled, X_male_test_scaled, _ = data_preprocessing.standardize_features(
        X_male_train, X_male_test
    )
    
    # Step 5: Evaluate fairness of baseline model
    print("\nStep 5: Evaluating fairness of baseline model...")
    
    # Use logistic regression as the baseline model
    model = models['lr']
    
    # Make predictions on gender-specific test sets
    y_female_pred = model.predict(X_female_test_scaled)
    y_male_pred = model.predict(X_male_test_scaled)
    
    # Calculate fairness metrics
    fairness_baseline = fairness_metrics.calculate_all_fairness_metrics(
        y_female_test, y_female_pred,
        y_male_test, y_male_pred
    )
    
    print("Fairness Metrics for Baseline Model:")
    print(f"Accuracy (Female): {fairness_baseline['accuracy_female']:.4f}")
    print(f"Accuracy (Male): {fairness_baseline['accuracy_male']:.4f}")
    print(f"Accuracy Parity Difference: {fairness_baseline['accuracy_parity_diff']:.4f}")
    print(f"Equalized Odds Difference: {fairness_baseline['equalized_odds_diff']:.4f}")
    print(f"Demographic Parity Difference: {fairness_baseline['demographic_parity_diff']:.4f}")
    
    # Step 6: Apply fairness interventions
    print("\nStep 6: Applying fairness interventions...")
    
    # Apply reweighting
    print("\nApplying reweighting intervention...")
    sample_weights = fairness_interventions.apply_reweighting(
        X_train, y_train, sensitive_attr='Sex'
    )
    
    # Train a model with reweighting
    model_reweighted = LogisticRegression(random_state=42)
    model_reweighted.fit(X_train_scaled, y_train, sample_weight=sample_weights)
    
    # Evaluate fairness after reweighting
    y_female_pred_reweighted = model_reweighted.predict(X_female_test_scaled)
    y_male_pred_reweighted = model_reweighted.predict(X_male_test_scaled)
    
    fairness_reweighted = fairness_metrics.calculate_all_fairness_metrics(
        y_female_test, y_female_pred_reweighted,
        y_male_test, y_male_pred_reweighted
    )
    
    # Apply post-processing (threshold adjustment)
    print("\nApplying post-processing intervention...")
    
    # Get prediction probabilities
    y_female_prob = model.predict_proba(X_female_test_scaled)[:, 1]
    y_male_prob = model.predict_proba(X_male_test_scaled)[:, 1]
    
    # Apply equalized odds post-processing
    _, _, y_female_pred_post, y_male_pred_post = fairness_interventions.equalize_opportunity(
        y_female_test, y_female_prob,
        y_male_test, y_male_prob
    )
    
    # Evaluate fairness after post-processing
    fairness_post = fairness_metrics.calculate_all_fairness_metrics(
        y_female_test, y_female_pred_post,
        y_male_test, y_male_pred_post
    )
    
    # Step 7: Create fairness comparison visualizations
    print("\nStep 7: Creating fairness visualizations...")
    
    # Fairness metrics visualization
    intervention_names = ['Baseline', 'Reweighting', 'Post-Processing']
    fairness_results = [fairness_baseline, fairness_reweighted, fairness_post]
    
    metrics_df = pd.DataFrame({
        'Intervention': intervention_names,
        'Accuracy_Parity_Diff': [f['accuracy_parity_diff'] for f in fairness_results],
        'Equalized_Odds_Diff': [f['equalized_odds_diff'] for f in fairness_results],
        'Demographic_Parity_Diff': [f['demographic_parity_diff'] for f in fairness_results]
    })
    
    # Create fairness metrics plot
    plt.figure(figsize=(10, 6))
    x = np.arange(len(intervention_names))
    width = 0.25
    
    plt.bar(x - width, metrics_df['Accuracy_Parity_Diff'], width, 
            label='Accuracy Parity Diff', color='blue')
    plt.bar(x, metrics_df['Equalized_Odds_Diff'], width, 
            label='Equalized Odds Diff', color='green')
    plt.bar(x + width, metrics_df['Demographic_Parity_Diff'], width, 
            label='Demographic Parity Diff', color='red')
    
    plt.xlabel('Intervention')
    plt.ylabel('Difference (Lower is Better)')
    plt.title('Fairness Metrics Across Interventions')
    plt.xticks(x, intervention_names)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("visualizations", "fairness", "fairness_metrics.png"))
    plt.close()
    
    # Accuracy vs. fairness trade-off
    accuracy_values = [
        (fairness_baseline['accuracy_female'] + fairness_baseline['accuracy_male']) / 2,
        (fairness_reweighted['accuracy_female'] + fairness_reweighted['accuracy_male']) / 2,
        (fairness_post['accuracy_female'] + fairness_post['accuracy_male']) / 2
    ]
    
    fairness_values = [
        (fairness_baseline['accuracy_parity_diff'] + 
         fairness_baseline['equalized_odds_diff'] + 
         fairness_baseline['demographic_parity_diff']) / 3,
        (fairness_reweighted['accuracy_parity_diff'] + 
         fairness_reweighted['equalized_odds_diff'] + 
         fairness_reweighted['demographic_parity_diff']) / 3,
        (fairness_post['accuracy_parity_diff'] + 
         fairness_post['equalized_odds_diff'] + 
         fairness_post['demographic_parity_diff']) / 3
    ]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(fairness_values, accuracy_values, c=['blue', 'green', 'red'], s=100)
    
    for i, txt in enumerate(intervention_names):
        plt.annotate(txt, (fairness_values[i], accuracy_values[i]), 
                    xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Unfairness (Average Difference Metrics, Lower is Better)')
    plt.ylabel('Accuracy (Average)')
    plt.title('Accuracy vs. Fairness Trade-off')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join("visualizations", "fairness", "accuracy_fairness_tradeoff.png"))
    plt.close()
    
    # Intervention comparison - before/after
    plt.figure(figsize=(10, 6))
    
    # Set width of bars
    barWidth = 0.25
    
    # Set positions of bars on X axis
    r1 = np.arange(2)
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    
    # Create bars
    plt.bar(r1, [fairness_baseline['accuracy_female'], fairness_baseline['accuracy_male']], 
            width=barWidth, label='Baseline', color='blue', alpha=0.7)
    plt.bar(r2, [fairness_reweighted['accuracy_female'], fairness_reweighted['accuracy_male']], 
            width=barWidth, label='Reweighting', color='green', alpha=0.7)
    plt.bar(r3, [fairness_post['accuracy_female'], fairness_post['accuracy_male']], 
            width=barWidth, label='Post-Processing', color='red', alpha=0.7)
    
    # Add labels
    plt.xlabel('Gender Group')
    plt.ylabel('Accuracy')
    plt.title('Intervention Comparison - Gender-specific Accuracy')
    plt.xticks([r + barWidth for r in range(2)], ['Female', 'Male'])
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join("visualizations", "fairness", "intervention_comparison.png"))
    plt.close()
    
    print("\nAnalysis complete! Visualizations saved in the 'visualizations' directory.")

if __name__ == "__main__":
    main()