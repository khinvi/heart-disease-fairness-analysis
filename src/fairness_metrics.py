"""
Fairness metrics module for evaluating and comparing model fairness.

This module provides functions to calculate various fairness metrics for model evaluation.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score


def calculate_accuracy_parity(y_female_test, y_female_pred, y_male_test, y_male_pred):
    """
    Calculate accuracy parity difference between male and female groups.
    
    Args:
        y_female_test: Ground truth labels for female group
        y_female_pred: Predicted labels for female group
        y_male_test: Ground truth labels for male group
        y_male_pred: Predicted labels for male group
        
    Returns:
        float: Absolute difference in accuracy between groups
    """
    accuracy_female = accuracy_score(y_female_test, y_female_pred)
    accuracy_male = accuracy_score(y_male_test, y_male_pred)
    
    return abs(accuracy_female - accuracy_male)


def calculate_demographic_parity(y_female_pred, y_male_pred):
    """
    Calculate demographic parity difference between male and female groups.
    
    Args:
        y_female_pred: Predicted labels for female group
        y_male_pred: Predicted labels for male group
        
    Returns:
        float: Absolute difference in positive prediction rates
    """
    pos_rate_female = np.mean(y_female_pred)
    pos_rate_male = np.mean(y_male_pred)
    
    return abs(pos_rate_female - pos_rate_male)


def calculate_equalized_odds(y_female_test, y_female_pred, y_male_test, y_male_pred):
    """
    Calculate equalized odds difference (TPR parity) between male and female groups.
    
    Args:
        y_female_test: Ground truth labels for female group
        y_female_pred: Predicted labels for female group
        y_male_test: Ground truth labels for male group
        y_male_pred: Predicted labels for male group
        
    Returns:
        float: Absolute difference in true positive rates between groups
    """
    tpr_female = recall_score(y_female_test, y_female_pred)
    tpr_male = recall_score(y_male_test, y_male_pred)
    
    return abs(tpr_female - tpr_male)


def calculate_all_fairness_metrics(y_female_test, y_female_pred, y_male_test, y_male_pred):
    """
    Calculate all fairness metrics between male and female groups.
    
    Args:
        y_female_test: Ground truth labels for female group
        y_female_pred: Predicted labels for female group
        y_male_test: Ground truth labels for male group
        y_male_pred: Predicted labels for male group
        
    Returns:
        dict: Dictionary of fairness metrics
    """
    # Accuracy for each group
    accuracy_female = accuracy_score(y_female_test, y_female_pred)
    accuracy_male = accuracy_score(y_male_test, y_male_pred)
    
    # True positive rates for each group
    tpr_female = recall_score(y_female_test, y_female_pred)
    tpr_male = recall_score(y_male_test, y_male_pred)
    
    # Positive prediction rates for each group
    pos_rate_female = np.mean(y_female_pred)
    pos_rate_male = np.mean(y_male_pred)
    
    # Calculate fairness metrics
    accuracy_parity_diff = abs(accuracy_female - accuracy_male)
    demographic_parity_diff = abs(pos_rate_female - pos_rate_male)
    equalized_odds_diff = abs(tpr_female - tpr_male)
    
    return {
        'accuracy_female': accuracy_female,
        'accuracy_male': accuracy_male,
        'tpr_female': tpr_female,
        'tpr_male': tpr_male,
        'pos_rate_female': pos_rate_female,
        'pos_rate_male': pos_rate_male,
        'accuracy_parity_diff': accuracy_parity_diff,
        'demographic_parity_diff': demographic_parity_diff,
        'equalized_odds_diff': equalized_odds_diff
    }


def discrimination_score(X, preds, sensitive_attr='Sex'):
    """
    Calculate discrimination score based on sensitive attribute.
    
    Args:
        X: Feature data containing sensitive attribute
        preds: Predicted labels
        sensitive_attr: Name of sensitive attribute column
        
    Returns:
        float: Discrimination score
    """
    if isinstance(X, pd.DataFrame):
        pos_female = np.mean(preds[X[sensitive_attr] == 1])  # Female group (assuming 1 for female)
        pos_male = np.mean(preds[X[sensitive_attr] == 0])    # Male group (assuming 0 for male)
    else:
        # If X is a numpy array, get the column index of sensitive attribute
        sensitive_attr_idx = 0  # Adjust index based on feature order
        pos_female = np.mean(preds[X[:, sensitive_attr_idx] == 1])
        pos_male = np.mean(preds[X[:, sensitive_attr_idx] == 0])
    
    return abs(pos_female - pos_male)  # Absolute discrimination score


def compare_fairness_metrics(metrics_before, metrics_after):
    """
    Compare fairness metrics before and after interventions.
    
    Args:
        metrics_before: Dictionary of metrics before intervention
        metrics_after: Dictionary of metrics after intervention
        
    Returns:
        pandas.DataFrame: Comparison table
    """
    comparison = pd.DataFrame({
        'Metric': list(metrics_before.keys()),
        'Before Intervention': list(metrics_before.values()),
        'After Intervention': list(metrics_after.values()),
        'Improvement': [metrics_after[m] - metrics_before[m] for m in metrics_before.keys()]
    })
    
    return comparison