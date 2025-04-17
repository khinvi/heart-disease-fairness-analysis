"""
Fairness interventions module for implementing bias mitigation techniques.

This module provides functions for preprocessing, inprocessing, and postprocessing 
fairness interventions.
"""

import numpy as np
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_sample_weight
from collections import Counter
import math

# ----- Preprocessing Interventions -----

def apply_massaging(X, y, sensitive_attr, model=None):
    """
    Apply massaging technique to reduce discrimination in training set.
    
    Args:
        X: Feature data
        y: Target labels
        sensitive_attr: Name of sensitive attribute
        model: Initial model to get decision scores (optional)
    
    Returns:
        tuple: (Adjusted X, Adjusted y)
    """
    df_train = X.copy()
    df_train['target'] = y.copy()
    
    # Get class distributions
    privileged_group = df_train[df_train[sensitive_attr] == 1]  # Males
    unprivileged_group = df_train[df_train[sensitive_attr] == 0]  # Females
    
    # Calculate imbalances
    pos_priv = privileged_group['target'].sum()
    pos_unpriv = unprivileged_group['target'].sum()
    
    # Compute the number of samples to swap
    m = int(abs((pos_priv / len(privileged_group)) - (pos_unpriv / len(unprivileged_group))) * len(df_train))
    
    # Swap class labels for fairness
    # Uses model scores if provided, otherwise sorts by target
    if model is not None:
        # Get scores for each instance
        scores = model.decision_function(X)
        
        # Find promotion candidates (female with negative label)
        promote_candidates = [
            (score, index)
            for score, index, row, y_true in zip(scores, range(len(X)), X.to_dict("records"), y)
            if row[sensitive_attr] == 0 and y_true == 0
        ]
        
        # Find demotion candidates (male with positive label)
        demote_candidates = [
            (score, index)
            for score, index, row, y_true in zip(scores, range(len(X)), X.to_dict("records"), y)
            if row[sensitive_attr] == 1 and y_true == 1
        ]
        
        # Sort promotion candidates in descending order (closer to positive boundary)
        promote_candidates.sort(reverse=True)
        
        # Sort demotion candidates in ascending order (closer to negative boundary)
        demote_candidates.sort()
    else:
        # Simple approach without model scores
        promote_candidates = [(0, idx) for idx, row in unprivileged_group[unprivileged_group['target'] == 0].iterrows()]
        demote_candidates = [(0, idx) for idx, row in privileged_group[privileged_group['target'] == 1].iterrows()]
    
    # Ensure we have enough candidates
    m = min(m, min(len(promote_candidates), len(demote_candidates)))
    
    # Modify labels
    y_fixed = y.copy()
    
    for i in range(m):
        # Promotion
        idx = promote_candidates[i][1]
        y_fixed.iloc[idx] = 1
        
        # Demotion
        idx = demote_candidates[i][1]
        y_fixed.iloc[idx] = 0
    
    return df_train.drop(columns=['target']), y_fixed


def apply_reweighting(X, y, sensitive_attr):
    """
    Apply reweighting technique to correct for bias.
    
    Args:
        X: Feature data
        y: Target labels
        sensitive_attr: Name of sensitive attribute
        
    Returns:
        array: Sample weights for model training
    """
    # Calculate probabilities for reweighting
    p_z1 = np.mean(X[sensitive_attr] == 1)  # Probability of being in group 1
    p_z0 = 1 - p_z1  # Probability of being in group 0
    
    # Probability of Y=1 or Y=0
    p_y1 = np.mean(y)
    p_y0 = 1 - p_y1
    
    # Joint probabilities P(Z=z, Y=y)
    p_z1_y1 = np.mean((X[sensitive_attr] == 1) & (y == 1))
    p_z1_y0 = np.mean((X[sensitive_attr] == 1) & (y == 0))
    p_z0_y1 = np.mean((X[sensitive_attr] == 0) & (y == 1))
    p_z0_y0 = np.mean((X[sensitive_attr] == 0) & (y == 0))
    
    # Calculate weights using the formula: w = (P(Z=z) * P(Y=y)) / P(Z=z, Y=y)
    w_pos_group1 = (p_z1 * p_y1) / p_z1_y1 if p_z1_y1 > 0 else 0
    w_neg_group1 = (p_z1 * p_y0) / p_z1_y0 if p_z1_y0 > 0 else 0
    w_pos_group0 = (p_z0 * p_y1) / p_z0_y1 if p_z0_y1 > 0 else 0
    w_neg_group0 = (p_z0 * p_y0) / p_z0_y0 if p_z0_y0 > 0 else 0
    
    # Assign weights based on group membership and label
    weights = np.array([
        w_pos_group1 if x == 1 and y_val else
        w_neg_group1 if x == 1 and not y_val else
        w_pos_group0 if x == 0 and y_val else
        w_neg_group0
        for x, y_val in zip(X[sensitive_attr], y)
    ])
    
    return weights


def apply_sampling(X, y, sensitive_attr, method='both'):
    """
    Apply sampling techniques to balance data.
    
    Args:
        X: Feature data
        y: Target labels
        sensitive_attr: Name of sensitive attribute
        method: Sampling method ('undersample', 'oversample', or 'both')
        
    Returns:
        tuple: (Adjusted X, Adjusted y)
    """
    if method not in ['undersample', 'oversample', 'both']:
        raise ValueError("Method must be one of: 'undersample', 'oversample', 'both'")
    
    # Separate by class
    X_positive = X[y == 1]
    X_negative = X[y == 0]
    
    if method == 'undersample' or method == 'both':
        # Undersample the majority class
        n_minority = min(len(X_positive), len(X_negative))
        X_majority_undersampled = resample(
            X_negative if len(X_negative) > len(X_positive) else X_positive,
            replace=False,
            n_samples=n_minority,
            random_state=42
        )
        
        if len(X_negative) > len(X_positive):
            X_balanced = np.vstack([X_positive, X_majority_undersampled])
            y_balanced = np.concatenate([np.ones(len(X_positive)), np.zeros(len(X_majority_undersampled))])
        else:
            X_balanced = np.vstack([X_majority_undersampled, X_negative])
            y_balanced = np.concatenate([np.ones(len(X_majority_undersampled)), np.zeros(len(X_negative))])
    
    if method == 'oversample' or method == 'both':
        # Oversample the minority class
        n_majority = max(len(X_positive), len(X_negative))
        X_minority_oversampled = resample(
            X_positive if len(X_positive) < len(X_negative) else X_negative,
            replace=True,
            n_samples=n_majority,
            random_state=42
        )
        
        if len(X_positive) < len(X_negative):
            X_balanced = np.vstack([X_minority_oversampled, X_negative])
            y_balanced = np.concatenate([np.ones(len(X_minority_oversampled)), np.zeros(len(X_negative))])
        else:
            X_balanced = np.vstack([X_positive, X_minority_oversampled])
            y_balanced = np.concatenate([np.ones(len(X_positive)), np.zeros(len(X_minority_oversampled))])
    
    return X_balanced, y_balanced


# ----- Inprocessing Interventions -----

def compute_fair_weights(y, sensitive_values):
    """
    Compute sample weights to balance influence in training.
    
    Args:
        y: Target labels
        sensitive_values: Values of sensitive attribute
        
    Returns:
        array: Computed weights
    """
    class_counts = Counter(zip(y, sensitive_values))
    total_count = sum(class_counts.values())
    
    # Compute weights
    weights = {k: total_count / v for k, v in class_counts.items()}
    
    # Assign weights
    sample_weights = np.array([weights[(y_i, s_i)] for y_i, s_i in zip(y, sensitive_values)])
    
    return sample_weights


def get_balanced_weights(X, y, sensitive_attr):
    """
    Get balanced class weights for model training.
    
    Args:
        X: Feature data
        y: Target labels
        sensitive_attr: Name of sensitive attribute
        
    Returns:
        array: Sample weights for model training
    """
    # Compute class weights
    return compute_sample_weight(class_weight='balanced', y=y)


# ----- Postprocessing Interventions -----

def find_threshold_for_metric(y_true, y_prob, metric_type, target_value):
    """
    Find threshold to achieve desired metric value.
    
    Args:
        y_true: Ground truth labels
        y_prob: Prediction probabilities
        metric_type: Type of metric ('tpr', 'accuracy', 'pos_rate')
        target_value: Target value for the metric
        
    Returns:
        float: Threshold value
    """
    import sklearn.metrics as metrics
    
    thresholds = np.linspace(0, 1, 100)
    values = []
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        
        if metric_type == 'tpr':
            tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
            value = tp / (tp + fn) if (tp + fn) > 0 else 0
        elif metric_type == 'accuracy':
            value = metrics.accuracy_score(y_true, y_pred)
        elif metric_type == 'pos_rate':
            value = np.mean(y_pred)
        else:
            raise ValueError("Metric type must be one of: 'tpr', 'accuracy', 'pos_rate'")
        
        values.append(value)
    
    values = np.array(values)
    idx = np.argmin(np.abs(values - target_value))
    
    return thresholds[idx]


def equalize_opportunity(y_female_test, y_female_prob, y_male_test, y_male_prob):
    """
    Adjust prediction thresholds to equalize opportunity (TPR).
    
    Args:
        y_female_test: Ground truth for female group
        y_female_prob: Prediction probabilities for female group
        y_male_test: Ground truth for male group
        y_male_prob: Prediction probabilities for male group
        
    Returns:
        tuple: (Female threshold, Male threshold, Female predictions, Male predictions)
    """
    from sklearn.metrics import confusion_matrix
    
    # Calculate current TPRs
    default_threshold = 0.5
    y_female_pred = (y_female_prob >= default_threshold).astype(int)
    y_male_pred = (y_male_prob >= default_threshold).astype(int)
    
    tn_f, fp_f, fn_f, tp_f = confusion_matrix(y_female_test, y_female_pred).ravel()
    tn_m, fp_m, fn_m, tp_m = confusion_matrix(y_male_test, y_male_pred).ravel()
    
    tpr_female = tp_f / (tp_f + fn_f) if (tp_f + fn_f) > 0 else 0
    tpr_male = tp_m / (tp_m + fn_m) if (tp_m + fn_m) > 0 else 0
    
    # Target TPR is the average of current TPRs
    target_tpr = (tpr_female + tpr_male) / 2
    
    # Find thresholds to achieve target TPR
    threshold_female = find_threshold_for_metric(y_female_test, y_female_prob, 'tpr', target_tpr)
    threshold_male = find_threshold_for_metric(y_male_test, y_male_prob, 'tpr', target_tpr)
    
    # Apply thresholds
    y_female_adj = (y_female_prob >= threshold_female).astype(int)
    y_male_adj = (y_male_prob >= threshold_male).astype(int)
    
    return threshold_female, threshold_male, y_female_adj, y_male_adj


def equalize_accuracy(y_female_test, y_female_prob, y_male_test, y_male_prob):
    """
    Adjust prediction thresholds to equalize accuracy.
    
    Args:
        y_female_test: Ground truth for female group
        y_female_prob: Prediction probabilities for female group
        y_male_test: Ground truth for male group
        y_male_prob: Prediction probabilities for male group
        
    Returns:
        tuple: (Female threshold, Male threshold, Female predictions, Male predictions)
    """
    from sklearn.metrics import accuracy_score
    
    # Calculate current accuracies
    default_threshold = 0.5
    y_female_pred = (y_female_prob >= default_threshold).astype(int)
    y_male_pred = (y_male_prob >= default_threshold).astype(int)
    
    accuracy_female = accuracy_score(y_female_test, y_female_pred)
    accuracy_male = accuracy_score(y_male_test, y_male_pred)
    
    # Target accuracy is the average of current accuracies
    target_accuracy = (accuracy_female + accuracy_male) / 2
    
    # Find thresholds to achieve target accuracy
    threshold_female = find_threshold_for_metric(y_female_test, y_female_prob, 'accuracy', target_accuracy)
    threshold_male = find_threshold_for_metric(y_male_test, y_male_prob, 'accuracy', target_accuracy)
    
    # Apply thresholds
    y_female_adj = (y_female_prob >= threshold_female).astype(int)
    y_male_adj = (y_male_prob >= threshold_male).astype(int)
    
    return threshold_female, threshold_male, y_female_adj, y_male_adj


def equalize_demographic_parity(y_female_prob, y_male_prob):
    """
    Adjust prediction thresholds to equalize demographic parity (same positive prediction rate).
    
    Args:
        y_female_prob: Prediction probabilities for female group
        y_male_prob: Prediction probabilities for male group
        
    Returns:
        tuple: (Female threshold, Male threshold, Female predictions, Male predictions)
    """
    # Calculate current positive prediction rates
    default_threshold = 0.5
    y_female_pred = (y_female_prob >= default_threshold).astype(int)
    y_male_pred = (y_male_prob >= default_threshold).astype(int)
    
    pos_rate_female = np.mean(y_female_pred)
    pos_rate_male = np.mean(y_male_pred)
    
    # Target positive rate is the average of current rates
    target_pos_rate = (pos_rate_female + pos_rate_male) / 2
    
    # Calculate female and male quantiles
    female_quantile = np.quantile(y_female_prob, 1 - target_pos_rate)
    male_quantile = np.quantile(y_male_prob, 1 - target_pos_rate)
    
    # Apply thresholds
    y_female_adj = (y_female_prob >= female_quantile).astype(int)
    y_male_adj = (y_male_prob >= male_quantile).astype(int)
    
    return female_quantile, male_quantile, y_female_adj, y_male_adj