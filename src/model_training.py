"""
Model training module for heart disease prediction with fairness considerations.

This module provides functions for training and evaluating different models
with and without fairness interventions.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def train_baseline_models(X_train, y_train, model_types=None):
    """
    Train baseline models without fairness interventions.
    
    Args:
        X_train: Training features
        y_train: Training labels
        model_types: List of model types to train (default: ['lr', 'rf', 'xgb'])
        
    Returns:
        dict: Trained models
    """
    if model_types is None:
        model_types = ['lr', 'rf', 'xgb']
    
    models = {}
    
    if 'lr' in model_types:
        models['lr'] = LogisticRegression(random_state=42, max_iter=1000)
        models['lr'].fit(X_train, y_train)
    
    if 'rf' in model_types:
        models['rf'] = RandomForestClassifier(n_estimators=100, random_state=42)
        models['rf'].fit(X_train, y_train)
    
    if 'xgb' in model_types:
        models['xgb'] = xgb.XGBClassifier(random_state=42)
        models['xgb'].fit(X_train, y_train)
    
    return models


def train_reweighted_models(X_train, y_train, sample_weights, model_types=None):
    """
    Train models with sample weights for fairness intervention.
    
    Args:
        X_train: Training features
        y_train: Training labels
        sample_weights: Weights for each sample
        model_types: List of model types to train (default: ['lr', 'rf', 'xgb'])
        
    Returns:
        dict: Trained models with reweighting
    """
    if model_types is None:
        model_types = ['lr', 'rf', 'xgb']
    
    models = {}
    
    if 'lr' in model_types:
        models['lr'] = LogisticRegression(random_state=42, max_iter=1000)
        models['lr'].fit(X_train, y_train, sample_weight=sample_weights)
    
    if 'rf' in model_types:
        models['rf'] = RandomForestClassifier(n_estimators=100, random_state=42)
        models['rf'].fit(X_train, y_train, sample_weight=sample_weights)
    
    if 'xgb' in model_types:
        models['xgb'] = xgb.XGBClassifier(random_state=42)
        models['xgb'].fit(X_train, y_train, sample_weight=sample_weights)
    
    return models


def evaluate_model(model, X_test, y_test, return_probs=False):
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        return_probs: Whether to return prediction probabilities
        
    Returns:
        dict: Evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
    except:
        y_prob = None
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    
    if y_prob is not None:
        metrics['auc_roc'] = roc_auc_score(y_test, y_prob)
    
    if return_probs:
        return metrics, y_pred, y_prob
    else:
        return metrics


def compare_models(models, X_test, y_test, model_names=None):
    """
    Compare multiple trained models on test data.
    
    Args:
        models: Dictionary of trained models
        X_test: Test features
        y_test: Test labels
        model_names: Optional list of names to use instead of dictionary keys
        
    Returns:
        pandas.DataFrame: Comparison of model performance
    """
    results = []
    
    for model_name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test)
        metrics['model'] = model_name if model_names is None else model_names[model_name]
        results.append(metrics)
    
    return pd.DataFrame(results)


def get_feature_importance(model, feature_names, model_type=None):
    """
    Extract feature importances from a trained model.
    
    Args:
        model: Trained model
        feature_names: List of feature names
        model_type: Type of model ('lr', 'rf', or 'xgb')
        
    Returns:
        pandas.DataFrame: Feature importances sorted by importance
    """
    if model_type is None:
        if isinstance(model, LogisticRegression):
            model_type = 'lr'
        elif isinstance(model, RandomForestClassifier):
            model_type = 'rf'
        elif isinstance(model, xgb.XGBClassifier):
            model_type = 'xgb'
        else:
            raise ValueError("Unknown model type")
    
    if model_type == 'lr':
        importances = abs(model.coef_[0])
    elif model_type in ['rf', 'xgb']:
        importances = model.feature_importances_
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    return importance_df


def train_gender_specific_models(X_female_train, y_female_train, X_male_train, y_male_train, model_type='lr'):
    """
    Train separate models for female and male groups.
    
    Args:
        X_female_train: Training features for female group
        y_female_train: Training labels for female group
        X_male_train: Training features for male group
        y_male_train: Training labels for male group
        model_type: Type of model to train ('lr', 'rf', or 'xgb')
        
    Returns:
        tuple: (Female model, Male model)
    """
    if model_type == 'lr':
        clf_female = LogisticRegression(random_state=42, max_iter=1000)
        clf_male = LogisticRegression(random_state=42, max_iter=1000)
    elif model_type == 'rf':
        clf_female = RandomForestClassifier(n_estimators=100, random_state=42)
        clf_male = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == 'xgb':
        clf_female = xgb.XGBClassifier(random_state=42)
        clf_male = xgb.XGBClassifier(random_state=42)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    clf_female.fit(X_female_train, y_female_train)
    clf_male.fit(X_male_train, y_male_train)
    
    return clf_female, clf_male