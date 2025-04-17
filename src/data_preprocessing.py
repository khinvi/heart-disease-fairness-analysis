"""
Data preprocessing module for heart disease fairness analysis.

This module handles loading, cleaning, and preprocessing the 2020 BRFSS Heart Disease Dataset.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def load_data(data_path):
    """
    Load the heart disease dataset from the specified path.
    
    Args:
        data_path (str): Path to the dataset file
        
    Returns:
        pandas.DataFrame: The loaded dataset
    """
    return pd.read_csv(data_path)


def encode_categorical_features(df):
    """
    Encode categorical features using LabelEncoder.
    
    Args:
        df (pandas.DataFrame): Input dataframe with categorical features
        
    Returns:
        tuple: (processed DataFrame, dictionary of label encoders)
    """
    df_encoded = df.copy()
    categorical_columns = df_encoded.select_dtypes(include=['object']).columns
    label_encoders = {}
    
    for col in categorical_columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le
        
    return df_encoded, label_encoders


def split_by_gender(df):
    """
    Split the dataset by gender.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        
    Returns:
        tuple: (female_df, male_df)
    """
    df_female = df[df["Sex"] == "Female"]
    df_male = df[df["Sex"] == "Male"]
    
    return df_female, df_male


def split_by_race(df):
    """
    Split the dataset by racial categories.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        
    Returns:
        dict: Dictionary with race categories as keys and dataframes as values
    """
    races = df["Race"].unique()
    race_dfs = {}
    
    for race in races:
        race_dfs[race] = df[df["Race"] == race]
        
    return race_dfs


def prepare_train_test_data(df, sensitive_attr="Sex", test_size=0.2, random_state=42):
    """
    Prepare training and testing datasets with features and target.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        sensitive_attr (str): Column name of sensitive attribute
        test_size (float): Proportion of data for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X = df.drop(columns=['HeartDisease'])
    y = df['HeartDisease']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test


def standardize_features(X_train, X_test):
    """
    Standardize numerical features.
    
    Args:
        X_train (pandas.DataFrame): Training features
        X_test (pandas.DataFrame): Testing features
        
    Returns:
        tuple: (standardized X_train, standardized X_test, fitted scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler


def create_balanced_sample(df, group_col, n_samples=200, random_state=42):
    """
    Create a balanced sample from each category of a column.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        group_col (str): Column to sample by
        n_samples (int): Number of samples from each category
        random_state (int): Random seed for reproducibility
        
    Returns:
        pandas.DataFrame: Balanced sample dataframe
    """
    groups = df[group_col].unique()
    samples = []
    
    for group in groups:
        group_df = df[df[group_col] == group]
        if len(group_df) >= n_samples:
            samples.append(group_df.sample(n=n_samples, random_state=random_state))
    
    return pd.concat(samples)