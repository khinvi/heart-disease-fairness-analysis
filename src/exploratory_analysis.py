"""
Exploratory data analysis module for heart disease dataset.

This module provides functions for data exploration, visualization, and analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


def summarize_dataset(df):
    """
    Provide summary statistics of the dataset.
    
    Args:
        df: Input dataframe
        
    Returns:
        dict: Summary statistics
    """
    summary = {}
    
    # Overall dataset stats
    summary['total_rows'] = len(df)
    summary['columns'] = df.columns.tolist()
    summary['missing_values'] = df.isnull().sum().sum()
    
    # Target distribution
    summary['target_distribution'] = df['HeartDisease'].value_counts().to_dict()
    summary['target_balance'] = df['HeartDisease'].value_counts(normalize=True).to_dict()
    
    # Gender distribution
    summary['gender_distribution'] = df['Sex'].value_counts().to_dict()
    
    # Race distribution
    summary['race_distribution'] = df['Race'].value_counts().to_dict()
    
    return summary


def group_statistics(df, group_column, target_column='HeartDisease'):
    """
    Calculate statistics grouped by a specific column.
    
    Args:
        df: Input dataframe
        group_column: Column to group by
        target_column: Target variable column
        
    Returns:
        pandas.DataFrame: Group statistics
    """
    # Group by the specified column and calculate statistics
    stats = df.groupby(group_column)[target_column].agg(['count', 'mean']).reset_index()
    stats.columns = [group_column, 'Count', 'Positive_Rate']
    
    return stats


def plot_distribution(df, column, title=None, figsize=(10, 6)):
    """
    Plot the distribution of a categorical column.
    
    Args:
        df: Input dataframe
        column: Column to plot
        title: Plot title (default: None)
        figsize: Figure size (default: (10, 6))
        
    Returns:
        matplotlib.figure.Figure: The generated plot
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.countplot(x=column, data=df, palette='viridis')
    
    if title:
        plt.title(title)
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    return fig


def plot_distribution_by_target(df, column, target_column='HeartDisease', title=None, figsize=(10, 6)):
    """
    Plot the distribution of a column by target variable.
    
    Args:
        df: Input dataframe
        column: Column to plot
        target_column: Target variable column (default: 'HeartDisease')
        title: Plot title (default: None)
        figsize: Figure size (default: (10, 6))
        
    Returns:
        matplotlib.figure.Figure: The generated plot
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.countplot(x=column, hue=target_column, data=df, palette='viridis')
    
    if title:
        plt.title(title)
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    return fig


def plot_correlated_features(df, target_column='HeartDisease', top_n=10, figsize=(12, 10)):
    """
    Plot correlation matrix of top features correlated with target.
    
    Args:
        df: Input dataframe
        target_column: Target variable column (default: 'HeartDisease')
        top_n: Number of top correlated features to include (default: 10)
        figsize: Figure size (default: (12, 10))
        
    Returns:
        matplotlib.figure.Figure: The generated plot
    """
    # Convert categorical columns to numeric
    df_numeric = df.copy()
    for col in df_numeric.select_dtypes(include=['object']).columns:
        df_numeric[col] = pd.Categorical(df_numeric[col]).codes
    
    # Calculate correlation with target
    correlations = df_numeric.corr()[target_column].sort_values(ascending=False)
    top_corr = correlations.head(top_n + 1)  # +1 to include target
    
    # Create correlation matrix with top features
    top_features = top_corr.index.tolist()
    corr_matrix = df_numeric[top_features].corr()
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
    plt.title(f'Correlation Matrix of Top {top_n} Features with {target_column}')
    plt.tight_layout()
    
    return fig


def plot_age_distribution(df, target_column='HeartDisease', figsize=(12, 6)):
    """
    Plot age distribution by heart disease status.
    
    Args:
        df: Input dataframe
        target_column: Target variable column (default: 'HeartDisease')
        figsize: Figure size (default: (12, 6))
        
    Returns:
        matplotlib.figure.Figure: The generated plot
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Count occurrences by age category and target
    age_pos = df[df[target_column] == 'Yes']['AgeCategory'].value_counts().sort_index()
    age_neg = df[df[target_column] == 'No']['AgeCategory'].value_counts().sort_index()
    
    # Create array for x-axis positions
    categories = age_pos.index.tolist()
    x = np.arange(len(categories))
    
    # Bar width
    width = 0.4
    
    # Create bars
    ax.bar(x - width/2, age_pos.values, width=width, label=f"{target_column} = Yes", color="red", alpha=0.7)
    ax.bar(x + width/2, age_neg.values, width=width, label=f"{target_column} = No", color="blue", alpha=0.7)
    
    # Labels and title
    ax.set_xlabel("Age Category", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distribution of Heart Disease by Age Category", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45)
    ax.legend()
    
    plt.tight_layout()
    
    return fig


def plot_feature_importance(importance_df, title="Feature Importance", figsize=(10, 8)):
    """
    Plot feature importance from a model.
    
    Args:
        importance_df: DataFrame with 'Feature' and 'Importance' columns
        title: Plot title (default: "Feature Importance")
        figsize: Figure size (default: (10, 8))
        
    Returns:
        matplotlib.figure.Figure: The generated plot
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort by importance
    df = importance_df.sort_values(by='Importance', ascending=True)
    
    # Plot horizontal bar chart
    ax.barh(df['Feature'], df['Importance'])
    
    # Add labels and title
    ax.set_xlabel('Importance')
    ax.set_title(title)
    
    plt.tight_layout()
    
    return fig


def compare_feature_importance(male_importance, female_importance, figsize=(12, 10)):
    """
    Compare feature importance between male and female models.
    
    Args:
        male_importance: DataFrame with male feature importance
        female_importance: DataFrame with female feature importance
        figsize: Figure size (default: (12, 10))
        
    Returns:
        matplotlib.figure.Figure: The generated plot
    """
    # Merge dataframes
    merged_df = pd.merge(
        male_importance.rename(columns={'Importance': 'Male_Importance'}),
        female_importance.rename(columns={'Importance': 'Female_Importance'}),
        on='Feature'
    )
    
    # Sort by average importance
    merged_df['Avg_Importance'] = (merged_df['Male_Importance'] + merged_df['Female_Importance']) / 2
    merged_df = merged_df.sort_values(by='Avg_Importance', ascending=True)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    features = merged_df['Feature']
    x = np.arange(len(features))
    width = 0.35
    
    ax.barh(x - width/2, merged_df['Male_Importance'], width, label='Male', color='blue', alpha=0.7)
    ax.barh(x + width/2, merged_df['Female_Importance'], width, label='Female', color='red', alpha=0.7)
    
    ax.set_yticks(x)
    ax.set_yticklabels(features)
    ax.legend()
    
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance Comparison: Male vs Female')
    
    plt.tight_layout()
    
    return fig


def analyze_categorical_distributions(df, categorical_cols=None, target_col='HeartDisease'):
    """
    Analyze the distribution of categorical variables with respect to the target.
    
    Args:
        df: Input dataframe
        categorical_cols: List of categorical columns (default: None, infers from dtype)
        target_col: Target variable column (default: 'HeartDisease')
        
    Returns:
        dict: Dictionary of DataFrames with conditional probabilities
    """
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if target_col in categorical_cols:
            categorical_cols.remove(target_col)
    
    result = {}
    
    for col in categorical_cols:
        # Calculate P(target=Yes | feature=value) for each value of the feature
        conditional_prob = df.groupby(col)[target_col].apply(
            lambda x: (x == 'Yes').mean()
        ).reset_index()
        conditional_prob.columns = [col, f'P({target_col}=Yes|{col})']
        
        # Add count of each category
        category_counts = df[col].value_counts().reset_index()
        category_counts.columns = [col, 'Count']
        
        # Merge
        result[col] = pd.merge(conditional_prob, category_counts, on=col)
        
    return result


def generate_word_cloud(df, target_val='Yes', title="Heart Disease Word Cloud", figsize=(10, 6)):
    """
    Generate a word cloud visualization of features for target class.
    
    Args:
        df: Input dataframe
        target_val: Target value to filter on (default: 'Yes')
        title: Plot title (default: "Heart Disease Word Cloud")
        figsize: Figure size (default: (10, 6))
        
    Returns:
        matplotlib.figure.Figure: The generated plot
    """
    # Filter data for target value
    df_filtered = df[df['HeartDisease'] == target_val]
    
    # Select categorical columns
    categorical_cols = [
        "Smoking", "AlcoholDrinking", "Stroke", "DiffWalking",
        "Diabetic", "PhysicalActivity", "Asthma", "KidneyDisease", "SkinCancer",
        "AgeCategory", "Race", "GenHealth"
    ]
    
    # Format text for word cloud
    formatted_text = set()
    
    for _, row in df_filtered.iterrows():
        row_text = set()
        
        for col in categorical_cols:
            if pd.notna(row[col]):
                row_text.add(f"{col}_{row[col]}")
        
        formatted_text.update(row_text)
    
    text = ' '.join(formatted_text)
    
    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=14)
    
    return fig