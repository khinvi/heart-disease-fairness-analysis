"""
Setup script for heart disease fairness analysis project.
"""

from setuptools import setup, find_packages

setup(
    name="heart_disease_fairness_analysis",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "kagglehub",
        "xgboost",
        "aif360",
        "tensorflow",
        "wordcloud",
    ],
    description="Fairness analysis of heart disease prediction models",
    keywords="fairness, machine learning, healthcare, bias mitigation, heart disease",
    url="https://github.com/khinvi/heart-disease-fairness-analysis",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.8",
)