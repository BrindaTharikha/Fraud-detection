import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Ensure plots directory exists
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

def basic_info(df):
    print("Dataset Shape:", df.shape)
    print("\nColumns:\n", df.columns.tolist())
    print("\nData Types:\n", df.dtypes)
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nStatistical Summary:\n", df.describe())

def target_distribution(df, target_col='Fraud_Label'):
    print(f"\nTarget Distribution ({target_col}):")
    print(df[target_col].value_counts(normalize=True))

    sns.countplot(x=target_col, data=df)
    plt.title("Target Variable Distribution")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/target_distribution.png")
    plt.close()

def plot_feature_distributions(df, numeric_cols):
    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(f"{PLOTS_DIR}/distribution_{col}.png")
        plt.close()

def detect_outliers(df, cols):
    for col in cols:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col}")
        plt.tight_layout()
        plt.savefig(f"{PLOTS_DIR}/boxplot_{col}.png")
        plt.close()

def correlation_with_target(df, target_col='Fraud_Label'):
    print(f"\nCorrelation with target ({target_col}):\n")
    correlations = df.corr(numeric_only=True)[target_col].sort_values(ascending=False)
    print(correlations)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=correlations.values, y=correlations.index)
    plt.title(f"Feature Correlation with {target_col}")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/correlation_with_{target_col}.png")
    plt.close()

def full_eda(df, target_col='Fraud_Label'):
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    basic_info(df)
    target_distribution(df, target_col=target_col)
    plot_feature_distributions(df, numeric_cols)
    detect_outliers(df, numeric_cols)
    correlation_with_target(df, target_col=target_col)
