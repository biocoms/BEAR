import pandas as pd
import numpy as np
import os
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy

def load_data(file_path):
    """Load the dataset from the given path and return as a DataFrame."""
    data = pd.read_csv(file_path)
    return data

def calculate_entropy(series):
    """Calculate the entropy of a pandas Series."""
    value_counts = series.value_counts(normalize=True)
    return entropy(value_counts, base=2)

def symmetrical_uncertainty(X, y):
    """Calculate the Symmetrical Uncertainty between each feature in X and the target variable y."""
    su_scores = []

    entropy_target = calculate_entropy(y)

    for col in X.columns:
        # Convert feature to discrete values if it is continuous
        if X[col].dtype == 'float' or X[col].dtype == 'int':
            X[col] = pd.cut(X[col], bins=10, labels=False)

        # Calculate Mutual Information between the feature and the target
        mi = mutual_info_score(y, X[col])

        # Calculate entropy of the feature
        entropy_feature = calculate_entropy(X[col])

        # Calculate Symmetrical Uncertainty
        su = 2.0 * mi / (entropy_feature + entropy_target) if (entropy_feature + entropy_target) != 0 else 0

        su_scores.append(su)

    return su_scores

def rank_features_sym_uncertainty(data):
    """Rank features based on their symmetrical uncertainty with the target variable."""
    X = data.drop(columns=['class'])  # Exclude 'class' during the feature selection
    y = data['class']

    # Calculate symmetrical uncertainty scores
    su_scores = symmetrical_uncertainty(X, y)

    # Create a Series for the SU scores and feature names
    su_series = pd.Series(su_scores, index=X.columns)

    # Rank features based on SU scores
    su_ranking = su_series.sort_values(ascending=False)
    return su_ranking

if __name__ == "__main__":
    # Predefined input and output paths based on the .sh file directory structure
    input_path = 'pipe_step_2_input/input.csv'
    output_path = 'pipe_step_2_output/sym_uncertainty_ranking.csv'
    
    # Ensure necessary directories exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load data
    data = load_data(input_path)
    
    # Rank features based on symmetrical uncertainty
    su_ranking = rank_features_sym_uncertainty(data)
    
    # Reorder the dataset columns based on the feature ranking and include 'class'
    reordered_data = data[list(su_ranking.index) + ['class']]
    
    # Save the reordered dataset to the specified output file
    reordered_data.to_csv(output_path, index=False)
    print("Feature ranking based on symmetrical uncertainty saved successfully.")
