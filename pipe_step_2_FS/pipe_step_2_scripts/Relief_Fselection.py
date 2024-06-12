import pandas as pd
import numpy as np
import os
from skrebate import ReliefF

def load_data(file_path):
    """Load the dataset from the given path and return as a DataFrame."""
    data = pd.read_csv(file_path)
    return data

def preprocess_features(X):
    """Preprocess features to ensure they're suitable for ReliefF."""
    # Convert features to float
    X_float = X.astype(np.float64)
    
    # Remove or handle features with zero variance
    variance = X_float.var()
    non_zero_variance_columns = variance[variance > 0].index
    return X_float[non_zero_variance_columns]

def rank_features_relief(data):
    """Rank features based on their ReliefF scores."""
    X = data.drop(columns=['class'])  # Exclude 'class' during feature selection
    y = data['class']

    # Preprocess features
    X_preprocessed = preprocess_features(X)

    # Initialize the ReliefF algorithm
    relief = ReliefF(n_neighbors=10, n_features_to_select=X_preprocessed.shape[1])

    # Fit the model
    relief.fit(X_preprocessed.values, y.values)

    # Get feature importances
    feature_scores = relief.feature_importances_

    # Create a Series for the ReliefF scores and feature names
    relief_series = pd.Series(feature_scores, index=X_preprocessed.columns)

    # Rank features based on ReliefF scores
    relief_ranking = relief_series.sort_values(ascending=False)
    return relief_ranking

if __name__ == "__main__":
    input_path = 'pipe_step_2_input/input.csv'
    output_path = 'pipe_step_2_output/relief_ranking.csv'
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    data = load_data(input_path)
    
    relief_ranking = rank_features_relief(data)
    
    reordered_data = data[relief_ranking.index.tolist() + ['class']]
    
    reordered_data.to_csv(output_path, index=False)
    print("Feature ranking based on Relief saved successfully.")
