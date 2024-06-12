
import pandas as pd
import numpy as np
import os
from sklearn.feature_selection import mutual_info_classif

def load_data(file_path):
    """Load the dataset from the given path and return as a DataFrame."""
    data = pd.read_csv(file_path)
    return data

def calculate_entropy(feature):
    """Calculate the entropy of a feature."""
    # Calculate the probability of each unique value in the feature
    probabilities = feature.value_counts(normalize=True)
    
    # Calculate entropy
    entropy = -np.sum([prob * np.log2(prob) for prob in probabilities])
    return entropy

def rank_features_info_gain_ratio(data):
    """Rank features based on their information gain ratio with the target variable."""
    X = data.drop(columns=['class'])  # Exclude 'CAM_ID' during the feature selection
    y = data['class']

    # Calculate information gain
    info_gain = mutual_info_classif(X, y, discrete_features='auto', random_state=0)

    # Calculate entropy for each feature
    entropies = X.apply(calculate_entropy)
    
    # Avoid division by zero by adding a small constant
    entropies += 1e-10

    # Calculate information gain ratio
    info_gain_ratio = info_gain / entropies

    # Create a Series for the info gain ratio scores and feature names
    info_gain_ratio_series = pd.Series(info_gain_ratio, index=X.columns)

    # Rank features based on information gain ratio scores
    info_gain_ratio_ranking = info_gain_ratio_series.sort_values(ascending=False)
    return info_gain_ratio_ranking

if __name__ == "__main__":
    # Predefined input and output paths based on the .sh file directory structure
    input_path = 'pipe_step_2_input/input.csv'
    output_path = 'pipe_step_2_output/info_gain_ratio_ranking.csv'
    
    # Ensure necessary directories exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load data
    data = load_data(input_path)
    
    # Save 'CAM_ID' for later use
    #cam_id = data[['CAM_ID']]
    
    # Rank features based on information gain ratio
    info_gain_ratio_ranking = rank_features_info_gain_ratio(data)
    
    # Reorder the dataset columns based on the feature ranking and include 'CAM_ID'
    reordered_data = data[list(info_gain_ratio_ranking.index)+ ['class']]
    #reordered_data = pd.concat([cam_id, reordered_data], axis=1)
    #reordered_data['class'] = reordered_data['class'].map({0: 'Negative', 1: 'Positive'})
    
    # Save the reordered dataset to the specified output file
    reordered_data.to_csv(output_path, index=False)
    print("Feature ranking based on information gain ratio saved successfully.")
