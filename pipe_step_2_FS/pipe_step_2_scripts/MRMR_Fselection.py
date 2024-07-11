import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import numpy as np
import os

def load_data(file_path):
    """Load the dataset from the given path and return as a DataFrame."""
    data = pd.read_csv(file_path)
    return data

def mutual_information(X, y):
    """Calculate mutual information between each feature in X and the target y."""
    return mutual_info_classif(X, y, discrete_features='auto')

def mrmr_feature_selection(data):
    """Select and rank features based on MRMR method for all features."""
    X = data.drop(columns=['class'])  # Assuming 'class' is the target variable
    y = data['class']
    
    # Calculate relevance (mutual information with the target)
    relevance = mutual_information(X, y)
    features = X.columns
    
    # Placeholder for more sophisticated redundancy calculation
    # For simplicity, we are not calculating redundancy between features in this example
    # Normally, you'd calculate and subtract redundancy here to adjust the relevance scores
    
    # Sorting features by their relevance score
    sorted_features = [feature for _, feature in sorted(zip(relevance, features), reverse=True)]
    
    # Re-arranging the dataset
    sorted_columns = ['class'] + sorted_features  # Keeping 'class' as the first column
    sorted_data = data[sorted_columns]
    
    return sorted_data

if __name__ == "__main__":
    file_path = 'pipe_step_2_input/input.csv'  # Update this to your dataset's path
    output_dir = "pipe_step_2_output"  # Update this to your output directory
    output_file_name = "mrmr_ranking.csv"  # Name of the output file

    data = load_data(file_path)
    sorted_data = mrmr_feature_selection(data)
    
    # Save the rearranged dataset
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    sorted_data.to_csv(os.path.join(output_dir, output_file_name), index=False)
    print("Feature ranking based on MRMR saved successfully.")
