
import pandas as pd
import os
from sklearn.feature_selection import mutual_info_classif

def load_data(file_path):
    """Load the dataset from the given path and return as a DataFrame."""
    data = pd.read_csv(file_path)
    return data

def rank_features_info_gain(data):
    """Rank features based on their information gain with the target variable."""
    X = data.drop(columns=['class'])  # Exclude 'CAM_ID' during the feature selection
    y = data['class']

    # Calculate information gain
    info_gain = mutual_info_classif(X, y, discrete_features='auto', random_state=0)
    
    # Create a Series for the info gain scores and feature names
    info_gain_series = pd.Series(info_gain, index=X.columns)

    # Rank features based on information gain scores
    info_gain_ranking = info_gain_series.sort_values(ascending=False)
    return info_gain_ranking

if __name__ == "__main__":
    # Predefined input and output paths based on the .sh file directory structure
    input_path = 'pipe_step_2_input/input.csv'
    output_path = 'pipe_step_2_output/info_gain_ranking.csv'
    
    # Ensure necessary directories exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load data
    data = load_data(input_path)
    
    # Save 'CAM_ID' for later use
    #cam_id = data[['CAM_ID']]
    
    # Rank features based on information gain
    info_gain_ranking = rank_features_info_gain(data)
    
    # Reorder the dataset columns based on the feature ranking and include 'CAM_ID'
    reordered_data = data[list(info_gain_ranking.index)+ ['class']]
    #reordered_data = pd.concat([cam_id, reordered_data], axis=1)
    #reordered_data['class'] = reordered_data['class'].map({0: 'Negative', 1: 'Positive'})
    
    # Save the reordered dataset to the specified output file
    reordered_data.to_csv(output_path, index=False)
    print("Feature ranking based on information gain saved successfully.")
