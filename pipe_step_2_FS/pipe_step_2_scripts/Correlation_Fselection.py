import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def load_data(file_path):
    """Load the dataset from the given path and return as a DataFrame."""
    data = pd.read_csv(file_path)
    return data

def rank_features_correlation(data):
    """Rank features based on their absolute correlation with the target variable."""
    # Label encoding class labels
    le = LabelEncoder()
    data['class_encoded'] = le.fit_transform(data['class'])
    
    # One-hot encoding the numerical class labels
    enc = OneHotEncoder(sparse_output=False, drop=None)
    class_encoded = enc.fit_transform(data[['class_encoded']])
    class_encoded_df = pd.DataFrame(class_encoded, columns=enc.get_feature_names_out(['class_encoded']))

    # Calculate correlation of each feature with each one-hot encoded class column
    correlations = data.drop(columns=['class', 'class_encoded']).corrwith(class_encoded_df, axis=0)

    # If correlations is a Series, calculate the average correlations and convert to a Series
    if isinstance(correlations, pd.Series):
        avg_correlations = pd.Series(correlations.abs().mean(), index=correlations.index)
    else:
        avg_correlations = correlations.abs().mean(axis=1)

    # Rank features based on average absolute correlation values
    correlation_ranking = avg_correlations.sort_values(ascending=False)
    
    return correlation_ranking, le

if __name__ == "__main__":
    # Predefined input and output paths based on the .sh file directory structure
    input_path = 'pipe_step_2_input/input.csv'
    output_path = 'pipe_step_2_output/correlation_ranking.csv'

    # Ensure necessary directories exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load data
    data = load_data(input_path)

    # Rank features based on correlation and get the LabelEncoder
    correlation_ranking, le = rank_features_correlation(data)

    # Ensure only the original feature columns are included in the reordering
    feature_columns = [col for col in correlation_ranking.index if col in data.columns]

    # Reorder the dataset columns based on the feature ranking
    reordered_data = data[feature_columns + ['class']]

    # Save the reordered dataset to the specified output file
    reordered_data.to_csv(output_path, index=False)
    print("Feature ranking based on correlation saved successfully.")
