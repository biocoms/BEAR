import pandas as pd
import os
import argparse

def preprocess_data(folder_path, n_columns):
    for subdir, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('.csv'):
                file_path = os.path.join(subdir, filename)
                try:
                    df = pd.read_csv(file_path, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(file_path, encoding='ISO-8859-1')

                # Check if 'class' column exists
                if 'class' in df.columns:
                    # Ensure the 'class' column is at the end after trimming
                    class_col = df.pop('class')
                    df = df.iloc[:, :n_columns]  # Trim to n columns
                    df['class'] = class_col  # Add the 'class' column back
                else:
                    # Handle the absence of 'class' column
                    print(f"'class' column not found in {file_path}. Skipping or handling accordingly.")
                    # If needed, trim the DataFrame to the specified number of columns
                    df = df.iloc[:, :n_columns]

                df.to_csv(file_path, index=False, encoding='utf-8')  # Save with UTF-8 encoding
                print(f'Processed {file_path}')  # Print the path of the processed file for confirmation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess CSV files in a directory by trimming columns.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing CSV files.")
    parser.add_argument("n_columns", type=int, help="Number of columns to retain (excluding the 'class' column).")
    
    args = parser.parse_args()
    
    preprocess_data(args.folder_path, args.n_columns)
