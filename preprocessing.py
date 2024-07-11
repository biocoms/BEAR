import pandas as pd
import os

# Specify the directory containing your CSV files
directory_path = 'BEAR/BEAR_Binary/inputs/jihyun/'

# Loop through each file in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.csv'):  # Check if the file is a CSV
        file_path = os.path.join(directory_path, filename)
        
        # Load the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        
        # Drop the first column
        df.drop(df.columns[0], axis=1, inplace=True)
        
        # Construct the output file name
        output_file_path = os.path.join(directory_path, f'modified_{filename}')
        
        # Save the modified DataFrame to a new CSV file
        df.to_csv(output_file_path, index=False)

        print(f'Processed {filename}')

