import os
import pandas as pd

# Get the absolute path to the data file
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, '../data/spam.csv')

print(f"Looking for data at: {data_path}")  # Debug print

def load_data():
    try:
        # Verify file exists first
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"File not found at {data_path}")
        
        df = pd.read_csv(data_path, encoding='latin-1')
        print("Dataset loaded successfully!")
        print(f"\nFirst message:\n{df.iloc[0]['v2']}")  # Changed to 'v2' as per your dataset
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

if __name__ == "__main__":
    load_data()