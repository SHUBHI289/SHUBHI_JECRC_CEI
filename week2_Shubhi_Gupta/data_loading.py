import pandas as pd

def load_data(filepath):
    df = pd.read_csv(filepath)
    print(f"Loaded data shape: {df.shape}")
    return df

if __name__ == "__main__":
    df = load_data("tesla_deliveries_dataset_2015_2025.csv")
    df.to_pickle("data_loaded.pkl")