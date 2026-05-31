import pandas as pd

def preprocess(df):
    num_cols = [
        'Year', 'Month', 'Estimated_Deliveries', 'Production_Units',
        'Avg_Price_USD', 'Battery_Capacity_kWh', 'Range_km',
        'CO2_Saved_tons', 'Charging_Stations'
    ]
    # Coerce types
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # Fill missing
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())
    return df

if __name__ == "__main__":
    df = pd.read_pickle("data_loaded.pkl")
    df = preprocess(df)
    df.to_pickle("data_preprocessed.pkl")