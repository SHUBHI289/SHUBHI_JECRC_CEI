import pandas as pd
from sklearn.preprocessing import LabelEncoder

def feature_engineer(df):
    # Label Encoding
    cat_cols = ['Region', 'Model', 'Source_Type']
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col])
    # Date & temporal
    df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(DAY=1))
    df = df.sort_values('Date').reset_index(drop=True)
    # New features
    df['Deliveries_per_Station'] = df['Estimated_Deliveries'] / df['Charging_Stations']
    return df

if __name__ == "__main__":
    df = pd.read_pickle("data_preprocessed.pkl")
    df = feature_engineer(df)
    df.to_pickle("data_features.pkl")