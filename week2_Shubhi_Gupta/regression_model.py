import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import  r2_score

def run_regression(df):
    features = [
        'Year', 'Month', 'Region', 'Model', 'Estimated_Deliveries', 'Production_Units',
        'Battery_Capacity_kWh', 'Range_km', 'CO2_Saved_tons', 'Source_Type', 'Charging_Stations',
        'Deliveries_per_Station'
    ]
    X = df[features]
    y = df['Avg_Price_USD']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    # print("RF RMSE:", mean_squared_error(y_test, y_pred, squared=False))
   
    from sklearn.metrics import root_mean_squared_error

    print("RMSE:", root_mean_squared_error(y_test, y_pred))
    print("RF R2:", r2_score(y_test, y_pred))

    pd.Series(model.feature_importances_, index=features).nlargest(10).plot(kind='barh')
    plt.title('Top 10 Feature Importances')
    plt.tight_layout()
    plt.savefig('rf_feature_importances.png')
    plt.close()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    df = pd.read_pickle("data_features.pkl")
    run_regression(df)