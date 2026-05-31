import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

def tune_hyperparameters(df):
    features = [
        'Year', 'Month', 'Region', 'Model', 'Estimated_Deliveries', 'Production_Units',
        'Battery_Capacity_kWh', 'Range_km', 'CO2_Saved_tons', 'Source_Type', 'Charging_Stations',
        'Deliveries_per_Station'
    ]
    X = df[features]
    y = df['Avg_Price_USD']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    param_grid = {
        'n_estimators': [60, 100],
        'max_depth': [6, 15],
    }
    grid = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, scoring='neg_root_mean_squared_error')
    grid.fit(X_train_s, y_train)
    print("Best Params:", grid.best_params_)
    print("Best RMSE:", -grid.best_score_)

if __name__ == "__main__":
    df = pd.read_pickle("data_features.pkl")
    tune_hyperparameters(df)