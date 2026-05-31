import pandas as pd
try:
    from prophet import Prophet
except ImportError:
    from fbprophet import Prophet  # fallback if using older prophet

import matplotlib.pyplot as plt

def ts_forecast(df):
    ts_df = df.groupby('Date').agg({'Avg_Price_USD':'mean'}).reset_index()
    ts_df.rename(columns={'Date':'ds','Avg_Price_USD':'y'}, inplace=True)
    model = Prophet(yearly_seasonality=True)
    model.fit(ts_df)
    future = model.make_future_dataframe(periods=12, freq='M')
    forecast = model.predict(future)
    fig1 = model.plot(forecast)
    plt.title("12-Month Price Forecast (Prophet)")
    plt.savefig('price_forecast_prophet.png')
    plt.close()

if __name__ == "__main__":
    df = pd.read_pickle("data_features.pkl")
    ts_forecast(df)