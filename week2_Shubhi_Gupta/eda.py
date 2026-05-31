import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(df):
    plt.figure(figsize=(8,5))
    sns.histplot(df['Avg_Price_USD'], bins=40, kde=True)
    plt.title('Average Price Distribution')
    plt.savefig('eda_avg_price_distribution.png'); plt.close()

    plt.figure(figsize=(8,5))
    sns.scatterplot(data=df, x='Estimated_Deliveries', y='Avg_Price_USD', hue='Model')
    plt.title('Deliveries vs Avg Price by Model')
    plt.savefig('eda_deliveries_vs_price.png'); plt.close()

if __name__ == "__main__":
    df = pd.read_pickle("data_preprocessed.pkl")
    perform_eda(df)