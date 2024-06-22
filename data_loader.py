import os
import pandas as pd

class DataLoader:
    @staticmethod
    def load_correlation_data(corr_file):
        corr_df = pd.read_csv(corr_file)
        edges = {}
        for _, row in corr_df.iterrows():
            stock_pair = (row['Stock1'], row['Stock2'])
            edges[stock_pair] = [row['daily_corr'], row['weekly_corr'], row['monthly_corr']]
        return edges

    @staticmethod
    def load_stock_data(tickers_dir):
        ticker_files = sorted([f for f in os.listdir(tickers_dir) if f.endswith('.csv')])
        stock_data = {}
        for file in ticker_files:
            ticker = file.split('.')[0]
            df = pd.read_csv(os.path.join(tickers_dir, file), index_col='Date', parse_dates=True)
            df = df['Adj Close'].dropna()  # Drop rows with NaN values
            stock_data[ticker] = df
        return stock_data