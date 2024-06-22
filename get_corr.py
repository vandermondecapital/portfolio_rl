import os
import pandas as pd
from itertools import combinations

def calculate_correlations(stock_data, period):
    return stock_data.pct_change().corr()

def calculate_periodic_correlations(stock_data, period):
    return stock_data.resample(period).ffill().pct_change().corr()

# Directory containing the stock data
tickers_dir = 'tickers'

# Get list of CSV files in the directory
ticker_files = sorted([f for f in os.listdir(tickers_dir) if f.endswith('.csv')])

# Load data into a dictionary
stock_data = {}
for file in ticker_files:
    ticker = file.split('.')[0]
    df = pd.read_csv(os.path.join(tickers_dir, file), index_col='Date', parse_dates=True)
    stock_data[ticker] = df['Adj Close']

# Combine all stock data into a single DataFrame
all_data = pd.DataFrame(stock_data)

# Calculate correlations for the entire period
daily_corr = calculate_correlations(all_data, 'D')
weekly_corr = calculate_periodic_correlations(all_data, 'W')
monthly_corr = calculate_periodic_correlations(all_data, 'M')

# Calculate correlations for the last 100 days
last_100_days_data = all_data.tail(100)
daily_corr_100 = calculate_correlations(last_100_days_data, 'D')
weekly_corr_100 = calculate_periodic_correlations(last_100_days_data, 'W')
monthly_corr_100 = calculate_periodic_correlations(last_100_days_data, 'M')

# Prepare the results
results = []

for stock1, stock2 in combinations(all_data.columns, 2):
    result = {
        'Stock1': stock1,
        'Stock2': stock2,
        'daily_corr': daily_corr.loc[stock1, stock2],
        'weekly_corr': weekly_corr.loc[stock1, stock2],
        'monthly_corr': monthly_corr.loc[stock1, stock2],
        'daily_corr_100': daily_corr_100.loc[stock1, stock2],
        'weekly_corr_100': weekly_corr_100.loc[stock1, stock2],
        'monthly_corr_100': monthly_corr_100.loc[stock1, stock2],
    }
    results.append(result)

# Convert the results to a DataFrame
results_df = pd.DataFrame(results)

# Save the results to a CSV file
results_df.to_csv('corr.csv', index=False)

print("Correlations calculated and saved to 'corr.csv'.")
