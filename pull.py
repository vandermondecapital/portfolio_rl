import yfinance as yf
import pandas as pd
import os
import datetime

# Define the start and end dates
start_date = '2014-01-01'
start_date2 = '2014-01-03'
end_date = datetime.datetime.now().strftime('%Y-%m-%d')

# Get the list of S&P 500 tickers
sp500_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()

# Create a directory to save the CSV files
if not os.path.exists('tickers'):
    os.makedirs('tickers')

# Loop through each ticker and download the stock data
for ticker in sp500_tickers:
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        
        # Check if the data goes back to the start date
        print(stock_data.index.min().strftime)
        if stock_data.index.min().strftime('%Y-%m-%d') < start_date2:
            # Save the data to a CSV file
            print(ticker, len(stock_data))
            stock_data.to_csv(f'tickers/{ticker}.csv')
            print(f"Downloaded and saved data for {ticker}")
        else:
            print(f"Ticker {ticker} does not have data going back to {start_date}")
    except Exception as e:
        print(f"Could not download data for {ticker}: {e}")

print("Download complete. Stock data saved in 'tickers' folder.")
