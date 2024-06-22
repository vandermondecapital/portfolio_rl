
tot = 0
weights = {"NVDA": 0.15, "META":0.15, "TSM":0.11, "MSFT":0.11, "GOOG": 0.09, "RXRX":0.06, "NVS":0.06, "ABNB":0.06, "ABSI":0.03, "UBER":0.03, "RDY":0.02, "AAPL":0.02, "BGNE":0.01, "INCY":0.01, "MDGL":0.01, "AMZN":0.01, "AMD":0.01, "TWST":0.01, "ENB":0.01}
returns = {"NVDA": 0.3, "META":0.3, "TSM":0.2, "MSFT":0.2, "GOOG": 0.2, "RXRX":0.2, "NVS":0.1, "ABNB":0.1, "ABSI":0.2, "UBER":0.1, "RDY":0.1, "AAPL":0.1, "BGNE":0.1, "INCY":0.1, "MDGL":0.1, "AMZN":0.1, "AMD":0.1, "TWST":0.1, "ENB":0.1}
for i, j in weights.items():
    tot += j
sp = 1-tot
weights["SPY"] = sp
#first try genetic algorithm baseline on the graph. Try doubling or halving weightings, take the gainers in risk/sharpe and move forwards, iterate X rounds and output new portfolios


#script to make graph from pflo

