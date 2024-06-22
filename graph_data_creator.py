import torch
import pandas as pd
from itertools import combinations
from torch_geometric.data import Data
from portfolio_simulator import PortfolioSimulator
import random
import numpy as np
min_assets = 5
max_assets = 25
class GraphDataCreator:
    @staticmethod
    def create_graph_data(stock_data, correlations, num_samples=10000):
        mode = 'sharpe'
        nodes = []
        portfolio_metrics = []

        tickers = list(stock_data.keys())

        for _ in range(num_samples):
            num_assets = random.randint(min_assets, max_assets)
            sampled_tickers = random.sample(tickers, num_assets)
            weights = PortfolioSimulator.sample_portfolio_weights(num_assets)

            start_date = random.choice(stock_data[sampled_tickers[0]].index[:-360])
            end_date = start_date + pd.Timedelta(days=360)
            stock_returns = pd.DataFrame({ticker: stock_data[ticker][start_date:end_date].pct_change().dropna() for ticker in sampled_tickers})
            
            if mode == 'sharpe':
                metric = PortfolioSimulator.calculate_sharpe_ratio(weights, stock_returns)
                returns = stock_returns.mean().values
                node_features = torch.tensor(np.column_stack((weights, returns)), dtype=torch.float)
            else:
                metric = PortfolioSimulator.calculate_portfolio_std(weights, stock_returns)
                node_features = torch.tensor(weights, dtype=torch.float).unsqueeze(1)

            edge_index = []
            edge_attr = []

            for stock1, stock2 in combinations(sampled_tickers, 2):
                if (stock1, stock2) in correlations:
                    edge_index.append([sampled_tickers.index(stock1), sampled_tickers.index(stock2)])
                    edge_attr.append(correlations[(stock1, stock2)])
                elif (stock2, stock1) in correlations:
                    edge_index.append([sampled_tickers.index(stock1), sampled_tickers.index(stock2)])
                    edge_attr.append(correlations[(stock2, stock1)])

            for i in range(num_assets):
                for j in range(i + 1, num_assets):
                    if [i, j] not in edge_index and [j, i] not in edge_index:
                        edge_index.append([i, j])
                        edge_index.append([j, i])
                        edge_attr.append([1.0, 1.0, 1.0])
                        edge_attr.append([1.0, 1.0, 1.0])
                edge_index.append([i, i])
                edge_attr.append([1.0, 1.0, 1.0])

            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)

            data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
            data.num_assets = num_assets
            data.stock_returns = torch.tensor(stock_returns.values, dtype=torch.float)
            nodes.append(data)
            portfolio_metrics.append(metric)

        return nodes, portfolio_metrics
