import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, BatchNorm
from itertools import combinations
import random

# Parameters
min_assets = 3
max_assets = 10

# Load correlation data
def load_correlation_data(corr_file):
    corr_df = pd.read_csv(corr_file)
    edges = {}
    for _, row in corr_df.iterrows():
        stock_pair = (row['Stock1'], row['Stock2'])
        edges[stock_pair] = [row['daily_corr'], row['weekly_corr'], row['monthly_corr']]
    return edges

# Load stock data and create a dictionary of DataFrames
def load_stock_data(tickers_dir):
    ticker_files = sorted([f for f in os.listdir(tickers_dir) if f.endswith('.csv')])
    stock_data = {}
    for file in ticker_files:
        ticker = file.split('.')[0]
        df = pd.read_csv(os.path.join(tickers_dir, file), index_col='Date', parse_dates=True)
        stock_data[ticker] = df['Adj Close']
    return stock_data

# Sample portfolio weights ensuring they sum to 1 and are less than 50%
def sample_portfolio_weights(num_stocks):
    weights = np.random.dirichlet(np.ones(num_stocks), size=1).flatten()
    print(weights)
    # while np.any(weights > 0.5):
    #     weights = np.random.dirichlet(np.ones(num_stocks), size=1).flatten()
    return weights

# Calculate portfolio standard deviation based on historical returns
# def calculate_portfolio_std(weights, stock_returns):
#     portfolio_return = np.dot(weights, stock_returns.mean())
#     portfolio_var = np.dot(weights.T, np.dot(stock_returns.cov(), weights))
#     return np.sqrt(portfolio_var)
# Calculate portfolio standard deviation based on historical returns
def calculate_portfolio_std(weights, stock_returns):
    # Calculate daily portfolio returns
    daily_portfolio_returns = np.dot(stock_returns, weights)
    # Calculate the standard deviation of the daily portfolio returns
    portfolio_std = np.std(daily_portfolio_returns)
    if portfolio_std and not np.isnan(portfolio_std) and portfolio_std != 'nan':
        return portfolio_std*1000
    else:
        return None
# Create graph data using the stock data and correlation data
def create_graph_data(stock_data, correlations, num_samples=10000):
    nodes = []
    portfolio_stds = []

    tickers = list(stock_data.keys())

    for _ in range(num_samples):
        num_assets = random.randint(min_assets, max_assets)
        sampled_tickers = random.sample(tickers, num_assets)
        weights = sample_portfolio_weights(num_assets)
        stock_returns = pd.DataFrame({ticker: stock_data[ticker].pct_change().dropna() for ticker in sampled_tickers})
        portfolio_std = calculate_portfolio_std(weights, stock_returns)
        print(portfolio_std)

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

        # Fully connect the graph and add self-loops with correlation 1
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
        nodes.append(data)
        portfolio_stds.append(portfolio_std)

    return nodes, portfolio_stds

import torch
import torch.nn.functional as F
from torch_geometric.nn import GlobalAttention, GATConv
from torch.nn import BatchNorm1d, Linear

class GNN(torch.nn.Module):
    def __init__(self, num_features, num_classes=1):
        super(GNN, self).__init__()
        self.conv1 = GATConv(num_features, 32, heads=8, concat=True, dropout=0.1)
        self.bn1 = BatchNorm1d(32 * 8)
        self.conv2 = GATConv(32 * 8, 64, heads=8, concat=True, dropout=0.1)
        self.bn2 = BatchNorm1d(64 * 8)
        self.conv3 = GATConv(64 * 8, 32, heads=8, concat=True, dropout=0.1)
        self.bn3 = BatchNorm1d(32 * 8)
        
        self.attention_pooling = GlobalAttention(gate_nn=Linear(32 * 8, 1))
        self.fc1 = Linear(32 * 8, 16)
        self.fc2 = Linear(16, num_classes)
        self.dropout = torch.nn.Dropout(p=0.1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        x = self.dropout(x)
        
        # Apply global attention pooling
        x = self.attention_pooling(x, data.batch)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


# Train the GNN model
def train_gnn(model, data_list, portfolio_stds, epochs=1000, learning_rate=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i, data in enumerate(data_list):
            optimizer.zero_grad()
            if portfolio_stds[i] != None:
                output = model(data).view(-1).mean()
                target = torch.tensor([portfolio_stds[i]], dtype=torch.float)
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(data_list)}')

if __name__ == "__main__":
    # Load correlation data
    corr_file = 'corr.csv'
    correlations = load_correlation_data(corr_file)

    # Load stock data
    tickers_dir = 'tickers'
    stock_data = load_stock_data(tickers_dir)

    # Create graph data
    nodes, portfolio_stds = create_graph_data(stock_data, correlations)

    # Initialize model
    model = GNN(num_features=1)

    # Train the model
    train_gnn(model, nodes, portfolio_stds)

    print("Training complete.")
