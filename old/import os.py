import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GATConv, BatchNorm, GlobalAttention
from torch.nn import BatchNorm1d, Linear
from itertools import combinations
import random

# Parameters
min_assets = 5
max_assets = 25
mode = 'sharpe'  # Set mode to 'sharpe' or 'correlation'

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
            stock_data[ticker] = df['Adj Close']
        return stock_data

class PortfolioSimulator:
    @staticmethod
    def sample_portfolio_weights(num_stocks):
        weights = np.random.dirichlet(np.ones(num_stocks), size=1).flatten()
        while np.any(weights > 0.5):
            weights = np.random.dirichlet(np.ones(num_stocks), size=1).flatten()
        return weights

    @staticmethod
    def calculate_portfolio_std(weights, stock_returns):
        daily_portfolio_returns = np.dot(stock_returns, weights)
        portfolio_std = np.std(daily_portfolio_returns)
        return portfolio_std

    @staticmethod
    def calculate_sharpe_ratio(weights, stock_returns):
        daily_portfolio_returns = np.dot(stock_returns, weights)
        mean_return = np.mean(daily_portfolio_returns)
        std_return = np.std(daily_portfolio_returns)
        sharpe_ratio = mean_return / std_return
        return sharpe_ratio

class GraphDataCreator:
    @staticmethod
    def create_graph_data(stock_data, correlations, num_samples=1000):
        nodes = []
        portfolio_metrics = []

        tickers = list(stock_data.keys())

        for _ in range(num_samples):
            num_assets = random.randint(min_assets, max_assets)
            sampled_tickers = random.sample(tickers, num_assets)
            weights = PortfolioSimulator.sample_portfolio_weights(num_assets)

            # Sample a random 360-day period for returns
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
            portfolio_metrics.append(metric)

        return nodes, portfolio_metrics

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

class PODQN:
    def __init__(self, model, state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.model = model
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = torch.nn.MSELoss()

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return np.argmax(q_values.numpy())

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        next_state = torch.tensor(next_state, dtype=torch.float).unsqueeze(0)
        reward = torch.tensor(reward, dtype=torch.float).unsqueeze(0)
        action = torch.tensor(action, dtype=torch.long).unsqueeze(0)
        done = torch.tensor(done, dtype=torch.float).unsqueeze(0)

        q_values = self.model(state)
        next_q_values = self.model(next_state)
        
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = reward + self.gamma * next_q_values.max(1)[0] * (1 - done)
        
        loss = self.loss_fn(q_value, next_q_value.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Training loop
def train_podqn(agent, data_list, portfolio_metrics, epochs=1000):
    for epoch in range(epochs):
        total_loss = 0
        for i, data in enumerate(data_list):
            state = data.x.numpy()
            action = agent.choose_action(state)
            next_state = state.copy()
            # Modify next_state based on the action taken (shrink or expand weights)
            reward = portfolio_metrics[i]
            done = False  # Define termination condition if necessary
            agent.train_step(state, action, reward, next_state, done)
            agent.update_epsilon()
            total_loss += reward  # Simplified; adapt based on your loss function
        print(f'Epoch {epoch+1}/{epochs}, Total Reward: {total_loss/len(data_list)}')

if __name__ == "__main__":
    # Load correlation data
    corr_file = 'corr.csv'
    correlations = DataLoader.load_correlation_data(corr_file)

    # Load stock data
    tickers_dir = 'tickers'
    stock_data = DataLoader.load_stock_data(tickers_dir)

    # Create graph data
    nodes, portfolio_metrics = GraphDataCreator.create_graph_data(stock_data, correlations)

    # Initialize model
    num_features = 2 if mode == 'sharpe' else 1
    model = GNN(num_features=num_features)

    # Initialize PODQN agent
    state_dim = num_features
    action_dim = 2  # Shrink or expand
    agent = PODQN(model, state_dim, action_dim)

    # Train the PODQN agent
    train_podqn(agent, nodes, portfolio_metrics)

    print("Training complete.")
