import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GATConv, BatchNorm, AttentionalAggregation
from torch.nn import BatchNorm1d, Linear
from itertools import combinations
import random
from collections import deque

# Parameters
min_assets = 5
max_assets = 25
mode = 'sharpe'  # Set mode to 'sharpe' or 'correlation'
update_target_freq = 100  # Frequency of updating the target network
replay_buffer_capacity = 10000
batch_size = 64
learning_rate = 0.0001  # Lower learning rate for stability

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done
        
    def __len__(self):
        return len(self.buffer)

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

    @staticmethod
    def adjust_weights(weights, action):
        """
        Adjusts the weights of the portfolio based on the action taken.
        Action is an integer representing 2*N possible actions:
        - 0 to N-1: Shrink the weight of asset i by 2x
        - N to 2*N-1: Expand the weight of asset i by 2x
        """
        num_assets = len(weights)
        asset_index = action % num_assets
        is_expand = action >= num_assets

        new_weights = weights.copy()
        if is_expand:
            new_weights[asset_index] *= 2
        else:
            new_weights[asset_index] /= 2

        # Normalize weights to sum to 1
        new_weights /= new_weights.sum()
        return new_weights

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
            data.num_assets = num_assets  # Store number of assets in the data object
            data.stock_returns = torch.tensor(stock_returns.values, dtype=torch.float)  # Store stock returns in the data object
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
        
        self.attention_pooling = AttentionalAggregation(gate_nn=Linear(32 * 8, 1))
        self.fc1 = Linear(32 * 8, 16)
        self.dropout = torch.nn.Dropout(p=0.1)
        self.output_layers = {}
        
        # Remove the weight initialization

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        x = self.dropout(x)
        
        # Apply attentional aggregation pooling
        x = self.attention_pooling(x, data.batch)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        num_assets = data.num_assets
        if num_assets not in self.output_layers:
            self.output_layers[num_assets] = Linear(16, 2 * num_assets)
            self.output_layers[num_assets].to(x.device)
        
        output_layer = self.output_layers[num_assets]
        q_values = output_layer(x)
        
        return q_values

class PODQN:
    def __init__(self, model, target_model, state_dim, action_dim, replay_buffer, lr=learning_rate, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.model = model
        self.target_model = target_model
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.replay_buffer = replay_buffer
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = torch.nn.MSELoss()
        self.update_target_network()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def choose_action(self, data):
        if np.random.rand() < self.epsilon:
            action = np.random.randint(2 * data.num_assets)
            print(f"Random action chosen: {action}")
            return action
        data = Data(x=data.x.clone().detach(), edge_index=data.edge_index, edge_attr=data.edge_attr, num_assets=data.num_assets)
        data = data.to(next(self.model.parameters()).device)
        self.model.eval()
        with torch.no_grad():
            q_values = self.model(data)
        print(f"q_values: {q_values.cpu().numpy()}")
        action = np.argmax(q_values.cpu().numpy())
        print(f"Action chosen by model: {action}")
        return action

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train_step(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return
        
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state = [Data(x=s.x.clone().detach(), edge_index=s.edge_index, edge_attr=s.edge_attr, num_assets=s.num_assets).to(next(self.model.parameters()).device) for s in state]
        next_state = [Data(x=ns.x.clone().detach(), edge_index=ns.edge_index, edge_attr=ns.edge_attr, num_assets=ns.num_assets).to(next(self.model.parameters()).device) for ns in next_state]
        reward = torch.tensor(reward, dtype=torch.float).unsqueeze(1).to(next(self.model.parameters()).device)
        action = torch.tensor(action, dtype=torch.long).unsqueeze(1).to(next(self.model.parameters()).device)
        done = torch.tensor(done, dtype=torch.float).unsqueeze(1).to(next(self.model.parameters()).device)

        losses = []
        for s, a, r, ns, d in zip(state, action, reward, next_state, done):
            q_values = self.model(s)
            next_q_values = self.target_model(ns)
            
            print(f"q_values shape: {q_values.shape}")
            print(f"action shape: {a.shape}")
            print(f"reward: {r}")
            print(f"done: {d}")
            
            q_value = q_values.gather(1, a).squeeze(1)
            next_q_value = r + self.gamma * next_q_values.max(1)[0] * (1 - d).squeeze(1)
            
            loss = self.loss_fn(q_value, next_q_value.detach())
            losses.append(loss)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        total_loss = sum(losses).item() / batch_size
        print(f"Loss: {total_loss}")

# Training loop
def train_podqn(agent, data_list, portfolio_metrics, epochs=1000):
    for epoch in range(epochs):
        total_loss = 0
        for i, data in enumerate(data_list):
            state = data
            print(f"State x shape: {state.x.shape}")
            print(f"State edge_index shape: {state.edge_index.shape}")
            print(f"State num_assets: {state.num_assets}")
            action = agent.choose_action(state)
            next_state_weights = PortfolioSimulator.adjust_weights(state.x[:, 0].numpy(), action)
            next_state = Data(x=torch.tensor(np.column_stack((next_state_weights, state.x[:, 1].numpy())), dtype=torch.float), edge_index=state.edge_index, edge_attr=state.edge_attr, num_assets=state.num_assets)
            print(f"Next state x shape: {next_state.x.shape}")
            reward = PortfolioSimulator.calculate_sharpe_ratio(next_state.x[:, 0].numpy(), data.stock_returns.numpy())
            done = False  # Define termination condition if necessary
            agent.replay_buffer.push(state, action, reward, next_state, done)
            agent.update_epsilon()
            agent.train_step(batch_size)
            total_loss += reward  # Simplified; adapt based on your loss function

            if (i + 1) % update_target_freq == 0:
                agent.update_target_network()

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
    online_net = GNN(num_features=num_features)
    target_net = GNN(num_features=num_features)

    # Initialize Replay Buffer
    replay_buffer = ReplayBuffer(replay_buffer_capacity)

    # Initialize PODQN agent
    state_dim = num_features
    action_dim = 2 * max_assets  # 2 * maximum number of assets
    agent = PODQN(online_net, target_net, state_dim, action_dim, replay_buffer)

    # Train the PODQN agent
    train_podqn(agent, nodes, portfolio_metrics)

    print("Training complete.")
