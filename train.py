import random
import torch
from data_loader import DataLoader
from graph_data_creator import GraphDataCreator
from model import GNN
from podqn import PODQN
from replay_buffer import ReplayBuffer
from portfolio_simulator import PortfolioSimulator
from torch_geometric.data import Data
import random
import torch
from data_loader import DataLoader
from graph_data_creator import GraphDataCreator
from model import GNN
from podqn import PODQN
from replay_buffer import ReplayBuffer
from portfolio_simulator import PortfolioSimulator
from torch_geometric.data import Data
import numpy as np
# Parameters
import random
import torch
from data_loader import DataLoader
from graph_data_creator import GraphDataCreator
from model import GNN
from podqn import PODQN
from replay_buffer import ReplayBuffer
from portfolio_simulator import PortfolioSimulator
from torch_geometric.data import Data

import random
import torch
from data_loader import DataLoader
from graph_data_creator import GraphDataCreator
from model import GNN
from podqn import PODQN
from replay_buffer import ReplayBuffer
from portfolio_simulator import PortfolioSimulator
from torch_geometric.data import Data

import random
import torch
from data_loader import DataLoader
from graph_data_creator import GraphDataCreator
from model import GNN
from podqn import PODQN
from replay_buffer import ReplayBuffer
from portfolio_simulator import PortfolioSimulator
from torch_geometric.data import Data

import random
import torch
from data_loader import DataLoader
from graph_data_creator import GraphDataCreator
from model import GNN
from podqn import PODQN
from replay_buffer import ReplayBuffer
from portfolio_simulator import PortfolioSimulator
from torch_geometric.data import Data

# Parameters
min_assets = 5
max_assets = 25
mode = 'sharpe'  # Set mode to 'sharpe' or 'correlation'
update_target_freq = 100  # Frequency of updating the target network
replay_buffer_capacity = 10000
batch_size = 64
learning_rate = 0.0001  # Lower learning rate for stability
from tqdm import tqdm
# Training loop
def train_podqn(agent, data_list, portfolio_metrics, epochs=1000):
    for epoch in range(epochs):
        total_loss = 0
        valid = 0
        for i, data in tqdm(enumerate(data_list)):
            state = data
            # print(f"State x shape: {state.x.shape}")
            # print(f"State edge_index shape: {state.edge_index.shape}")
            # print(f"State num_assets: {state.num_assets}")
            action = agent.choose_action(state)
            next_state_weights = PortfolioSimulator.adjust_weights(state.x[:, 0].numpy(), action)
            next_state = Data(x=torch.tensor(np.column_stack((next_state_weights, state.x[:, 1].numpy())), dtype=torch.float), edge_index=state.edge_index, edge_attr=state.edge_attr, num_assets=state.num_assets)
            # print(f"Next state x shape: {next_state.x.shape}")
            reward = PortfolioSimulator.calculate_sharpe_ratio(next_state.x[:, 0].numpy(), data.stock_returns.numpy()) if mode == 'sharpe' else PortfolioSimulator.calculate_correlation(next_state.x[:, 0].numpy(), data.stock_returns.numpy())
            done = False  # Define termination condition if necessary
            done = torch.tensor(done, dtype=torch.float).to(next(agent.model.parameters()).device)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            agent.update_epsilon()
            agent.train_step(batch_size)
            if not np.isnan(reward) and reward!='nan':
                total_loss += reward  # Simplified; adapt based on your loss function
                valid += 1

            if (i + 1) % update_target_freq == 0:
                agent.update_target_network()

        print(f'Epoch {epoch+1}/{epochs}, Total Reward: {total_loss/valid}')

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
    num_actions = 2 * max_assets
    online_net = GNN(num_features=num_features, num_actions=num_actions)
    target_net = GNN(num_features=num_features, num_actions=num_actions)

    # Initialize Replay Buffer
    replay_buffer = ReplayBuffer(replay_buffer_capacity)

    # Initialize PODQN agent
    state_dim = num_features
    action_dim = num_actions  # 2 * maximum number of assets
    agent = PODQN(online_net, target_net, state_dim, action_dim, replay_buffer)

    # Train the PODQN agent
    train_podqn(agent, nodes, portfolio_metrics)

    print("Training complete.")
