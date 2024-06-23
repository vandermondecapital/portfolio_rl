import torch
import numpy as np
from replay_buffer import ReplayBuffer
from portfolio_simulator import PortfolioSimulator
from torch_geometric.data import Data
min_assets = 10
max_assets = 10
import torch
import numpy as np
from replay_buffer import ReplayBuffer
from portfolio_simulator import PortfolioSimulator
from torch_geometric.data import Data
import torch
import numpy as np
from replay_buffer import ReplayBuffer
from portfolio_simulator import PortfolioSimulator
from torch_geometric.data import Data
import torch
import numpy as np
from replay_buffer import ReplayBuffer
from portfolio_simulator import PortfolioSimulator
from torch_geometric.data import Data
class PODQN:
    def __init__(self, model, target_model, state_dim, action_dim, replay_buffer, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
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
        self.loss_fn = torch.nn.SmoothL1Loss()
        self.update_target_network()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def choose_action(self, data):
        num_assets = data.num_assets
        valid_actions = np.arange(2 * num_assets)
        
        if np.random.rand() < self.epsilon:
            action = np.random.choice(valid_actions)
            # print(f"Random action chosen: {action}")
            return action
        
        data = Data(x=data.x.clone().detach(), edge_index=data.edge_index, edge_attr=data.edge_attr, num_assets=data.num_assets)
        data = data.to(next(self.model.parameters()).device)
        self.model.eval()
        with torch.no_grad():
            q_values = self.model(data)
        q_values = q_values.cpu().numpy().flatten()
        
        valid_q_values = q_values[valid_actions]
        action = valid_actions[np.argmax(valid_q_values)]
        
        # print(f"q_values: {q_values}")
        # print(f"Action chosen by model: {action}")
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

        # Compute Q-values
        q_values = torch.cat([self.model(s) for s in state])
        next_q_values = torch.cat([self.target_model(ns) for ns in next_state])

        if not torch.isnan(q_values).any() and not torch.isnan(next_q_values).any():
            # Compute target Q-values
            next_q_values, _ = torch.max(next_q_values, dim=1, keepdim=True)
            target_q_values = reward + (self.gamma * next_q_values * (1 - done))
            
            # Compute the loss
            q_value = q_values.gather(1, action).squeeze(1)
            loss = self.loss_fn(q_value, target_q_values.squeeze(1))
            
            if not torch.isnan(loss).any():
                self.optimizer.zero_grad()
                loss.backward()
                for param in self.model.parameters():
                    if param.grad != None:
                        param.grad.data.clamp_(-1, 1)
                
                self.optimizer.step()

                self.update_epsilon()
