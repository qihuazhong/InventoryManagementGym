import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingQNet(nn.Module):
    def __init__(self, n_actions, hidden_units=256, states_dim=4):
        super().__init__()
        self.action_dim = n_actions
        self.states_dim = states_dim
        
        self.encode_fc1 =  nn.Linear(states_dim, hidden_units)
        
        self.value_fc1 = nn.Linear(hidden_units, hidden_units)
        self.value_fc2 = nn.Linear(hidden_units, 1)
        
        self.advantage_fc1 = nn.Linear(hidden_units, hidden_units)
        self.advantage_fc2 = nn.Linear(hidden_units, n_actions)
        
    
    def forward(self, x):
        
        x = x.view(-1, self.states_dim)
        
        encoded_states = torch.relu(self.encode_fc1(x))
        
        value = torch.relu(self.value_fc1(encoded_states))
        value = self.value_fc2(value)
        
        advantage = torch.relu(self.advantage_fc1(encoded_states))
        advantage = self.advantage_fc2(advantage)
        
        q_values = value + advantage - torch.mean(advantage)
        
        return q_values
    
    
    
class QNet(nn.Module):

    def __init__(self, n_actions, hidden_units=256, states_dim=4):
        super().__init__()
        self.states_dim = states_dim
        self.fc1 = nn.Linear(states_dim, hidden_units)
        self.fc3 = nn.Linear(hidden_units, n_actions)
        
        
    def forward(self, x):
        
        x = x.view(-1, self.states_dim)
        x = F.relu(self.fc1(x))
        x = self.fc3(x)
        return  x 