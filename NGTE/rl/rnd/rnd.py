import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class RNDModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(RNDModel, self).__init__()
        self.target = nn.Sequential(
            nn.Linear(input_size, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, output_size)
        )
        self.predictor = nn.Sequential(
            nn.Linear(input_size, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, output_size)
        )
        
        # Initialize the target network's parameters with random weights and freeze it.
        for p in self.modules():
            if isinstance(p, nn.Linear):
                nn.init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, input):
        target_feature = self.target(input)
        predict_feature = self.predictor(input)
        return predict_feature, target_feature


class RND:
    def __init__(self, state_dim, out_dim, lr, device):
        self.device = device
        self.rnd_model = RNDModel(state_dim, out_dim).to(device=device, non_blocking=True)
        self.optimizer = optim.Adam(self.rnd_model.predictor.parameters(), lr=lr)
        self.mse_loss = nn.MSELoss()
        
        self.reward_mean = np.zeros((), 'float32')
        self.reward_max = np.zeros((), 'float32')
        self.reward_count = 0.0001
        self.rewems = None 
        self.gamma = 0.99

        self.obs_mean = np.zeros(state_dim, 'float32')
        self.obs_var = np.zeros(state_dim, 'float32')
        self.obs_count = 0.0001
        

    def update(self, states):
        self.optimizer.zero_grad()
        states = torch.FloatTensor(states).to(device=self.device, non_blocking=True)
        predict_features, target_features = self.rnd_model(states)
        loss = self.mse_loss(predict_features, target_features.detach())

        loss.backward()
        self.optimizer.step()

        return loss.item()  # Returns the loss value to track the training process.


    def intrinsic_reward(self, states, update=True):
        with torch.no_grad():
            states = torch.FloatTensor(states).to(device=self.device, non_blocking=True)
            predict_features, target_features = self.rnd_model(states)
            raw_int = ((predict_features - target_features)**2).sum(dim=1).detach().cpu().numpy()
            if update:
                self.update_from_moments(raw_int)
            scaled_int = raw_int / (self.reward_max + 1e-8)
            return (scaled_int[:, None])

    
    def update_from_moments(self, x): 
        """
        target is 'reward' or 'obs'
        x is batch
        """
        batch_max = np.max(x, axis = 0)
        self.reward_max = 0.99 * self.reward_max + 0.01 * batch_max

            
    