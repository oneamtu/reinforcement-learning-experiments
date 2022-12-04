import numpy as np
from algo import ValueFunctionWithApproximation

import torch
import torch.nn as nn
from torch import optim

class ValueFunctionWithNN:
    def __init__(self, state_dims, hidden_units=32):
        # Initialize the neural network model
        self.model = nn.Sequential(
          nn.Linear(state_dims, hidden_units),
          nn.ReLU(),
          nn.Linear(hidden_units, hidden_units),
          nn.ReLU(),
          nn.Linear(hidden_units, 1),
        )

        # Compile the model with the AdamOptimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999))

    def __call__(self, s) -> float:
        # Use the model to predict the value of the given state
        self.model.eval()
        input = torch.from_numpy(s).float()
        return self.model(input)[0].item()

    def update(self, alpha, G, s_tau):
        # Calculate the predicted value of the state at the time step where the return was calculated
        self.model.train()

        input = torch.from_numpy(s_tau).float()
        predicted_value = self.model(input)

        # Compute the loss between the predicted and target values using the MSE loss
        loss = 0.5*nn.MSELoss()(predicted_value, torch.tensor(G))

        # Use the Adam optimizer to update the parameters of the neural network based on the computed loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Return the loss value
        return loss.item()