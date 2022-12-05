from typing import Iterable
import numpy as np

import torch
import torch.nn as nn
from torch import optim

class PiApproximationWithNN():
    def __init__(self,
                 state_dims,
                 num_actions,
                 alpha,
                 hidden_units=32):
        """
        state_dims: the number of dimensions of state space
        action_dims: the number of possible actions
        alpha: learning rate
        """
        self.model = nn.Sequential(
            nn.Linear(state_dims, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, num_actions),
            nn.Softmax(dim=-1)
        )
        self.num_actions = num_actions
        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha, betas=(0.9, 0.999))

    def __call__(self,s) -> torch.tensor:
        self.model.eval()
        input = torch.from_numpy(s).float()
        return self.model(input)
    
    def update(self, s, a, gamma_t, delta):
        """
        s: state S_t
        a: action A_t
        gamma_t: gamma^t
        delta: G-v(S_t,w)
        """
        self.model.train()

        input = torch.from_numpy(s).float()
        pi_a = self.model(input)[a]

        # grad l(x) = - delta*gamma_t*ln pi(a|s), so by integral we get
        # loss = delta*gamma_t*(pi_a*torch.log(pi_a) - pi_a)
        loss = -1*delta*gamma_t*torch.log(pi_a)

        # Use the Adam optimizer to update the parameters of the neural network based on the computed loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Return the loss value
        return loss.item()

class Baseline(object):
    """
    The dumbest baseline; a constant for every state
    """
    def __init__(self,b):
        self.b = b

    def __call__(self,s) -> float:
        return self.b

    def update(self,s,G):
        return 0

class VApproximationWithNN(Baseline):
    def __init__(self,
                 state_dims,
                 alpha,
                 hidden_units=32):
        """
        state_dims: the number of dimensions of state space
        alpha: learning rate
        """
        self.model = nn.Sequential(
            nn.Linear(state_dims, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 1),
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha, betas=(0.9, 0.999))

    def __call__(self,s) -> float:
        self.model.eval()
        input = torch.from_numpy(s).float()
        return self.model(input)[0].item()

    def update(self,s,G):
        self.model.train()

        input = torch.from_numpy(s).float()
        predicted_value = self.model(input)

        # Compute the loss between the predicted and target values using the MSE loss
        loss = 0.5*nn.MSELoss()(predicted_value, torch.tensor(G))

        # Use the Adam optimizer to update the parameters of the neural network based on the computed loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Return the loss value
        return loss.item()

def REINFORCE(
    env, #open-ai environment
    gamma:float,
    num_episodes:int,
    pi:PiApproximationWithNN,
    V:Baseline) -> Iterable[float]:
    """
    implement REINFORCE algorithm with and without baseline.

    input:
        env: target environment; openai gym
        gamma: discount factor
        num_episode: #episodes to iterate
        pi: policy
        V: baseline
    output:
        a list that includes the G_0 for every episodes.
    """
    Gs = []

    for episode_i in range(num_episodes):
        states, actions, rewards, done = [env.reset()], [], [], False

        # generate episode
        while not done:
            action_probs = pi(states[-1])
            action = np.argmax(np.random.multinomial(1, action_probs.detach().numpy()))
            state, reward, done, _ = env.step(action)
            # env.render()

            states.append(state)
            actions.append(action)
            rewards.append(reward)

        T = len(states) - 1 # ignore final state/action
        gammas = np.power(gamma, range(T))
        R = np.array(rewards)
        V_loss = 0
        pi_loss = 0

        # REINFORCE
        for t in range(T):
            S_t = states[t]
            A_t = actions[t]

            G = np.sum(gammas[:(T-t)] * R[t:])
            if t == 0: Gs.append(G)

            delta = G - V(S_t)
            V_loss += V.update(S_t, G)
            pi_loss += pi.update(S_t, A_t, gammas[t], delta)
        print(f'Episode {episode_i}: V_loss {V_loss} pi_loss {pi_loss} G_0 {Gs[-1]}')

    return Gs