from typing import Tuple

import numpy as np
from env import EnvWithModel
from policy import Policy

def q(env:EnvWithModel, V:np.array, state:int, action:int) -> float:
    """
    Computes q(s, a) given a model env
    inp:
        env: environment with model information, i.e. you know transition dynamics and reward function
        pi: policy
        initV: initial V(s); numpy array shape of [nS,]
        theta: exit criteria
    return:
        V: $v_\pi$ function; numpy array shape of [nS]
        Q: $q_\pi$ function; numpy array shape of [nS,nA]
    """
    q = 0
    for state_p in range(env.spec.nS):
        q += env.TD[state, action, state_p] * (env.R[state, action, state_p] + env.spec.gamma * V[state_p])
    return q


def value_prediction(env:EnvWithModel, pi:Policy, initV:np.array, theta:float) -> Tuple[np.array,np.array]:
    """
    inp:
        env: environment with model information, i.e. you know transition dynamics and reward function
        pi: policy
        initV: initial V(s); numpy array shape of [nS,]
        theta: exit criteria
    return:
        V: $v_\pi$ function; numpy array shape of [nS]
        Q: $q_\pi$ function; numpy array shape of [nS,nA]
    """

    V = np.copy(initV)
    Q = np.zeros((env.spec.nS, env.spec.nA))

    delta = theta
    while delta >= theta:
        delta = 0
        for state in range(env.spec.nS):
            old_v = V[state]
            new_v = 0
            for action in range(env.spec.nA):
                Q[state, action] = q(env, V, state, action)
                new_v += pi.action_prob(state, action) * Q[state, action]
            V[state] = new_v
            delta = np.maximum(delta, np.absolute(old_v - V[state]))

    return V, Q

class DeterministicPolicy(Policy):
    """
    Deterministic Policy

    Each state has only one (optimal) action; the probability of that action is 1
    """
    def __init__(self, state_actions:dict):
        self._state_actions = state_actions

    def action_prob(self,state,action=None):
        return 1 if self._state_actions[state] == action else 0

    def action(self,state):
        return self._state_actions[state]

def value_iteration(env:EnvWithModel, initV:np.array, theta:float) -> Tuple[np.array,Policy]:
    """
    inp:
        env: environment with model information, i.e. you know transition dynamics and reward function
        initV: initial V(s); numpy array shape of [nS,]
        theta: exit criteria
    return:
        value: optimal value function; numpy array shape of [nS]
        policy: optimal deterministic policy; instance of Policy class
    """

    V = np.copy(initV)
    optimal_state_actions = np.zeros(env.spec.nS)

    delta = theta
    while delta >= theta:
        delta = 0
        for state in range(env.spec.nS):
            v = V[state]
            optimal_q = -float("inf")
            optimal_action = None
            for action in range(env.spec.nA):
                candidate_q = q(env, V, state, action)
                if optimal_q < candidate_q:
                    optimal_q = candidate_q
                    optimal_action = action
            V[state] = optimal_q
            optimal_state_actions[state] = optimal_action
            delta = np.maximum(delta, np.absolute(v - V[state]))

    return V, DeterministicPolicy(optimal_state_actions)
