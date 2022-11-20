from typing import Iterable, Tuple

import numpy as np
from env import EnvSpec
from policy import Policy

def on_policy_n_step_td(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    n:int,
    alpha:float,
    initV:np.array
) -> Tuple[np.array]:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        n: how many steps?
        alpha: learning rate
        initV: initial V values; np array shape of [nS]
    ret:
        V: $v_pi$ function; numpy array shape of [nS]
    """

    #####################
    # TODO: Implement On Policy n-Step TD algorithm
    # sampling (Hint: Sutton Book p. 144)
    #####################

    V = np.copy(initV)

    # pre-compute gammas power series
    gammas = np.power(env_spec.gamma, range(n+1))

    for episode in trajs:
        T = len(episode)
        for tau in range(T):
            (state_tau, _, _, _) = episode[tau]

            len_mc = min(tau + n, T) - tau
            # TODO: this could be O(1) rather than O(n)
            rewards = np.array([reward for (_, _, reward, _) in episode[tau:tau+len_mc]])
            G = np.sum(gammas[:len_mc] * rewards)
            if tau + n < T:
                G += gammas[len_mc] * V[episode[tau+n][0]]
            V[state_tau] += alpha * (G - V[state_tau])

    return V

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

def off_policy_n_step_sarsa(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    bpi:Policy,
    n:int,
    alpha:float,
    initQ:np.array
) -> Tuple[np.array,Policy]:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        n: how many steps?
        alpha: learning rate
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_star$ function; numpy array shape of [nS,nA]
        policy: $pi_star$; instance of policy class
    """

    Q = np.copy(initQ)
    state_transitions = np.random.choice(env_spec.nA, env_spec.nS)
    pi = DeterministicPolicy(state_transitions)

    # pre-compute gammas power series
    gammas = np.power(env_spec.gamma, range(n+1))

    for episode in trajs:
        T = len(episode)
        for tau in range(T):
            (state_tau, action_tau, _, _) = episode[tau]

            rho = 1
            for i in range(tau+1, min(tau+n, T-1)+1):
                (state_i, action_i, _, _) = episode[i]
                rho *= pi.action_prob(state_i, action_i) / bpi.action_prob(state_i, action_i)

            len_mc = min(tau + n, T) - tau
            # TODO: this could be O(1) rather than O(n)
            rewards = np.array([reward for (_, _, reward, _) in episode[tau:tau+len_mc]])
            G = np.sum(gammas[:len_mc] * rewards)
            if tau + n < T:
                G += gammas[len_mc] * Q[episode[tau+n][0], episode[tau+n][1]]
            Q[state_tau, action_tau] += alpha * rho * (G - Q[state_tau, action_tau])
            state_transitions[state_tau] = np.argmax(Q[state_tau])

    return Q, pi