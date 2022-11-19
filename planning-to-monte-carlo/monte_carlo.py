from typing import Iterable, Tuple

import numpy as np
from env import EnvSpec
from policy import Policy

def off_policy_mc_prediction_ordinary_importance_sampling(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    bpi:Policy,
    pi:Policy,
    initQ:np.array
) -> np.array:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        pi: evaluation target policy
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_pi$ function; numpy array shape of [nS,nA]
    """

    Q = np.copy(initQ)
    C = np.zeros((env_spec.nS, env_spec.nA))

    for episode in trajs:
        G = 0
        W = 1
        # s_{t-1}, a_{t-1}, r_t, s_t
        for (state, action, reward, _state_p) in episode:
            if W == 0:
                break
            
            G = env_spec.gamma * G + reward
            C[state, action] += 1
            Q[state, action] += W / C[state, action] * (G - Q[state, action])
            W *= pi.action_prob(state, action) / bpi.action_prob(state, action)

    return Q

def off_policy_mc_prediction_weighted_importance_sampling(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    bpi:Policy,
    pi:Policy,
    initQ:np.array
) -> np.array:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using behavior policy bpi
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        pi: evaluation target policy
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_pi$ function; numpy array shape of [nS,nA]
    """

    Q = np.copy(initQ)
    C = np.zeros((env_spec.nS, env_spec.nA))

    for episode in trajs:
        G = 0
        W = 1
        # s_{t-1}, a_{t-1}, r_t, s_t
        for (state, action, reward, _state_p) in episode:
            if W == 0:
                break
            
            G = env_spec.gamma * G + reward
            C[state, action] += W
            Q[state, action] += W / C[state, action] * (G - Q[state, action])
            W *= pi.action_prob(state, action) / bpi.action_prob(state, action)

    return Q
