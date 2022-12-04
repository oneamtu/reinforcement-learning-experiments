import numpy as np
import pickle
import os
from policy import Policy

class ValueFunctionWithApproximation(object):
    def __call__(self,s) -> float:
        """
        return the value of given state; \hat{v}(s)

        input:
            state
        output:
            value of the given state
        """
        raise NotImplementedError()

    def update(self,alpha,G,s_tau):
        """
        Implement the update rule;
        w <- w + \alpha[G- \hat{v}(s_tau;w)] \nabla\hat{v}(s_tau;w)

        input:
            alpha: learning rate
            G: TD-target
            s_tau: target state for updating (yet, update will affect the other states)
        ouptut:
            None
        """
        raise NotImplementedError()

def semi_gradient_n_step_td(
    env, #open-ai environment
    gamma:float,
    pi:Policy,
    n:int,
    alpha:float,
    V:ValueFunctionWithApproximation,
    num_episode:int,
):
    """
    implement n-step semi gradient TD for estimating v

    input:
        env: target environment
        gamma: discounting factor
        pi: target evaluation policy
        n: n-step
        alpha: learning rate
        V: value function
        num_episode: #episodes to iterate
    output:
        None
    """
    trajs = []

    cached_path = f'trajs.pkl'
    if os.path.exists(cached_path):
        trajs = pickle.load(open(cached_path, 'rb'))
    else:
        for episode in range(num_episode):
            states, actions, rewards, done = [env.reset()], [], [], False
            # env.render()

            while not done:
                action = pi.action(states[-1])
                state, reward, done, _ = env.step(action)
                # env.render()

                states.append(state)
                actions.append(action)
                rewards.append(reward)

            traj = list(zip(states[:-1],actions,rewards,states[1:]))
            trajs.append(traj)

        pickle.dump(trajs, open(cached_path, 'wb'))

    # pre-compute gammas power series
    gammas = np.power(gamma, range(n+1))

    for i, episode in enumerate(trajs):
        T = len(episode)
        loss = 0
        print(f'Episode {i}')
        for tau in range(T):
            (state_tau, _, _, _) = episode[tau]

            len_mc = min(tau + n, T) - tau
            # TODO: this could be O(1) rather than O(n)
            rewards = np.array([reward for (_, _, reward, _) in episode[tau:tau+len_mc]])
            G = np.sum(gammas[:len_mc] * rewards)
            if tau + n < T:
                G += gammas[len_mc] * V(episode[tau+n][0])
            loss += V.update(alpha, G, state_tau)
        print(f'Loss {loss}')

    return V