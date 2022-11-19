from functools import partial
import numpy as np
from tqdm import tqdm

from env import EnvSpec, Env, EnvWithModel
from policy import Policy

from dp import value_iteration, value_prediction
from monte_carlo import off_policy_mc_prediction_ordinary_importance_sampling as mc_ois
from monte_carlo import off_policy_mc_prediction_weighted_importance_sampling as mc_wis
from n_step_bootstrap import off_policy_n_step_sarsa as nsarsa
from n_step_bootstrap import on_policy_n_step_td as ntd

class RandomPolicy(Policy):
    def __init__(self,nA,p=None):
        self.p = p if p is not None else np.array([1/nA]*nA)

    def action_prob(self,state,action=None):
        return self.p[action]

    def action(self,state):
        return np.random.choice(len(self.p), p=self.p)

if __name__ == "__main__":
    class OneStateMDP(Env): # MDP introduced at Fig 5.4 in Sutton Book
        def __init__(self):
            final_state = 1
            env_spec=EnvSpec(2,2,1.,[final_state])

            super().__init__(env_spec)
            self.final_state = final_state
            self.trans_mat, self.r_mat = self._build_trans_mat()

        def _build_trans_mat(self):
            trans_mat = np.zeros((2,2,2))

            trans_mat[0,0,0] = 0.9
            trans_mat[0,0,1] = 0.1
            trans_mat[0,1,0] = 0.
            trans_mat[0,1,1] = 1.0
            trans_mat[1,:,1] = 1.

            r_mat = np.zeros((2,2,2))
            r_mat[0,0,1] = 1.

            return trans_mat, r_mat

        def reset(self):
            self._state = 0
            return self._state

        def step(self, action):
            assert action in list(range(self.spec.nA)), "Invalid Action"
            assert self._state != self.final_state, "Episode has ended!"

            prev_state = self._state
            self._state = np.random.choice(self.spec.nS,p=self.trans_mat[self._state,action])
            r = self.r_mat[prev_state,action,self._state]

            if self._state == self.final_state:
                return self._state, r, True
            else:
                return self._state, r, False

    class OneStateMDPWithModel(OneStateMDP,EnvWithModel):
        @property
        def TD(self) -> np.array:
            return self.trans_mat

        @property
        def R(self) -> np.array:
            return self.r_mat

    env = OneStateMDP()
    env_with_model = OneStateMDPWithModel()

    # Test Value Iteration
    V_star, pi_star = value_iteration(env_with_model,np.zeros(env_with_model.spec.nS),1e-4)

    assert np.allclose(V_star,np.array([1.,0.]),1e-5,1e-2), V_star
    assert pi_star.action(0) == 0

    eval_policy = pi_star
    behavior_policy = RandomPolicy(env.spec.nA)

    # Test Value Prediction
    V, Q = value_prediction(env_with_model,eval_policy,np.zeros(env.spec.nS),1e-4)
    assert np.allclose(V,np.array([1.,0.]),1e-5,1e-2), V
    assert np.allclose(Q,np.array([[1.,0.],[0.,0.]]),1e-5,1e-2), Q

    V, Q = value_prediction(env_with_model,behavior_policy,np.zeros(env.spec.nS),1e-4)
    assert np.allclose(V,np.array([0.1,0.]),1e-5,1e-2), V
    assert np.allclose(Q,np.array([[0.19,0.],[0.,0.]]),1e-5,1e-2), Q

    # Gather experience using behavior policy
    N_EPISODES = 100000

    trajs = []
    for _ in tqdm(range(N_EPISODES)):
        states, actions, rewards, done =\
            [env.reset()], [], [], []

        while not done:
            a = behavior_policy.action(states[-1])
            s, r, done = env.step(a)

            states.append(s)
            actions.append(a)
            rewards.append(r)

        traj = list(zip(states[:-1],actions,rewards,states[1:]))
        trajs.append(traj)

    # On-poilicy evaluation test
    Q_est_ois = mc_ois(env.spec,trajs,behavior_policy,behavior_policy,np.zeros((env.spec.nS,env.spec.nA)))
    Q_est_wis = mc_wis(env.spec,trajs,behavior_policy,behavior_policy,np.zeros((env.spec.nS,env.spec.nA)))
    V_est_td = ntd(env.spec,trajs,1,0.005,np.zeros((env.spec.nS)))

    assert np.allclose(Q_est_ois,np.array([[0.19,0.],[0.,0.]]),1e-5,1e-1), 'due to stochasticity, this test might fail'
    assert np.allclose(Q_est_wis,np.array([[0.19,0.],[0.,0.]]),1e-5,1e-1), 'due to stochasticity, this test might fail'
    assert np.allclose(Q_est_ois,Q_est_wis), 'Both implementation should be equal in on policy case'
    assert np.allclose(V_est_td,np.array([0.1,0.]),1e-5,1e-1), 'due to stochasticity, this test might fail'

    # Off-policy evaluation test
    Q_est_ois = mc_ois(env.spec,trajs,behavior_policy,eval_policy,np.zeros((env.spec.nS,env.spec.nA)))
    Q_est_wis = mc_wis(env.spec,trajs,behavior_policy,eval_policy,np.zeros((env.spec.nS,env.spec.nA)))

    # Don't panic even though Q_est_ois shows high estimation error. It's expected one!
    print(Q_est_ois)
    print(Q_est_wis)

    # Off-policy SARSA test
    Q_star_est, pi_star_est = nsarsa(env.spec,trajs,behavior_policy,n=1,alpha=0.005,initQ=np.zeros((env.spec.nS,env.spec.nA)))
    assert pi_star_est.action(0) == 0

    # sarsa also could fail to converge because of the similar reason above.
    print(Q_star_est)
