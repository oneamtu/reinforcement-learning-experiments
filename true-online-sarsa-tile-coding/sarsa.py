import numpy as np
import time

class StateActionFeatureVectorWithTile():
    def __init__(self,
                 state_low:np.array,
                 state_high:np.array,
                 num_actions:int,
                 num_tilings:int,
                 tile_width:np.array):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maimum value for each dimension in state
        num_actions: the number of possible actions
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        self.state_low = state_low
        self.state_high = state_high
        self.num_actions = num_actions
        self.num_tilings = num_tilings
        self.num_tilings = num_tilings
        self.tile_width = tile_width

        self.tiling_dims = (np.ceil((state_high - state_low)/tile_width) + np.ones(len(tile_width))).astype(int)
        tiling_offsets = np.outer(np.arange(num_tilings, dtype=float)/num_tilings, tile_width)
        self.tiling_starts = np.tile(state_low, (num_tilings, 1)) - tiling_offsets

    def feature_vector_len(self) -> int:
        """
        return length of feature_vector: d = (num_actions * num_tilings * num_tiles)
        """
        return np.prod(self.feature_vector_dims())

    def feature_vector_dims(self) -> int:
        """
        return dimension of feature_vector: d = (num_actions, num_tilings, num_tiles)
        """
        return (self.num_actions, self.num_tilings, *self.tiling_dims)

    def __call__(self, s, done, a) -> np.array:
        """
        implement function x: S+ x A -> [0,1]^d
        if done is True, then return 0^d
        """
        if done:
            return np.zeros(self.feature_vector_len())
        else:
            state_tilings = np.tile(s, (self.num_tilings, 1))
            state_indexes = np.floor((state_tilings - self.tiling_starts)/self.tile_width).astype(int)
            x = np.zeros(self.feature_vector_dims())
            indexes = np.concatenate(
                (
                    np.repeat(a, self.num_tilings).reshape((self.num_tilings, 1)),
                    np.arange(self.num_tilings).reshape((self.num_tilings, 1)), 
                    state_indexes
                ),
                axis=1
            )
            x[tuple(indexes.T)] = 1
            return x.ravel()

def SarsaLambda(
    env, # openai gym environment
    gamma:float, # discount factor
    lam:float, # decay rate
    alpha:float, # step size
    X:StateActionFeatureVectorWithTile,
    num_episode:int,
) -> np.array:
    """
    Implement True online Sarsa(\lambda)
    """
    def epsilon_greedy_policy(s,done,w,epsilon=.0):
        nA = env.action_space.n
        Q = [np.dot(w, X(s,done,a)) for a in range(nA)]

        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Q)

    w = np.zeros(X.feature_vector_len())

    for episode in range(num_episode):
        start_time = time.perf_counter_ns()
        step_counter = 0
        total_error = 0
        total_reward = 0

        state = env.reset()
        done = False
        action = epsilon_greedy_policy(state, done, w)

        x = X(state, done, action)

        z = 0
        Q_old = 0

        while not done:
            next_state, reward, done, _ = env.step(action)
            next_action = epsilon_greedy_policy(next_state, done, w)
            # env.render()

            x_prime = X(next_state, done, next_action)
            Q = np.dot(w, x)
            Q_prime = np.dot(w, x_prime)

            error = reward + gamma * Q_prime - Q
            z = gamma*lam*z + (1 - alpha*gamma*lam*np.dot(z, x))*x
            w = w + alpha*(error + Q - Q_old)*z - alpha*(Q - Q_old)*x
            
            Q_old = Q_prime
            x = x_prime
            action = next_action

            step_counter += 1
            total_error += error
            total_reward += reward

        print(f'Episode: {episode} steps: {step_counter} '\
            f'reward {total_reward} total_error {total_error} '\
            f'in {(time.perf_counter_ns() - start_time)/1_000_000} ms')
    return w

if  __name__ == "__main__":
    X = StateActionFeatureVectorWithTile(
        np.array([-1.5, -0.5]),
        np.array([1.5, 0.5]),
        2,
        2,
        np.array([1.4, 0.6])
    )
    assert np.allclose(X.tiling_dims, np.array([4, 3])), X.tiling_dims
    assert np.allclose(X.tiling_starts, np.array([[-1.5, -0.5], [-2.2, -0.8]])), X.tiling_dims

    assert np.allclose(X([-1.5, -0.5], True, 0), np.zeros(X.feature_vector_len()))

    assert np.allclose(X([-1.5, -0.5], False, 0).reshape(X.feature_vector_dims()), np.array([
        [
            [
                [1, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            [
                [1, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]
        ],
        [
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]
        ]
    ]))