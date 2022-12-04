import numpy as np
from algo import ValueFunctionWithApproximation

class ValueFunctionWithTile(ValueFunctionWithApproximation):
    def __init__(self,
                 state_low:np.array,
                 state_high:np.array,
                 num_tilings:int,
                 tile_width:np.array):
        """
        Tiling value function approximation using symmetric tiling

        state_low: possible minimum value for each dimension in state
        state_high: possible maximum value for each dimension in state
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        self.state_low = state_low
        self.state_high = state_high
        self.num_tilings = num_tilings
        self.tile_width = tile_width

        self.tiling_dims = (np.ceil((state_high - state_low)/tile_width) + np.ones(len(tile_width))).astype(int)
        tiling_offsets = np.outer(np.arange(num_tilings, dtype=float)/num_tilings, tile_width)
        self.tiling_starts = np.tile(state_low, (num_tilings, 1)) - tiling_offsets
        self.weights = np.zeros((num_tilings, *self.tiling_dims))

    def __call__(self,s) -> float:
        """ Return value function for state s"""
        x = self.x(s)
        return self.value(x)

    def update(self,alpha,G,s_tau) -> None:
        """ Update value function via gradient towards G on state s_tau, step scaled by alpha"""
        x_tau = self.x(s_tau)
        self.weights[tuple(x_tau.T)] += alpha * (G - self.value(x_tau))
        return None

    def x(self, s) -> np.array:
        """ x(s), the input feature of state s, in a dense (index) form """
        state_tilings = np.tile(s, (self.num_tilings, 1))
        state_indexes = np.floor((state_tilings - self.tiling_starts)/self.tile_width).astype(int)
        return np.concatenate((np.arange(self.num_tilings).reshape((self.num_tilings, 1)), state_indexes), axis=1)
    
    def value(self, x) -> float:
        return np.sum(self.weights[tuple(x.T)])

if  __name__ == "__main__":
    V = ValueFunctionWithTile(
        np.array([-1.5, -0.5]),
        np.array([1.5, 0.5]),
        2,
        np.array([0.7, 0.3])
    )
    assert np.allclose(V.tiling_dims, np.array([6, 5])), V.tiling_dims
    assert np.allclose(V.tiling_starts, np.array([[-1.5, -0.5], [-1.85, -0.65]])), V.tiling_dims
    assert V.weights.shape == (2, 6, 5)

    assert np.allclose(V.x([-1.5, -0.5]), np.array([[0, 0, 0], [1, 0, 0]]))
    assert np.allclose(V.x([-1.5, -0.3]), np.array([[0, 0, 0], [1, 0, 1]]))
    assert np.allclose(V.x([-1.0, -0.2]), np.array([[0, 0, 1], [1, 1, 1]]))
    assert np.allclose(V.x([1.5, 0.5]), np.array([[0, 4, 3], [1, 4, 3]]))

    assert V([-1.5, 0.5]) == 0.

    V.weights = np.arange(V.weights.size).reshape(V.weights.shape)
    assert V([-1.5, -0.5]) == 30 # 0 + 30
    assert V([-1.5, 0.5]) == 36 # 3 + 33
    assert V([-1.0, -0.2]) == 37 # 1 + 36

    V.weights = np.zeros(V.weights.size).reshape(V.weights.shape)

    V.update(0.1, 1, [-1.5, -0.5])
    assert V([-1.5, -0.5]) == 0.2
    assert V([-1.5, -0.3]) == 0.1