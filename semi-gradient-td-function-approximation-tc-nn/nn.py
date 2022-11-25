import numpy as np
from algo import ValueFunctionWithApproximation

import tensorflow as tf

class ValueFunctionWithNN(ValueFunctionWithApproximation):
    def __init__(self,
                 state_dims):
        """
        state_dims: the number of dimensions of state space
        """
        # TODO: implement this method

    def __call__(self,s):
        # TODO: implement this method
        return 0.

    def update(self,alpha,G,s_tau):
        # TODO: implement this method
        return None

