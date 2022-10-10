from scipy import sparse
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import random


class Reservoire:
    def __init__(
        self,
        n_internal_units=100,
        connectivity=1,
        a=0.02,
        b=0.2,
        reset_val=-65,
        d=8,
        dt=0.5,
        internal_weights=None,
    ) -> None:
        self.n_internal_units = n_internal_units
        self.connectivity = connectivity
        self.a = a
        self.b = b
        self.reset_val = reset_val
        self.d = d
        self.dt = dt
        self._initialize_internal_weights(internal_weights)
        self.spike_value = 30

        # Input weights depend on input size: they are set when data is provided
        self._input_weights = None
        self.state_matrix = None

    def _initialize_internal_weights(self, internal_weights=None):
        if internal_weights is None:
            n_internal_units = self.n_internal_units
            connectivity = self.connectivity
            internal_weights = sparse.rand(
                n_internal_units, n_internal_units, density=connectivity
            ).todense()
            self.internal_weights = internal_weights
        else:
            self.internal_weights = internal_weights
            print(self.internal_weights)

    # TODO: maybe sim neuron should be reconsideered
    def sim_neuron(self, u, v, i, w):
        if v < self.spike_value:
            dv = 0.04 * v**2 + 5 * v + 140 - u + w
            du = self.a * (self.b + v - u)
            v_new = v + (dv + i) * self.dt
            if v_new > self.spike_value:
                return (self.reset_val, u + self.d)
            else:
                u_new = u + self.dt * du
                return (v_new, u_new)
        else:
            v_new = self.reset_val
            u_new = u + self.d
            return (v_new, u_new)

    def simulate_network(self, previous_state, input_matrix):
        N, T = previous_state.shape
        new_state_matrix = np.zeros((N, T), dtype=float)
        for n in range(N):
            for neuron in range(self.n_internal_units):
                v = previous_state[n][neuron]
                u = previous_state[n][self.n_internal_units + neuron]
                w = sum(
                    self.internal_weights[j, neuron] * (v - previous_state[n][j])
                    for j in range(self.n_internal_units)
                    if j != neuron
                )
                input_val = input_matrix[n][neuron]

                v_new, u_new = self.sim_neuron(u, v, input_val, w)
                new_state_matrix[n][neuron] = v_new
                new_state_matrix[n][self.n_internal_units + neuron] = u_new

        return new_state_matrix

    def get_states(self, X, input_weight="eq"):
        """
        The shape of X should be:
        * N: number of multivariate times series
        * T: lenght of the MVTS
        * V: number of variables
        Should return:
        * N: same as above
        * T: same as above
        * 2 * U: where U is the number of internal units. The dimensionality doubles since we care about both I and U
        """
        N, T, V = X.shape

        if self._input_weights is None and input_weight == "binom":
            self._input_weights = (
                2.0 * np.random.binomial(1, 0.5, [self.n_internal_units, V]) - 1.0
            )

        elif self._input_weights is None:
            self._input_weights = np.ones((self.n_internal_units, V))

        # initial state
        initial_state = [
            *[-70 + random.uniform(-10, 10) for _ in range(self.n_internal_units)],
            *[-14 + random.uniform(-3, 3) for _ in range(self.n_internal_units)],
        ]

        # State can be described with
        previous_state = np.array([initial_state for _ in range(N)])
        self.state_matrix = np.empty((N, T, 2 * self.n_internal_units), dtype=float)

        for t in range(T):
            current_input = X[:, t, :]  # [N, V]
            input_matrix = self._input_weights.dot(current_input.T).T
            previous_state = self.simulate_network(previous_state, input_matrix)
            self.state_matrix[:, t, :] = previous_state

        return self.state_matrix

    def plot_n_th(self, state_matrix, n_th, neuron, time):
        fig1 = plt.figure()
        for n in neuron:
            V = state_matrix[n_th][:, n]
            plt.plot(time, V, label="Membrane Potential")[0]
        plt.show()


if __name__ == "__main__":
    data = scipy.io.loadmat("../data/JpVow.mat")

    # Xtr = data["X"]  # shape is [N,T,V]
    n_internal_units = 2
    steps = 1000
    dt = 0.5
    time_points = 1000
    Xtr = 10 * np.ones((1, time_points, 1))
    internal_weights = np.array([[0, 0.1], [-0.1, 0]])

    # internal_weights = None
    reservoire = Reservoire(
        n_internal_units=n_internal_units,
        dt=dt,
        internal_weights=internal_weights,
    )

    state_matrix = reservoire.get_states(
        Xtr,
        input_weight="eq",
    )

    time = np.arange(0, time_points * dt, dt)  # step values [ms]
    reservoire.plot_n_th(state_matrix, 0, [0, 1], time)
