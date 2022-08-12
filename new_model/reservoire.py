from scipy import sparse
import numpy as np
import scipy.io


class Reservoire:
    def __init__(
        self,
        n_internal_units=100,
        connectivity=1,
        coupling_coefficient=0.3,
        reset_val=-65,
        a=0.02,
        b=0.2,
        d=8,
        dt=0.5,
    ) -> None:
        self.n_internal_units = n_internal_units
        self.connectivity = connectivity
        self.coupling_coefficient = coupling_coefficient
        self.reset_val = reset_val
        self.a = a
        self.b = b
        self.d = d
        self.dt = dt
        self._initialize_internal_weights()
        self._initialize_connection()
        self.internal_v = [[] for _ in range(n_internal_units)]
        self.internal_u = [[] for _ in range(n_internal_units)]
        self.spike_value = 30

    def _initialize_internal_weights(self):
        n_internal_units = self.n_internal_units
        connectivity = self.connectivity
        internal_weights = sparse.rand(
            n_internal_units, n_internal_units, density=connectivity
        ).todense()
        self.internal_weights = internal_weights

    def _initialize_connection(self):
        def single_neuron_connection(i):
            return sum(
                self.internal_weights[j, i]
                for j in range(self.n_internal_units)
                if i != j
            )

        self.connections = [
            single_neuron_connection(i) for i in range(self.n_internal_units)
        ]

    def simulate_izhikevich_one_step(self, neuron, t, I):
        v = self.internal_v[neuron]
        u = self.internal_u[neuron]
        if v[-1] < self.spike_value:
            dV = (0.04 * v[-1] + 5) * v[-1] + 140 - u[-1]
            v.append(v[-1] + (dV + I) * self.dt)
            du = self.a * (self.b * v[-1] - u[-1])
            u.append(u[t - 1] + self.dt * du)
        else:
            v[-1] = self.spike_value
            v.append(self.reset_val)

    # spike for all the things
    def simulate_izhikevich_network(self, time, state_matrix, I):
        # T = 1000  # total simulation length [ms]
        time = np.arange(0, time + self.dt, self.dt)  # step values [ms]

        for neuron in range(n_internal_units):
            print("lll")

        # for t in range(1, len(time)):
        #     if V[t - 1] < self.spike_value:
        #         dV = (0.04 * V[t - 1] + 5) * V[t - 1] + 140 - u[t - 1]
        #         V[t] = V[t - 1] + (dV + I[t - 1]) * dt
        #         du = self.a * (self.b * V[t - 1] - u[t - 1])
        #         u[t] = u[t - 1] + dt * du
        #     else:
        #         V[t - 1] = self.spike_value  # set to spike value
        #         V[t] = self.reset_val  # reset membrane voltage
        #         u[t] = u[t - 1] + self.d  # reset recovery

    def _compute_state_matrix(
        self,
        X,
    ):
        N, T, _ = X.shape
        # Storage
        state_matrix = np.empty((N, T, self.n_internal_units), dtype=float)
        for t in range(T):
            current_input = X[:, t, :]
            print(current_input)
        return state_matrix

    def get_states(self, X):
        # Number of variables, time
        N, T, _ = X.shape
        for t in range(T):
            current_input = X[:, t, :]


if __name__ == "__main__":
    data = scipy.io.loadmat("../data/JpVow.mat")
    # shape is [N: number of series,T: length of the max series, V: number of variables]

    Xtr = data["X"]  # shape is [N,T,V]
    Ytr = data["Y"]  # shape is [N,1]
    Xte = data["Xte"]
    Yte = data["Yte"]
    res = Reservoire(n_internal_units=4)

    # res.get_states()
    print(res)
