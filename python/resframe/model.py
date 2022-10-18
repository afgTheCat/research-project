import resframe
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPClassifier
from resframe import NetworkInitPrimitive, ConnectivityPrimitive
import numpy as np


class RCModel:
    def __init__(self, readout="lin", w_ridge=5, dt=0.05, number_of_neurons=20) -> None:
        # or I can choose something else!
        variant_chooser = resframe.VariantChooser(
            network_init_primitive=NetworkInitPrimitive.NormalRandomWeight,
            connectivity_primitive=ConnectivityPrimitive.ErdosUniform,
            network_membrane_potential_dev=10.0,
            network_recovery_variable=3.0,
            erdos_connectivity=0.25,
            erdos_normal_mean=0.0,
        )
        self.reservoire = resframe.Reservoire(
            variant_chooser=variant_chooser, dt=dt, number_of_neurons=20
        )
        if readout == "lin":
            self.readout = Ridge(alpha=w_ridge)
        elif readout == "mlp":
            self.readout = self.readout = MLPClassifier(
                hidden_layer_sizes=(20, 10),  # this can change
                activation="relu",
                learning_rate="adaptive",  # 'constant' or 'adaptive'
                learning_rate_init=0.001,
                early_stopping=False,  # if True, set validation_fraction > 0
                validation_fraction=0.0,  # used for early early_stopping
            )

    # last state prep
    def train(self, res_inputs, Y):
        all_states = []
        input_len = len(res_inputs)
        for i, inp in enumerate(res_inputs):
            print(f"processing {i+1} of {input_len}")
            state = self.reservoire.get_states(inp)
            print(state)
            if state is not None:
                all_states.append(state)

        last_states = []
        for t_states in all_states:
            print(t_states)
            _, states = t_states
            np.array(last_states.append(states[-1]))
        last_states = np.array(last_states)
        print(last_states.shape)
        # print("this thing!")

    def reservoire_states(self, input_steps):
        return self.reservoire.get_states(input_steps)

    def test_reservoire(self) -> None:
        input_steps = resframe.InputSteps([(1000.0, [10])])
        test_run = self.reservoire.get_states(input_steps)

        if test_run is not None:
            time, neurons = test_run
            plt.plot(time, neurons[0], label="Membrane Potential")[0]
            plt.show()
