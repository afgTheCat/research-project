from os import read
from typing import List
import resframe
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPClassifier
from resframe import (
    InputPrimitive,
    InputSteps,
    NetworkInitPrimitive,
    ConnectivityPrimitive,
)
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def compute_test_scores(pred_class, Yte):
    """
    Wrapper to compute classification accuracy and F1 score
    """
    true_class = np.argmax(Yte, axis=1)
    accuracy = accuracy_score(true_class, pred_class)
    if Yte.shape[1] > 2:
        f1 = f1_score(true_class, pred_class, average="weighted")
    else:
        f1 = f1_score(true_class, pred_class, average="binary")
    return accuracy, f1


class RCModel:
    def __init__(
        self,
        readout_type="lin",
        representation="last",
        w_ridge=5,
        w_ridge_embedding=10.0,
        dt=0.05,
        number_of_neurons=20,
        network_init_primitive=NetworkInitPrimitive.NormalRandomWeight,
        network_membrane_potential=-65.0,
        network_membrane_potential_dev=10.0,
        network_recovery_variable=-14.0,
        network_recovery_variable_dev=3.0,
        connectivity_primitive=ConnectivityPrimitive.ErdosUniform,
        erdos_connectivity=0.25,
        erdos_normal_dev=1.0,
        erdos_normal_mean=0.0,
        erdos_uniform_lower=0.0,
        erdos_uniform_upper=1.0,
        erdos_spectral_radius=0.59,
        input_primitive=InputPrimitive.AllConnected,
        input_connectivity_p=0.5,
        n_drop=5,
    ) -> None:
        variant_chooser = resframe.VariantChooser(
            network_init_primitive=network_init_primitive,
            connectivity_primitive=connectivity_primitive,
            input_primitive=input_primitive,
            network_membrane_potential=network_membrane_potential,
            network_membrane_potential_dev=network_membrane_potential_dev,
            network_recovery_variable=network_recovery_variable,
            network_recovery_variable_dev=network_recovery_variable_dev,
            erdos_connectivity=erdos_connectivity,
            erdos_normal_mean=erdos_normal_mean,
            erdos_normal_dev=erdos_normal_dev,
            erdos_uniform_lower=erdos_uniform_lower,
            erdos_uniform_upper=erdos_uniform_upper,
            erdos_spectral_radius=erdos_spectral_radius,
            input_connectivity_p=input_connectivity_p,
        )
        self.reservoire = resframe.Reservoire(
            dt=dt, number_of_neurons=number_of_neurons, variant_chooser=variant_chooser
        )

        # representation
        match representation:
            case "last":
                self.representation = representation
            case "output":
                self.representation = representation
                self._ridge_embedding = Ridge(
                    alpha=w_ridge_embedding, fit_intercept=True
                )

            case other:
                raise RuntimeError(
                    f"representation {representation} is not implemented"
                )

        # readout
        self.readout_type = readout_type
        match readout_type:
            case "lin":
                self.readout = Ridge(alpha=w_ridge)
            case "mlp":
                self.readout = MLPClassifier(
                    hidden_layer_sizes=(20, 10),  # this can change
                    activation="relu",
                    learning_rate="adaptive",  # 'constant' or 'adaptive'
                    learning_rate_init=0.001,
                    early_stopping=False,  # if True, set validation_fraction > 0
                    validation_fraction=0.0,  # used for early early_stopping
                )
            case other:
                raise RuntimeError(f"readout method {other} is not implemented")
        self.n_drop = n_drop

    def reservoire_states_with_times(self, res_inputs):
        all_states = []
        input_len = len(res_inputs)
        for i, inp in enumerate(res_inputs):
            print(f"processing {i+1} of {input_len}")
            states = self.reservoire.get_states(
                inp
            )  # [N, States] but the states may have different lenghts
            all_states.append(states)
        return all_states

    def reservoire_states(self, res_inputs):
        all_states = []
        input_len = len(res_inputs)
        for i, inp in enumerate(res_inputs):
            print(f"processing {i+1} of {input_len}")
            _, run_states = self.reservoire.get_states(
                inp
            )  # [N, States] but the states may have different lenghts
            all_states.append(run_states)
        return all_states

    def _state_repr(self, states, res_inputs: List[InputSteps]):
        match self.representation:
            case "last":
                last_states = [[neuron[-1] for neuron in run] for run in states]
                return np.array(last_states)
            # TODO: fix this!
            case "output":
                coeff_tr = []
                biases_tr = []

                all_but_last_state = []
                states_with_dropoff = []
                print(len(res_inputs[0].vals()))
                # print(len(states_with_dropoff[0]))
                # print(len(states_with_dropoff[0][0]))

                # self._ridge_embedding.fit(
                #     states[i, 0:-1, :], states[i, self.n_drop + 1 :, :]
                # )
                #     coeff_tr.append(self._ridge_embedding.coef_.ravel())
                #     biases_tr.append(self._ridge_embedding.intercept_.ravel())
                #
                # return np.concatenate(
                #     (np.vstack(coeff_tr), np.vstack(biases_tr)), axis=1
                # )
                last_states = [[neuron[-1] for neuron in run] for run in states]
                return np.array(last_states)
            case other:
                raise RuntimeError(
                    f"representation: {self.readout_type} not implemented"
                )

    def train_readout(self, representation, Y):
        match self.readout_type:
            case "lin":
                self.readout.fit(representation, Y)

    def train(self, res_inputs, Y):
        # Gather all the states
        all_states = self.reservoire_states(res_inputs)

        # TODO: dimensionality reduction

        # Represent the data
        representation = self._state_repr(all_states, res_inputs)

        # Set the readout
        self.train_readout(representation, Y)

    def _predict(self, representation):
        match self.readout_type:
            case "lin":
                logits = self.readout.predict(representation)
                return np.argmax(logits, axis=1)
            case readout:
                raise RuntimeError(f"readout {readout} not implemented")

    def test(self, res_inputs, Ytest):
        all_states = self.reservoire_states(res_inputs)
        representation = self._state_repr(all_states, res_inputs)
        print(representation)

        pred_class = self._predict(representation)
        print(pred_class)

        accuracy, f1 = compute_test_scores(pred_class, Ytest)

        return (accuracy, f1)
