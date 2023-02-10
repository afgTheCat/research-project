import resframe
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPClassifier
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


class RCModelHeterogenous:
    def __init__(
        self,
        number_of_neurons,
        dt,
        representation="last",
        readout="lin",
        w_ridge=5,
        w_ridge_embedding=10.0,
        input_delay=0.0,
        input_bias=0,
        input_scale=10,
        n_drop=0,
    ):
        self.reservoire = resframe.HeterogenousReservoire(number_of_neurons, dt)

        # representation
        match representation:
            case "last":
                self.representation = representation
            case "output" | "reservoire" | "reservoire_adjusted":
                self.representation = representation
                self._ridge_embedding = Ridge(
                    alpha=w_ridge_embedding, fit_intercept=True
                )
            case other:
                raise RuntimeError(
                    f"representation {representation} is not implemented"
                )

        self.n_drop = n_drop
        self.bias = input_bias
        self.input_delay = input_delay
        self.readout_type = readout
        self.scale = input_scale
        self.dt = dt
        # readout
        self.readout_type = readout
        match readout:
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

    def create_reservoire_input(self, X):
        N, T, _ = X.shape
        inputs = [[] for _ in range(N)]
        for t in range(T):
            current_input = X[:, t, :]
            for i, input_at_time in enumerate(current_input):
                inputs[i].append(
                    (self.input_delay, self.bias + input_at_time * self.scale)
                )

        return [resframe.InputStepsHomogenous(run_input) for run_input in inputs]

    def reservoire_states_with_times(self, res_inputs):
        all_states = []
        for i, inp in enumerate(res_inputs):
            states = self.reservoire.get_states(
                inp
            )  # [N, States] but the states may have different lenghts
            all_states.append(states)
        return all_states

    def reservoire_states(self, res_inputs):
        all_states = []
        t = []
        for i, inp in enumerate(res_inputs):
            t, run_states = self.reservoire.get_states(
                inp
            )  # [N, States] but the states may have different lenghts
            run_states = np.array(run_states).transpose()
            all_states.append(run_states)  # run_states
        self.times = t
        return (t, np.array(all_states))

    def train_readout(self, representation, Y):
        match self.readout_type:
            case "lin":
                self.readout.fit(representation, Y)
            case "mlp":
                self.readout.fit(representation, Y)

    def _state_repr(self, red_states, X):
        match self.representation:
            case "last":
                return red_states[:, -1, :]
            case "output":
                coeff_tr = []
                biases_tr = []
                current_state = red_states[0, 0:-1, :]
                states_to_skip = int(self.input_delay / self.dt)

                for run in range(X.shape[0]):
                    current_state = red_states[
                        run,
                        0:-states_to_skip:states_to_skip,
                        :,
                    ]
                    next_inputs = X[run][self.n_drop + 1 :, :]
                    self._ridge_embedding.fit(current_state, next_inputs)
                    coeff_tr.append(self._ridge_embedding.coef_.ravel())
                    biases_tr.append(self._ridge_embedding.intercept_.ravel())

                return np.concatenate(
                    (np.vstack(coeff_tr), np.vstack(biases_tr)), axis=1
                )

            case "reservoire":
                coeff_tr = []
                biases_tr = []
                states_to_skip = int(self.input_delay / self.dt)

                for run in range(X.shape[0]):
                    self._ridge_embedding.fit(
                        red_states[run, 0:-states_to_skip:states_to_skip, :],
                        red_states[run, states_to_skip::states_to_skip, :],
                    )
                    coeff_tr.append(self._ridge_embedding.coef_.ravel())
                    biases_tr.append(self._ridge_embedding.intercept_.ravel())

                return np.concatenate(
                    (np.vstack(coeff_tr), np.vstack(biases_tr)), axis=1
                )
            case "reservoire_adjusted":
                coeff_tr = []
                biases_tr = []
                input_len = int(self.input_delay)
                for run in range(X.shape[0]):
                    self._ridge_embedding.fit(
                        red_states[0, 0:-input_len:input_len, :],
                        red_states[0, input_len::input_len, :],
                    )
                    coeff_tr.append(self._ridge_embedding.coef_.ravel())
                    biases_tr.append(self._ridge_embedding.intercept_.ravel())

                return np.concatenate(
                    (np.vstack(coeff_tr), np.vstack(biases_tr)), axis=1
                )

            case other:
                raise RuntimeError(
                    f"representation: {self.readout_type} not implemented"
                )

    def train(self, Xtrain, Ytrain):
        # Gather all the states
        reservoire_inputs = self.create_reservoire_input(Xtrain)

        _, all_states = self.reservoire_states(reservoire_inputs)

        # TODO: dimensionality reduction

        # Represent the data
        representation = self._state_repr(all_states, Xtrain)

        # Set the readout
        self.train_readout(representation, Ytrain)

    def _predict(self, representation):
        match self.readout_type:
            case "lin":
                logits = self.readout.predict(representation)
                return np.argmax(logits, axis=1)
            case "mlp":
                pred_class = self.readout.predict(representation)
                return np.argmax(pred_class, axis=1)
            case readout:
                raise RuntimeError(f"readout {readout} not implemented")

    def _inspect_preditions(self, pred_class, Y):
        pass

    def test(self, Xtest, Ytest):
        reservoire_inputs = self.create_reservoire_input(Xtest)

        _, all_states = self.reservoire_states(reservoire_inputs)
        representation = self._state_repr(all_states, Xtest)

        pred_class = self._predict(representation)
        self._inspect_preditions(pred_class, Ytest)

        accuracy, f1 = compute_test_scores(pred_class, Ytest)

        return (accuracy, f1)
