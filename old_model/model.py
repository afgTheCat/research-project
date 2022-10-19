from reservoire import Reservoire
from sklearn.linear_model import Ridge
import scipy.io
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier


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
        n_internal_units=100,
        connectivity=1,
        izhikevich_params=[
            0.02,
            0.2,
            -65,
            8,
        ],
        internal_weights=None,
        w_ridge=5,
        readout="lin",
    ):
        a, b, reset_val, d = izhikevich_params
        self._reservoir = Reservoire(
            a=a,
            b=b,
            reset_val=reset_val,
            d=d,
            n_internal_units=n_internal_units,
            internal_weights=internal_weights,
            connectivity=connectivity,
        )
        if readout == "lin":
            self.readout = Ridge(alpha=w_ridge)
        elif readout == "mlp":
            self.readout = self.readout = MLPClassifier(
                hidden_layer_sizes=(20, 10),  # this can change
                activation="relu",
                # alpha=w_l2,
                # batch_size=32,
                learning_rate="adaptive",  # 'constant' or 'adaptive'
                learning_rate_init=0.001,
                # max_iter=num_epochs,
                early_stopping=False,  # if True, set validation_fraction > 0
                validation_fraction=0.0,  # used for early early_stopping
            )

    def repr_state(self, res_states):
        return res_states[:, -1, :]

    def train(self, Xtrain, Y):
        res_states = self._reservoir.get_states(Xtrain)

        # last state representation
        input_repr = self.repr_state(res_states)

        # fitting a linear model on both parameters is probably a bad idea, oh well
        self.readout.fit(input_repr, Y)

    def test(self, Xtest, Yte):
        res_states = self._reservoir.get_states(Xtest)

        # last state representation
        input_repr_te = self.repr_state(res_states)

        # Here lies the problem
        logits = self.readout.predict(input_repr_te)
        pred_class = np.argmax(logits, axis=1)
        accuracy, f1 = compute_test_scores(pred_class, Yte)
        return (accuracy, f1)


if __name__ == "__main__":
    data = scipy.io.loadmat("../data/JpVow.mat")

    Xtrain = data["X"]
    Ytrain = data["Y"]

    Xtest = data["Xte"]
    Ytest = data["Yte"]

    onehot_encoder = OneHotEncoder(sparse=False)
    Ytrain = onehot_encoder.fit_transform(Ytrain)
    Ytest = onehot_encoder.transform(Ytest)

    rc_model = RCModel(n_internal_units=20)
    rc_model.train(Xtrain, Ytrain)
    accuracy, f1 = rc_model.test(Xtest, Ytest)

    print("Accuracy = %.3f, F1 = %.3f" % (accuracy, f1))
