import resframe
import scipy.io
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from resframe import (
    InputPrimitive,
    ThalmicPrimitive,
    NetworkInitPrimitive,
    ConnectivityPrimitive,
)
import logging
import numpy as np
from typing import Tuple


def create_input_steps(X):
    inputs = []
    for t in X:
        greater_then_delta = any(t > 0.1)
        if greater_then_delta:
            inputs.append((25.0, 1 + t * 10.0))
    return resframe.InputStepsHomogenous(inputs)


def create_all_inputs(X):
    N, T, _ = X.shape
    inputs = [[] for _ in range(N)]
    for t in range(T):
        current_input = X[:, t, :]
        for i, input_at_time in enumerate(current_input):
            inputs[i].append((25.0, input_at_time))
    return [resframe.InputStepsHomogenous(run_input) for run_input in inputs]


def inspect_neuron(res_output: Tuple[float, np.ndarray], number_of_neurons: int):
    plt.figure()
    time, neuron_list = res_output
    for n in range(number_of_neurons):
        neuron_vals = neuron_list[n]
        plt.plot(time, neuron_vals, label="Membrane Potential")[0]
    plt.show()


def test_training(thalmic_mean=0):
    data = scipy.io.loadmat("./data/JpVow.mat")
    onehot_encoder = OneHotEncoder(sparse=False)

    rc_model = resframe.RCModelHomogenous(
        dt=0.0005,
        a=0.2,
        b=2,
        c=-56,
        d=-16,
        number_of_neurons=10,
        erdos_connectivity=1,
        connectivity_primitive=resframe.ConnectivityPrimitive.ErdosUniform,
        erdos_uniform_lower=0,
        erdos_uniform_upper=2,
        input_primitive=InputPrimitive.PercentageConnected,
        input_connectivity_p=1,
        representation="output",
        readout="mlp",
        input_scale=10,
        input_bias=0,
        thalmic_primitive=ThalmicPrimitive.Const,
        thalmic_mean=thalmic_mean * 10,
    )

    Xtrain = data["X"]
    Ytrain = data["Y"]
    Xtest = data["Xte"]
    Ytest = data["Yte"]

    onehot_encoder = OneHotEncoder(sparse=False)
    Ytrain = onehot_encoder.fit_transform(Ytrain)
    Ytest = onehot_encoder.transform(Ytest)

    rc_model.train(Xtrain, Ytrain)
    accuracy, f1 = rc_model.test(Xtest, Ytest)
    print(f"Accuracy = {accuracy:.3f}, F1 = {f1:.3f}, thalmic mean = {thalmic_mean}")
    return accuracy


def neuron_visualize():
    data = scipy.io.loadmat("./data/JpVow.mat")
    Xtrain = data["X"][0:1]
    number_of_neurons = 100
    rc_model = resframe.RCModelHomogenous(
        dt=0.5,
        a=0.2,
        b=2.0,
        c=-56.0,
        d=-16.0,
        number_of_neurons=number_of_neurons,
        erdos_connectivity=1,
        connectivity_primitive=resframe.ConnectivityPrimitive.ErdosUniform,
        erdos_uniform_lower=0,
        erdos_uniform_upper=2,
        input_primitive=InputPrimitive.PercentageConnected,
        input_connectivity_p=0.5,
        representation="output",
        input_scale=10,
        input_bias=0,
        thalmic_primitive=ThalmicPrimitive.Const,
        thalmic_mean=-99,
    )
    inputs = rc_model.create_reservoire_input(Xtrain)
    t, states = rc_model.reservoire_states(inputs)
    states = states[0].transpose()
    for n in range(2):
        neuron_vals = states[n]
        plt.plot(t, neuron_vals, label="Membrane Potential")[0]
    plt.show()


def homogenous_with_const_input():
    rc_model = resframe.RCModelHomogenous(
        a=0.02,
        b=0.2,
        c=-65,
        d=8,
        dt=0.05,
        input_primitive=InputPrimitive.PercentageConnected,
        input_connectivity_p=1,
        number_of_neurons=1,
        thalmic_primitive=ThalmicPrimitive.Const,
        thalmic_mean=0,
        network_init_primitive=NetworkInitPrimitive.NoRandomWeight,
        connectivity_primitive=ConnectivityPrimitive.ErdosUniform,
        erdos_uniform_lower=1.0,
        erdos_uniform_upper=1.0,
    )
    inp = resframe.InputStepsHomogenous([(1000, [0])])
    output = rc_model.reservoire_states([inp])
    inspect_neuron(output, 1)


def heterogenous_with_const_input():
    rc_model = resframe.RCModelHeterogenous(1, 0.05)
    inp = resframe.InputStepsHeterogenous([(1000, [10])])
    output = rc_model.reservoire_states([inp])
    inspect_neuron(output, 1)


if __name__ == "__main__":
    FORMAT = "%(levelname)s %(name)s %(asctime)-15s %(filename)s:%(lineno)d %(message)s"
    logging.basicConfig(format=FORMAT)
    logging.getLogger().setLevel(logging.INFO)
    heterogenous_with_const_input()
    # homogenous_with_const_input()
    # neuron_visualize()
    # test_training()
