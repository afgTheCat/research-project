import resframe
import scipy.io
import matplotlib.pyplot as plt


def create_train_data(X):
    inputs = []
    for t in X:
        greater_then_delta = any(t > 0.1)
        if greater_then_delta:
            inputs.append((25.0, t * 10.0))
    return resframe.InputSteps(inputs)


def inspect_neuron(res_output, n):
    time, neuron_list = res_output
    neuron_vals = neuron_list[n]
    fig1 = plt.figure()
    plt.plot(time, neuron_vals, label="Membrane Potential")[0]
    plt.show()


if __name__ == "__main__":
    data = scipy.io.loadmat("./data/JpVow.mat")

    # shape (270, 29, 12): (N, T, V)
    Xtrain = data["X"]
    Ytrain = data["Y"]

    Xtest = data["Xte"]
    Ytest = data["Yte"]

    res_inputs = [create_train_data(x) for x in Xtrain]
    rc_model = resframe.RCModel(dt=0.05)
    # rc_model.train(res_inputs[0:1], Ytrain[0:10])

    neuron_states = rc_model.reservoire_states(res_inputs[0])
    for n in range(10):
        print(neuron_states[0])
        inspect_neuron(neuron_states, n)
