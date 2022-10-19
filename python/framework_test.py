import resframe
import scipy.io
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder


def create_train_data(X):
    inputs = []
    for t in X:
        greater_then_delta = any(t > 0.1)
        if greater_then_delta:
            inputs.append((25.0, t * 100.0))
    return resframe.InputSteps(inputs)


def inspect_neuron(res_output, number_of_neurons):
    plt.figure()
    time, neuron_list = res_output
    for n in range(number_of_neurons):
        neuron_vals = neuron_list[n]
        plt.plot(time, neuron_vals, label="Membrane Potential")[0]
    plt.show()


def testing_model():
    data = scipy.io.loadmat("./data/JpVow.mat")
    onehot_encoder = OneHotEncoder(sparse=False)

    Xtrain = data["X"]
    Ytrain = data["Y"]

    Xtest = data["Xte"]
    Ytest = data["Yte"]

    Xtrain = [create_train_data(x) for x in Xtrain]
    Xtest = [create_train_data(x) for x in Xtest]

    Ytrain = onehot_encoder.fit_transform(Ytrain)
    Ytest = onehot_encoder.transform(Ytest)

    rc_model = resframe.RCModel(dt=0.05, number_of_neurons=100)

    neuron_states = rc_model.train(Xtrain, Ytrain)
    accuracy, f1 = rc_model.test(Xtest, Ytest)

    print("Accuracy = %.3f, F1 = %.3f" % (accuracy, f1))


if __name__ == "__main__":
    data = scipy.io.loadmat("./data/JpVow.mat")
    onehot_encoder = OneHotEncoder(sparse=False)

    Xtrain = data["X"]
    Xtrain = [create_train_data(x) for x in Xtrain]
    number_of_neurons = 100

    rc_model = resframe.RCModel(
        dt=0.05,
        number_of_neurons=number_of_neurons,
        erdos_connectivity=1,
        connectivity_primitive=resframe.ConnectivityPrimitive.ErdosNormal,
        erdos_normal_dev=10,
        erdos_normal_mean=0,
    )
    first_run = rc_model.reservoire_states_with_times(Xtrain[0:1])
    inspect_neuron(first_run[0], number_of_neurons)
