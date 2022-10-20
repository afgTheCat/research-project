import resframe
import scipy.io
from sklearn.preprocessing import OneHotEncoder
from resframe import InputPrimitive
import numpy as np
from multiprocessing import Pool


def search_for_param(
    inp,
):
    Xtrain = inp["Xtrain"]
    Xtest = inp["Xtest"]
    Ytrain = inp["Ytrain"]
    Ytest = inp["Ytest"]
    erdos_connectivity = inp["erdos_connectivity"]
    uniform_lower = inp["uniform_lower"]
    uniform_upper = inp["uniform_upper"]
    input_connectivity_p = inp["input_connectivity_p"]
    representation = inp["representation"]
    input_scale = inp["input_scale"]
    input_bias = inp["input_bias"]
    rc_model = resframe.RCModel(
        dt=1,
        number_of_neurons=100,
        erdos_connectivity=erdos_connectivity,
        connectivity_primitive=resframe.ConnectivityPrimitive.ErdosUniform,
        erdos_uniform_lower=uniform_lower,
        erdos_uniform_upper=uniform_upper,
        input_primitive=InputPrimitive.PercentageConnected,
        input_connectivity_p=input_connectivity_p,
        representation=representation,
        input_scale=input_scale,
        input_bias=input_bias,
    )
    rc_model.train(Xtrain, Ytrain)
    accuracy, f1 = rc_model.test(Xtest, Ytest)
    print(
        "Accuracy = %.3f, F1 = %.3f, erdos conn: %.3f, uniform lower: %.3f input connectivity: %.3f, representation: %s input scale: %.3f input bias: %.3f"
        % (
            accuracy,
            f1,
            erdos_connectivity,
            uniform_lower,
            input_connectivity_p,
            representation,
            input_scale,
            input_bias,
        )
    )


if __name__ == "__main__":
    data = scipy.io.loadmat("./data/JpVow.mat")
    onehot_encoder = OneHotEncoder(sparse=False)

    Xtrain = data["X"]
    Ytrain = data["Y"]
    Xtest = data["Xte"]
    Ytest = data["Yte"]

    Ytrain = onehot_encoder.fit_transform(Ytrain)
    Ytest = onehot_encoder.transform(Ytest)

    erdos = np.arange(0.2, 1, 0.1)
    input_connectivity_p = np.arange(0.1, 1, 0.1)
    representation = ["last", "output"]
    input_scale = np.arange(7.5, 12.5, 0.5)
    input_bias = np.arange(0, 2, 0.2)
    uniform_lower = np.arange(0, 2, 0.2)
    uniform_upper = uniform_lower + 1
    uniform_zip = zip(uniform_lower, uniform_upper)

    params = [
        {
            "Xtrain": Xtrain,
            "Ytrain": Ytrain,
            "Xtest": Xtest,
            "Ytest": Ytest,
            "erdos_connectivity": e,
            "uniform_lower": ul,
            "uniform_upper": up,
            "input_connectivity_p": ic,
            "representation": r,
            "input_scale": s,
            "input_bias": ib,
        }
        for e in erdos
        for (ul, up) in uniform_zip
        for ic in input_connectivity_p
        for r in representation
        for s in input_scale
        for ib in input_bias
    ]

    with Pool() as p:
        p.map(search_for_param, params)

    # pool.close()
