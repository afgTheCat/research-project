import nest
import nest.voltage_trace
import matplotlib.pyplot as plt
import re

DATA = {"test_data": "../data/ae.test", "train_data": "../data/ae.train"}


def parse_test_data():
    test_data = DATA["test_data"]
    lines = open(test_data, "r")
    clusters = []
    current_clustter = []
    for line in lines:
        print(line)
        if line == "":
            print("ehhhh")


def parse_train_data():
    test_data = DATA["train_data"]
    lines = open(test_data, "r")
    clusters = []
    current_cluster = []
    current_utterance = []
    for line in lines:
        lpc_coeffs = [float(x) for x in re.findall("([+-]?[0-9].[0-9]*)", line)]
        if len(lpc_coeffs) == 12:
            current_utterance.append(lpc_coeffs)
        else:
            current_cluster.append(current_utterance)
            current_utterance = []
            if len(current_cluster) == 30:
                clusters.append(current_cluster)
                current_cluster = []


def create_population():
    ndict = {"I_e": 200.0, "tau_m": 20.0}
    neuronpop = nest.Create(
        "iaf_psc_alpha", 100, conn_spec="pairwise_bernoulli", p=0.3, params=ndict
    )
    spike_generator_params = {"spike_times": [i * 10 for i in range(12)]}
    spike_generator = nest.Create("spike_generator", params=spike_generator_params)


if __name__ == "__main__":
    parse_train_data()
