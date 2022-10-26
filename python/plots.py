import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import resframe
import itertools


def save_boxplot(last_state, output, reservoire):
    fig = plt.figure(dpi=80)
    box_plot_data = [last_state, output, reservoire]
    plt.boxplot(
        box_plot_data,
        patch_artist=True,
        labels=[
            "last state",
            "output",
            "reservoire",
        ],
    )

    plt.ylabel("Prediction accuracy")
    plt.title("Model prediction accuracy depending on the representation")
    plt.legend()

    pdf = PdfPages("models_and_repr.pdf")
    fig.savefig(pdf, format="pdf", bbox_inches="tight")
    pdf.close()
    plt.show()


def representation_plot():
    erdos_connectivity = [0.25, 0.50, 1]
    input_connectivity = [0.50]
    input_scale = [10.0]
    uniform_lower = [0]
    uniform_upper = [1]
    uniform_zip = zip(uniform_lower, uniform_upper)
    thalmic_mean = [-30, -15, 0, 15, 30]
    representation = ["last", "output", "reservoire"]

    table_name = "representation_test"

    reservoire_params = [
        {
            "erdos_connectivity": e,
            "erdos_uniform_lower": ul,
            "erdos_uniform_upper": up,
            "input_connectivity_p": ic,
            "representation": r,
            "input_scale": s,
            "thalmic_mean": t,
        }
        for e in erdos_connectivity
        for (ul, up) in uniform_zip
        for ic in input_connectivity
        for r in representation
        for s in input_scale
        for t in thalmic_mean
    ]
    resframe.sensitivity_analysis(reservoire_params, table_name)
    connection = resframe.create_connection()
    query = f"select accuracy, representation from {table_name};"
    if connection is not None:
        cur = connection.cursor()
        cur.execute(query)
        vals = cur.fetchall()
        cur.close()
        connection.commit()
        last_states = [res[0] for res in vals if res[1] == "last"]
        output = [res[0] for res in vals if res[1] == "output"]
        reservoire = [res[0] for res in vals if res[1] == "reservoire"]
        save_boxplot(last_states, output, reservoire)
    else:
        raise RuntimeError("Connection to db could not be established")


def sensitivity_plot(datasets):
    fig = plt.figure(figsize=(16, 9), dpi=80)
    labels = [
        "Thalmic noise [nA]",
        "Neuron connectivity",
        "Connectivity stregth between the neurons",
        "Input scaling",
    ]
    for i, dataset in enumerate(datasets):
        dataset.sort(key=lambda x: x[1])
        plt.subplot(2, 2, i + 1)
        accuracy = [thalmic_val[0] for thalmic_val in dataset]
        plt.ylabel("Prediction accuracy")
        plt.xlabel(labels[i])
        x = [thalmic_val[1] for thalmic_val in dataset]
        plt.plot(x, accuracy)
    pdf = PdfPages("output_repr.pdf")
    fig.savefig(pdf, format="pdf", bbox_inches="tight")
    pdf.close()
    plt.show()


def output_sensitivity_plot():
    table_name = "output_sensitivity_test"

    thalmic_dependent_inp = [
        {
            "erdos_connectivity": 0.5,
            "input_connectivity_p": 0.5,
            "representation": "output",
            "input_scale": 10,
            "thalmic_mean": t,
            "erdos_uniform_lower": 0,
            "erdos_uniform_upper": 1,
        }
        for t in np.arange(-90, 90, 10)
    ]

    connectivity_dependent_inp = [
        {
            "erdos_connectivity": c,
            "input_connectivity_p": 0.5,
            "representation": "output",
            "input_scale": 10,
            "thalmic_mean": 0,
            "erdos_uniform_lower": 0,
            "erdos_uniform_upper": 1,
        }
        for c in np.arange(0, 1, 0.1)
    ]

    input_connection_strength = [
        {
            "erdos_connectivity": 0.5,
            "input_connectivity_p": 0.5,
            "representation": "output",
            "input_scale": 10,
            "thalmic_mean": 0,
            "erdos_uniform_lower": 0,
            "erdos_uniform_upper": c,
        }
        for c in np.arange(1, 10, 1)
    ]

    input_scale_inp = [
        {
            "erdos_connectivity": 0.5,
            "input_connectivity_p": 0.5,
            "representation": "output",
            "input_scale": i,
            "thalmic_mean": 0,
            "erdos_uniform_lower": 0,
            "erdos_uniform_upper": 1,
        }
        for i in np.arange(0, 20, 1)
    ]

    all_input = np.array(
        list(
            itertools.chain.from_iterable(
                [
                    thalmic_dependent_inp,
                    connectivity_dependent_inp,
                    input_connection_strength,
                    input_scale_inp,
                ]
            )
        )
    )

    # to be sure
    all_inputs_repeated = np.repeat(all_input, 5)

    # save things to the db
    # resframe.sensitivity_analysis(all_inputs_repeated, "output_sensitivity_test")

    connection = resframe.create_connection()
    thalmic_query = (
        f"select avg(accuracy), thalmic_mean from {table_name} group by thalmic_mean;"
    )
    connectivity_query = f"select avg(accuracy), reservoire_connectivity from {table_name} group by reservoire_connectivity;"
    connection_strength_query = f"select avg(accuracy), connectivity_strength from {table_name} group by connectivity_strength;"
    input_scale_query = (
        f"select avg(accuracy), input_scaling from {table_name} group by input_scaling;"
    )

    if connection is not None:
        cur = connection.cursor()
        cur.execute(thalmic_query)
        thalmic_vals = cur.fetchall()

        cur.execute(connectivity_query)
        connectivity_vals = cur.fetchall()

        cur.execute(connection_strength_query)
        connection_strength_val = cur.fetchall()

        cur.execute(input_scale_query)
        input_scale_vals = cur.fetchall()

        sensitivity_plot(
            [thalmic_vals, connectivity_vals, connection_strength_val, input_scale_vals]
        )

        cur.close()
        connection.commit()
    else:
        raise RuntimeError("Connection to db could not be established")


if __name__ == "__main__":
    # representation_plot()
    output_sensitivity_plot()
