import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import resframe


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
    q = "select accuracy, representation from representation_test;"
    if connection is not None:
        cur = connection.cursor()
        cur.execute(q)
        vals = cur.fetchall()
        cur.close()
        connection.commit()
        last_states = [res[0] for res in vals if res[1] == "last"]
        output = [res[0] for res in vals if res[1] == "output"]
        reservoire = [res[0] for res in vals if res[1] == "reservoire"]
        save_boxplot(last_states, output, reservoire)
    else:
        raise RuntimeError("Connection to db could not be established")


def output_sensitivity_plot():
    erdos_connectivity = [0.5]
    input_connectivity = [0.5]
    representation = ["output"]
    input_scale = np.arange(7.5, 12.5, 0.5)
    uniform_lower = [0]
    uniform_upper = [1]
    uniform_zip = zip(uniform_lower, uniform_upper)
    thalmic_mean = [0]

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
    # save things to the db
    resframe.sensitivity_analysis(reservoire_params, "output_sensitivity_test")


if __name__ == "__main__":
    representation_plot()
    # output_sensitivity_plot()
