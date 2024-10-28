import matplotlib.pyplot as plt
import csv

DATA_FILE = "../data/output.csv"


def show_izhikevich():
    f = open(DATA_FILE)
    reader = csv.reader(f)

    data = [(float(d[0]), float(d[1])) for d in reader]
    d = [d[0] for d in data]
    # let's make this nice
    r = [min(d[1], 30) for d in data]

    plt.plot(d, r)
    plt.show()


show_izhikevich()

