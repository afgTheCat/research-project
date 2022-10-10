import csv
import matplotlib.pyplot as plt


def parse_output(filename):
    csv_file = open(f"outputs/{filename}")
    reader = csv.reader(csv_file)
    time = []
    v_values = []
    for row in reader:
        time.append(float(row[0]))
        v_values.append((float(row[1])))

    fig1 = plt.figure()
    plt.plot(time, v_values, label="Membrane Potential")[0]
    plt.show()


if __name__ == "__main__":
    parse_output("izikevich_model.dat")
