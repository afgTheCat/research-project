import resframe
import matplotlib.pyplot as plt


class RCModel:
    def __init__(self) -> None:
        self.reservoire = resframe.Reservoire(dt=0.05, number_of_neurons=1)

    def test_reservoire(self) -> None:
        input_steps = resframe.InputSteps([(1000.0, [10])])
        test_run = self.reservoire.get_states(input_steps)
        if test_run is not None:
            time, neurons = test_run
            plt.plot(time, neurons[0], label="Membrane Potential")[0]
            plt.show()
