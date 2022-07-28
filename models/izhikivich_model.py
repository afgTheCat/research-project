import numpy as np
import matplotlib.pyplot as plt

MODEL_VALS = [
    ((0.02, 0.2, -65, 8), "regular spiking"),
    ((0.02, 0.2, -55, 4), "intrinsically spiking"),
    ((0.02, 0.2, -50, 2), "chattering"),
    ((0.1, 0.2, -65, 8), "fast spiking"),
    ((0.02, 0.25, -65, 8), "low treshold spiking"),
    ((0.1, 0.26, -65, 8), "resonator"),
]


def izhikivitch_model(a=0.02, b=0.2, c=-65, d=8):
    spike_value = 35  # Maximal Spike Value

    T = 1000  # total simulation length [ms]
    dt = 0.5  # step size [ms]
    time = np.arange(0, T + dt, dt)  # step values [ms]
    V = np.zeros(len(time))  # array for saving voltage history
    V[0] = -70  # set initial to resting potential
    u = np.zeros(len(time))  # array for saving Recovery history
    u[0] = -14
    I = np.zeros(len(time))
    I[200:1500] = 10

    for t in range(1, len(time)):
        # if we still didnt reach spike potential
        if V[t - 1] < spike_value:
            # ODE for membrane potential
            dV = (0.04 * V[t - 1] + 5) * V[t - 1] + 140 - u[t - 1]
            V[t] = V[t - 1] + (dV + I[t - 1]) * dt
            # ODE for recovery variable
            du = a * (b * V[t - 1] - u[t - 1])
            u[t] = u[t - 1] + dt * du
        # spike reached!
        else:
            V[t - 1] = spike_value  # set to spike value
            V[t] = c  # reset membrane voltage
            u[t] = u[t - 1] + d  # reset recovery

    return V


def I_values(time=None):
    I_len = 1000 if time is None else len(time)
    I = np.zeros(I_len)
    I[200:1500] = 10
    return I


def izhikivitch_sim():
    # time parameters for plotting
    T = 1000  # total simulation length [ms]
    dt = 0.5  # step size [ms]
    time = np.arange(0, T + dt, dt)  # step values [ms]

    for model in MODEL_VALS:
        model_params = model[0]
        model_name = model[1]

        V = izhikivitch_model(*model_params)
        I = I_values(time=time)
        fig1 = plt.figure()
        plt.plot(time, V, label="Membrane Potential")[0]
        plt.plot(time, I, label="Applied Current")[0]
        plt.show()
        fig1.savefig(model_name)


if __name__ == "__main__":
    izhikivitch_sim()
