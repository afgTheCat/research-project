from scipy.integrate import ode
import matplotlib.pyplot as plt

I = 2.0
E_leak = -50
thresh_value = -30
reset_value = -70
g_leak = 1.5
thr = 2
V_1_2 = -20
delta_t = 0.01
k = 0.1
threshold = 40


def resonate_and_fire(_t, x, *args):
    V = x[0]
    W = x[1]

    dVdt = I - g_leak * (V - E_leak) - W
    dWdt = (V - V_1_2) / k - W

    return [dVdt, dWdt]


def main():
    r = ode(resonate_and_fire).set_integrator("zvode").set_initial_value([60, -20], 0.0)

    t1 = 10
    dt = 0.01
    res = []

    while r.successful() and r.t < t1:
        v = r.integrate(r.t + dt)
        if v[1] > threshold:
            p = [-30, 40]
            r.set_initial_value(p, r.t)
            res.append((p, r.t))
        else:
            res.append((v, r.t))

    time = [x[1] for x in res]
    points = [(x[0][0].real, x[0][1].real) for x in res]
    x = [p[0] for p in points]
    y = [p[1] for p in points]

    fig, axs = plt.subplots(2)
    axs[0].plot(x, y)
    axs[1].plot(time, y)
    plt.show()
    fig.savefig("x.png")


if __name__ == "__main__":
    main()
