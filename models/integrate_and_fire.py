from scipy.integrate import ode
import matplotlib.pyplot as plt

I = 2.0
E_leak = -50
thresh_value = -30
reset_value = -70
g_leak = 1.5
thr = 2
delta_t = 0.01
euler_counter = 0


def leaky_model(_t, y, *arg):
    return I - g_leak * (y - E_leak)


def main():
    # initial value, time
    res = [(-70.0, 0.0)]
    r = ode(leaky_model).set_integrator("zvode").set_initial_value(-70.0, 0.0)
    t1 = 10
    dt = 0.01
    while r.successful() and r.t < t1:
        v = r.integrate(r.t + dt)
        res.append((v, r.t))

    xvals = [v[1] for v in res]
    yvals = [v[0].real for v in res]

    fig1 = plt.figure()
    plt.plot(xvals, yvals)
    plt.show()
    fig1.savefig("3.png")


if __name__ == "__main__":
    main()
