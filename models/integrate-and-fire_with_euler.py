import matplotlib.pyplot as plt
from scipy.stats import norm, bernoulli

I = 2.0
E_leak = -50
thresh_value = -30
reset_value = -70
g_leak = 1.5
thr = 2
delta_t = 0.01


def euler(f, y, t, h):
    return y + h * f(y), t + h


def integrate_and_fire(V):
    return I - g_leak * (V - E_leak)


def main():
    res = [(-70.0, 0.0)]
    for _ in range(10000):
        impulse = bernoulli.rvs(p=0.01)
        y_last, t_last = res[-1]
        if impulse:
            impulse = norm.rvs(loc=10, scale=20)
            if impulse + y_last > thresh_value:
                res.append((thresh_value, t_last + delta_t))
                res.append((reset_value, t_last + 2 * delta_t))
            else:
                res.append((impulse + y_last, t_last + delta_t))
        else:
            y_last, t_last = res[-1]
            y, t = euler(integrate_and_fire, y_last, t_last, delta_t)
            res.append((y, t))

    xvals = [v[1] for v in res]
    yvals = [v[0] for v in res]

    fig1 = plt.figure()
    plt.plot(xvals, yvals)
    plt.show()
    fig1.savefig("2.png")


if __name__ == "__main__":
    main()
