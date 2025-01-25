import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sympy import symbols, solve

def system(beta, gamma, alpha):
    def rhs(t, X):
        x, y = X
        dxdt = x * (beta - beta * x - (alpha + beta) * y)
        dydt = y * (-gamma + (alpha + gamma) * x + gamma * y)
        return [dxdt, dydt]
    return rhs

def analyze_equilibria(beta, gamma, alpha):

    return [(0, 0), (0, 1), (1, 0), (gamma / (alpha + beta + gamma), beta / (alpha + beta + gamma))] #for eq in equilibria if 0 <= eq[0] <= 1 and 0 <= eq[1] <= 1

def analyze_stability(rhs, equilibrium, beta, gamma, alpha):
    x_eq, y_eq = equilibrium
    J = np.array([
        [beta - 2 * beta * x_eq - (alpha + beta) * y_eq, - (alpha + beta) * x_eq],
        [(alpha + gamma) * y_eq, -gamma + (alpha + gamma) * x_eq + 2 * gamma * y_eq]
    ])
    eigvals, eigvecs = np.linalg.eig(J)
    eigvecs = eigvecs.T
    # if x_eq == gamma / (alpha + beta + gamma):
    #     eigvals = np.array([1j * np.sqrt(alpha * beta * gamma / (alpha + beta + gamma)), -1j * np.sqrt(alpha * beta * gamma / (alpha + beta + gamma))])
    #print(np.all(np.real(eigvals)))
    if np.all(np.real(eigvals) < 0):
        if np.all(np.imag(eigvals) == 0):
            stability = "устойчивый узел"  
        else:
            stability = "устойчивый фокус"
    elif np.all(np.real(eigvals) > 0):
        if np.all(np.imag(eigvals) == 0):
            stability = "неустойчивый узел" 
        else:
            stability = "неустойчивый фокус"
    elif np.all(np.real(eigvals) == 0) and np.all(np.imag(eigvals) != 0):
        stability = "центр"
    elif np.any(np.real(eigvals) > 0) and np.any(np.real(eigvals) < 0):
        stability = "седло"
    else:
        stability = "неопределено"
    return stability, eigvals, eigvecs

def eq_quiver(rhs, limits, N=12):
    xlims, ylims = limits
    xs = np.linspace(xlims[0], xlims[1], N)
    ys = np.linspace(ylims[0], ylims[1], N)
    U, V = np.zeros((N, N)), np.zeros((N, N))

    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            if x + y <= 1:  # Проверка условия x + y <= 1
                u, v = rhs(0., (x, y))
                U[i, j] = u
                V[i, j] = v
            else:
                U[i, j] = np.nan  # Не рисуем стрелки вне области
                V[i, j] = np.nan
            # u, v = rhs(0., (x, y))
            # U[i, j] = u
            # V[i, j] = v

    return xs, ys, U, V

def plot_trajectories(rhs):
    for x0 in np.linspace(0, 1, 6):
        for y0 in np.linspace(0, 1, 6):
            if x0 + y0 <= 1:  # Проверка начальных условий
                # Останавливаем интегрирование, если x + y > 1
                def event(t, X):
                    x, y = X
                    return x + y - 1
                event.terminal = True  # Остановить интегрирование при event == 0
                event.direction = 1    # Остановить только при переходе из x + y < 1 в x + y > 1
                sol_pos = solve_ivp(rhs, [0., 6.], (x0, y0), method="RK45", rtol=1e-6)
                sol_neg = solve_ivp(rhs, [0., -6.], (x0, y0), method="RK45", rtol=1e-6)

                xsol = np.concatenate((sol_neg.y[0][::-1], sol_pos.y[0]))
                ysol = np.concatenate((sol_neg.y[1][::-1], sol_pos.y[1]))
                plt.plot(xsol, ysol, 'g-')
            # sol_pos = solve_ivp(rhs, [0., 6.], (x0, y0), method="RK45", rtol=1e-6)
            # sol_neg = solve_ivp(rhs, [0., -6.], (x0, y0), method="RK45", rtol=1e-6)

            # xsol = np.concatenate((sol_neg.y[0][::-1], sol_pos.y[0]))
            # ysol = np.concatenate((sol_neg.y[1][::-1], sol_pos.y[1]))
            # plt.plot(xsol, ysol, 'g-')

def plot_separatrices(rhs, equilibrium, eigvecs):
    for eigvec in eigvecs:
        eigvec = eigvec / np.linalg.norm(eigvec)

        sol_pos = solve_ivp(rhs, [0., 8.], (equilibrium[0] + 0.01*eigvec[0], equilibrium[1] + 0.01*eigvec[1]), method="RK45", rtol=1e-6)
        sol_neg = solve_ivp(rhs, [0., -8.], (equilibrium[0] + 0.01*eigvec[0], equilibrium[1] + 0.01*eigvec[1]), method="RK45", rtol=1e-6)

        xsol = np.concatenate((sol_neg.y[0][::-1], sol_pos.y[0]))
        ysol = np.concatenate((sol_neg.y[1][::-1], sol_pos.y[1]))
        plt.plot(xsol, ysol, 'r--', dashes=(5, 6))

        sol_pos = solve_ivp(rhs, [0., 8.], (equilibrium[0] - 0.01*eigvec[0], equilibrium[1] - 0.01*eigvec[1]), method="RK45", rtol=1e-6)
        sol_neg = solve_ivp(rhs, [0., -8.], (equilibrium[0] - 0.01*eigvec[0], equilibrium[1] - 0.01*eigvec[1]), method="RK45", rtol=1e-6)

        xsol = np.concatenate((sol_neg.y[0][::-1], sol_pos.y[0]))
        ysol = np.concatenate((sol_neg.y[1][::-1], sol_pos.y[1]))
        plt.plot(xsol, ysol, 'r--', dashes=(5, 6))

def plane_plot(beta, gamma, alpha, limits):
    rhs = system(beta, gamma, alpha)
    equilibria = analyze_equilibria(beta, gamma, alpha)

    plt.figure(figsize=(8, 8))
    plt.title(f"\u03B1={alpha}, \u03B2={beta}, \u03B3={gamma}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(-0.25, 1.25)
    plt.ylim(-0.25, 1.25)

    plot_trajectories(rhs)

    xs, ys, U, V = eq_quiver(rhs, limits)
    plt.quiver(xs, ys, U, V)

    for eq in equilibria:
        stability, eigvals, eigvecs = analyze_stability(rhs, eq, beta, gamma, alpha)
        color = 'bo' if ("устойчивый") in stability else 'ro'
        if ("центр") in stability:
          color = 'bo'
        plt.plot(eq[0], eq[1], color, markersize=8, label=f"({eq[0]:.2f}, {eq[1]:.2f}) - {stability}")

        if stability == "седло":
            plot_separatrices(rhs, eq, eigvecs)


    plt.legend(loc='upper right')
    plt.grid()
    plt.show()

alpha = float(input("Введите значение α (alpha): "))
beta = float(input("Введите значение β (beta): "))
gamma = float(input("Введите значение γ (gamma): "))

limits = [(0, 1), (0, 1)]
plane_plot(beta, gamma, alpha, limits)