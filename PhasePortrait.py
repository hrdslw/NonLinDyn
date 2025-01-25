import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ---------- ПАРАМЕТРЫ ----------
alpha = 1.0
beta  = 2.0
gamma = 1.5


# ---------- ПРАВЫЕ ЧАСТИ СИСТЕМЫ ----------
def system(t, xy):
    x, y = xy
    dx = x * (beta - beta*x - (alpha + beta)*y)
    dy = y * (-gamma + (alpha + gamma)*x + gamma*y)
    return [dx, dy]

# ---------- НАХОДИМ ТОЧКИ РАВНОВЕСИЯ (аналитически) ----------
EQ = []
# (0,0)
EQ.append( (0.0, 0.0) )
# (0,1)
EQ.append( (0.0, 1.0) )
# (1,0)
EQ.append( (1.0, 0.0) )
# (gamma/(alpha+beta+gamma), beta/(alpha+beta+gamma))
S = alpha + beta + gamma
if S > 1e-12:  # чтобы избежать деления на 0
    EQ.append( (gamma/S, beta/S) )

# ---------- ЧИСЛЕННАЯ ИНТЕГРАЦИЯ ДЛЯ ТРАЕКТОРИЙ ----------
t_span = (0, 50)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# набор начальных условий для "обычных" (оранжевых) траекторий
ics = [
    (0.1, 0.05),
    (0.2, 0.8),
    (0.8, 0.1),
    (0.4, 0.4),
    (0.2, 0.2),
    (0.3, 0.3),
    (0.1, 0.9),
    (1, 1.5)
]

plt.figure(figsize=(7,6))

# Интегрируем и рисуем траектории (оранжевые)
for (x0, y0) in ics:
    sol = solve_ivp(system, t_span, [x0, y0], t_eval=t_eval)
    plt.plot(sol.y[0], sol.y[1], color='orange', lw=2)

# ---------- ПОСТРОЕНИЕ СЕПАРАТРИС СЕДЕЛ ----------
# Для седловой точки уравнения 2D можно найти собственные векторы
# и "запустить" интегрирование вперед/назад по этим направлениям.

def jacobian(x, y):
    dfdx = beta - 2*beta*x - (alpha + beta)*y
    dfdy = -(alpha + beta)*x
    dgdx = (alpha + gamma)*y
    dgdy = -gamma + (alpha + gamma)*x + 2*gamma*y
    return np.array([[dfdx, dfdy],
                     [dgdx, dgdy]])

def integrate_separatrix(eq_point, tmax=50, npoints=2000, eps=1e-5):
    """Интегрирует траектории, стартующие вблизи седла eq_point
       вдоль собственных векторов Jacobian (устойчивого/неустойчивого).
    """
    # 1) Собственные значения и векторы
    J = jacobian(eq_point[0], eq_point[1])
    vals, vecs = np.linalg.eig(J)
    
    # 2) Для каждоого собственного значения проверим знак Re(lambda)
    # Если Re(lambda) < 0 -- устойчивое направление (интегрируем вперед),
    # Если Re(lambda) > 0 -- неустойчивое направление (интегрируем назад).
    # (или наоборот, в зависимости от удобства)
    separatrices = []
    
    for i in range(2):
        lam = vals[i]
        vec = np.real(vecs[:, i])
        vec /= np.linalg.norm(vec)
        
        # Стартовая точка слегка смещена от eq_point на eps вдоль vec
        # или против vec (если хотим обе "ветви").
        for sign in [+1, -1]:
            init = eq_point + sign*eps*vec
            
            def f_for_separatrix(t, xy):
                return system(t, xy)
            
            if lam.real < 0:
                # устойчивое -- траекторию "вычисляем" вперед
                # (т.к. при t->+&infin; она должна приходить в eq_point)
                sol = solve_ivp(f_for_separatrix,
                                [0, tmax],
                                init,
                                t_eval=np.linspace(0, tmax, npoints))
                separatrices.append((sol.y[0], sol.y[1]))
            else:
                # неустойчивое -- траекторию "вычисляем" назад
                # (чтобы при t->-&infin; она приходила в eq_point)
                sol = solve_ivp(f_for_separatrix,
                                [-tmax, 0],
                                init,
                                t_eval=np.linspace(-tmax, 0, npoints))
                separatrices.append((sol.y[0], sol.y[1]))
                
    return separatrices

# Выбираем седловые точки для сепаратрис
saddles = []
for (x_e, y_e) in EQ:
    # Смотрим собственные значения в этой точке
    J = jacobian(x_e, y_e)
    vals, _ = np.linalg.eig(J)
    # Проверяем наличие положительной и отрицательной реальной части
    if (vals.real.min() < 0) and (vals.real.max() > 0):
        # это седло
        saddles.append( (x_e, y_e) )

# Рисуем каждую сепаратрису красной пунктирной линией
for sd in saddles:
    seps = integrate_separatrix(sd)
    for (xx, yy) in seps:
        plt.plot(xx, yy, 'r--', lw=1.5)

# ---------- РИСУЕМ САМИ ТОЧКИ РАВНОВЕСИЯ ----------
for (x_e, y_e) in EQ:
    plt.plot(x_e, y_e, 'ko', ms=8)

plt.title(f"Фазовый портрет при alpha={alpha}, beta={beta}, gamma={gamma}")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim(-1, 2)
plt.ylim(-1, 2)
plt.grid(True)
plt.show()




# ---------- БИФУРКАЦИОННАЯ ДИАГРАММА ПО ПАРАМЕТРУ alpha ----------
beta = 2.0
gamma = 1.5

alpha_values = np.linspace(0, 5, 200)

x_eq_1 = []  # (0,0)
x_eq_2 = []  # (0,1)
x_eq_3 = []  # (1,0)
x_eq_in = [] # внутреннее

for alpha in alpha_values:
    x_eq_1.append(0.0)  # для (0,0)
    x_eq_2.append(0.0)  # для (0,1)
    x_eq_3.append(1.0)  # для (1,0)
    S = alpha + beta + gamma
    if S > 1e-10:
        x_eq_in.append( gamma / S )
    else:
        x_eq_in.append( np.nan )  # нет смысла

# plt.figure(figsize=(7,5))
# plt.plot(alpha_values, x_eq_1, 'b-', label='(0,0): x=0')
# plt.plot(alpha_values, x_eq_2, 'g-', label='(0,1): x=0')
# plt.plot(alpha_values, x_eq_3, 'r-', label='(1,0): x=1')
# plt.plot(alpha_values, x_eq_in, 'm-', label='внутреннее: x=γ/(α+β+γ)')

# plt.xlabel('alpha')
# plt.ylabel('x-координата равновесия')
# plt.title('Бифуркационная диаграмма по параметру alpha (при beta=2, gamma=1.5)')
# plt.legend()
# plt.grid(True)
# plt.show()