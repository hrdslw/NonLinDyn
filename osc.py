import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

##############################################################################
# 1) ПАРАМЕТРЫ, КОТОРЫЕ МЫ ФИКСИРУЕМ

beta = 2.0
gamma = 1.5

# Диапазон alpha, по которому идём
alpha_values = np.linspace(0.1, 5.0, 30)

##############################################################################
# 2) ФУНКЦИЯ, ОПРЕДЕЛЯЮЩАЯ СИСТЕМУ

def system(t, xy, alpha, beta, gamma):
    x, y = xy
    dx = x*(beta - beta*x - (alpha + beta)*y)
    dy = y*(-gamma + (alpha + gamma)*x + gamma*y)
    return [dx, dy]

##############################################################################
# 3) ФУНКЦИЯ ДЛЯ ЧИСЛЕННОГО НАХОЖДЕНИЯ ПЕРИОДА (метод локальных максимумов)

def find_period(t, x):
    """
    По массивам t и x(t) находим время между локальными максимумами x(t).
    Функция возвращает оценку периода, или np.nan, если не удаётся.
    """
    # Ищем индексы, где x(t) меняет направление derivative с + на -
    # локальный максимум => x'(i-1) >0, x'(i+1)<0. 
    # Проще через numpy:
    #   maxima = (np.diff(np.sign(np.diff(x))) < 0).nonzero()[0] + 1
    # затем берём t[maxima], считаем среднюю разность между соседними максимумами.
    
    dx = np.diff(x)
    # sign(dx) - знаки приращений, np.diff(sign(dx)) < 0 => индексы максимумов
    sign_dx = np.sign(dx)
    dsign = np.diff(sign_dx)
    # точки, где dsign < 0 => переход от + к -
    max_idx = np.where(dsign < 0)[0] + 1  # +1 смещение, т.к. diff
    
    if len(max_idx) < 2:
        return np.nan  # мало максимумов, не можем измерить период
    
    # берём массив времён максимумов
    t_max = t[max_idx]
    # считаем среднее расстояние по времени между соседними максимумами
    if len(t_max) >= 2:
        Tvals = np.diff(t_max)
        T_mean = np.mean(Tvals)
        return T_mean
    else:
        return np.nan

##############################################################################
# 4) ОСНОВНОЙ ЦИКЛ: ДЛЯ КАЖДОГО alpha -> ЧИСЛЕННО НАХОДИМ ПЕРИОД

num_periods = []
lin_periods = []

for alpha in alpha_values:
    # ЛИНЕЙНЫЙ ПЕРИОД:
    # T_lin = 2*pi * sqrt( (alpha+beta+gamma)/(alpha*beta*gamma ) )
    # но будьте аккуратны, если alpha=0 => вырожденность. 
    # Предположим alpha>0, beta>0, gamma>0
    T_lin = 2*np.pi * np.sqrt( (alpha + beta + gamma)/(alpha*beta*gamma) )
    lin_periods.append(T_lin)
    
    # Найдём точку внутреннего равновесия (x*, y*) 
    S = alpha + beta + gamma
    if S <= 1e-14:
        # вырожденность
        num_periods.append(np.nan)
        continue
    x_eq = gamma / S
    y_eq = beta / S
    
    # Зададим начальное условие, чуть сдвинутое от равновесия
    x0 = x_eq * 1.01
    y0 = y_eq * 0.99
    
    # Интегрируем систему
    t_span = (0, 300)  # достаточно большой интервал, чтобы накопить колебания
    t_eval = np.linspace(t_span[0], t_span[1], 30000)  # нужна достаточно плотная сетка
    
    sol = solve_ivp(system, t_span, [x0, y0], args=(alpha,beta,gamma),
                    t_eval=t_eval, rtol=1e-8, atol=1e-10)
    
    x_t = sol.y[0]
    T_num = find_period(sol.t, x_t)
    num_periods.append(T_num)

num_periods = np.array(num_periods)
lin_periods = np.array(lin_periods)

##############################################################################
# 5) ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ

plt.figure(figsize=(7,5))
plt.title("Зависимость периода колебаний x(t) от alpha (beta=2.0, gamma=1.5)")
plt.plot(alpha_values, num_periods, 'o-', label='Численный период')
plt.plot(alpha_values, lin_periods, 'r--', label='Линейный период')
plt.xlabel('alpha')
plt.ylabel('Период T')
plt.legend()
plt.grid(True)
plt.show()