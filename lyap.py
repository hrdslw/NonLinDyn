import numpy as np
from sympy import symbols, Matrix, simplify

# Объявляем параметры и переменные
alpha, beta, gamma, x, y = symbols('alpha beta gamma x y')

# Система: функция Ляпунова, Матрица Якоби
J = Matrix([
    [beta - 2*beta*x - (alpha + beta)*y, -(alpha + beta)*x],
    [(alpha + gamma)*y, -gamma + (alpha + gamma)*x + 2*gamma*y]
])

# Обчисление собственных значений и характеристического уравнения
eigvals = J.eigenvals()
det_J = J.det()

# Параметры для проверки
params = [
    (0.5, 2, 1),  # пример 1
    (1.5, 1, 1),  # пример 2
    (2, 1, 3),    # пример 3
]

for param in params:
    alpha_val, beta_val, gamma_val = param
    J_subs = J.subs({alpha: alpha_val, beta: beta_val, gamma: gamma_val})
    eigvals_subs = [eigval.subs({alpha: alpha_val, beta: beta_val, gamma: gamma_val}) for eigval in eigvals]
    print(f"Для параметров alpha = {alpha_val}, beta = {beta_val}, gamma = {gamma_val}:")
    if np.all(np.imag(eigvals_subs) != 0):
        print("  Равновесие - Центр или Фокус (проводим дальнейший анализ).")
    
    print(f"  Собственные значения: {eigvals_subs}")