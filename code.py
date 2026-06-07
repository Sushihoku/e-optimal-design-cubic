import numpy as np
import sympy as sp
from scipy.optimize import root_scalar
from scipy.optimize import minimize

# 1. Поиск критической точки
def characteristic_polynomial(z: float) -> float:
    # Характеристический полином для кубической регрессии
    return 3*z**5 - 11*z**4 + 30*z**3 - 60*z**2 + 32*z - 64


def find_z_star(tolerance: float = 1e-12) -> float:
    # Нахождение квадрата критического радиуса методом Ньютона
    def df(z):
        # Аналитическая производная полинома для метода Ньютона
        return 15*z**4 - 44*z**3 + 90*z**2 - 120*z + 32

    result = root_scalar(characteristic_polynomial, fprime=df,
                         x0=2.6, method='newton', xtol=tolerance)
    if result.converged:
        return result.root
    else:
        raise RuntimeError(
            "Численный метод нахождения критической точки не сошелся")


# 2. Определение геометрических параметров интервала
def calculate_geometry(r1: float, r2: float):
    # Расчет радиуса r, центра омега и проверка критичности отрезка
    if r1 >= r2:
        raise ValueError(f"Левая граница должна быть строго меньше правой!")

    r = (r2 - r1) / 2.0
    omega = (r1 + r2) / 2.0
    z = r**2

    poly_val = characteristic_polynomial(z)

    # Учитываем погрешность вычислений при проверке знака полинома
    if abs(poly_val) < 1e-13:
        is_critical = False
    else:
        is_critical = poly_val > 0

    return r, omega, is_critical


# 3. Алгебра степенных рядов
def series_add(A, B):
    # Сложение коэффициентов рядов
    return A + B


def series_sub(A, B):
    # Вычитание коэффициентов рядов
    return A - B


def series_mul(A, B, order=10):
    # Умножение рядов через свертку коэффициентов
    C = np.zeros(order + 1)
    for n in range(order + 1):
        C[n] = sum(A[i] * B[n - i] for i in range(n + 1))
    return C


def series_div(A, B, order=10):
    # Деление рядов методом неопределенных коэффициентов
    C = np.zeros(order + 1)
    C[0] = A[0] / B[0]
    for n in range(1, order + 1):
        C[n] = (A[n] - sum(C[i] * B[n - i] for i in range(n))) / B[0]
    return C


# 4. Построение плана для докритического симметричного отрезка
def solve_symmetric_subcritical(r: float):
    # Оптимальный план Чебышёва при r <= r*
    points = np.array([-r, -r/2.0, r/2.0, r])

    r4 = r**4
    mu = (12 * r4 + 16) / (3 * (9 * r4 + 16))
    weights = np.array([0.5 - mu, mu, mu, 0.5 - mu])
    lambda_star = (r**6) / (9 * r4 + 16)

    return points, weights, lambda_star


# 5. Рекурсивное вычисление коэффициентов разложения
def compute_base_coefficients(order: int = 15):
    # Нахождение коэффициентов разложения для симметричного отрезка
    a = np.zeros(order + 1)
    b = np.zeros(order + 1)
    d = np.zeros(order + 1)
    lam = np.zeros(order + 1)

    # Начальные условия при бесконечном радиусе отрезка
    a[0], b[0], d[0], lam[0] = 0.0, 0.5, 1.0, 1.0

    b2 = np.zeros(order + 1)
    d2 = np.zeros(order + 1)
    db = np.zeros(order + 1)

    b2[0] = b[0] * b[0]
    d2[0] = d[0] * d[0]
    db[0] = d[0] * b[0]

    def val(arr, idx):
        return arr[idx] if 0 <= idx < len(arr) else 0.0

    # Нахождение коэффициентов на шаге n
    for n in range(1, order + 1):
        # 1. Вычисление b_n
        sum_b1 = sum(d[i]*d[n-i] - a[i]*a[n-i] for i in range(1, n))
        sum_b2 = sum(b[i]*(a[n-i] - d[n-i]) for i in range(1, n))
        b[n] = 0.5 * sum_b1 + 2.0 * sum_b2

        # 2. Вычисление d_n
        psi = 1.0 if n == 3 else (-1.0 if n == 4 else 0.0)

        term1 = sum(d2[i] * (val(b2, n-1-i) - 2*val(b, n-1-i))
                    for i in range(n))
        term2 = -val(d2, n-2)
        term3 = sum(val(b2, n-i) * (d[i] - d2[i]) for i in range(1, n))
        term4 = sum(d2[i] * val(b, n-2-i) for i in range(n-1))
        term5 = (val(d, n-2) - 2*val(d, n-3) + val(b2, n-2) + val(b, n-4) +
                 val(db, n-1) - 2*val(db, n-2) + val(db, n-3) + psi)

        sum_d1 = sum(d[i]*d[n-i] for i in range(1, n))
        d[n] = 4.0 * (term1 + term2 + term3 + term4 + term5) - sum_d1

        # 3. Вычисление a_n
        sum_a1 = sum(val(db, n-i)*(d[i] - a[i]) + a[i] *
                     (val(d, n-i) - val(d, n-1-i)) for i in range(1, n))
        a[n] = -d[n] + 2.0 * (val(d, n-1) - val(a, n-1) +
                              val(b, n-2) - val(d, n-2) - val(d2, n-1) - sum_a1)

        # 4. Вычисление lambda_n
        sum_l1 = sum(lam[i] * (val(d, n-1-i) + val(db, n-i) -
                     val(b2, n-i)) for i in range(1, n))
        lam[n] = -4.0 * (val(lam, n-2) + val(d, n-1) + sum_l1)

        # Обновление сверток для последующих шагов рекурсии
        b2[n] = sum(b[i]*b[n-i] for i in range(n + 1))
        d2[n] = sum(d[i]*d[n-i] for i in range(n + 1))
        db[n] = sum(d[i]*b[n-i] for i in range(n + 1))

    return a, b, d, lam, b2, d2, db


# 6. Построение плана для критического симметричного отрезка
def solve_symmetric_critical(r: float, order: int = 15):
    # Вычисление точек и весов по рядам Тейлора при r > r*
    z = r**2
    a, b, d, lam, _, _, _ = compute_base_coefficients(order)

    inv_z = 1.0 / z
    inv_z_pow = 1.0
    a_val, b_val, d_val, lam_val = 0.0, 0.0, 0.0, 0.0

    for n in range(order + 1):
        a_val += a[n] * inv_z_pow
        b_val += b[n] * inv_z_pow
        d_val += d[n] * inv_z_pow
        lam_val += lam[n] * inv_z_pow
        inv_z_pow *= inv_z

    t_a = a_val * z
    t_b = b_val * z
    t_d = d_val * z

    num = (1.0 + t_d) * (t_b - z) - z * t_b * (t_d - t_b)
    den = (t_a - z) * (1.0 + t_d + t_b * (t_d - t_b))
    mu_val = num / den

    x_tilde = np.sqrt(t_a)

    points = np.array([-r, -x_tilde, x_tilde, r])
    weights = np.array([(1.0 - mu_val) / 2.0, mu_val / 2.0,
                       mu_val / 2.0, (1.0 - mu_val) / 2.0])

    return points, weights, lam_val


# 7. Генерация и вывод таблицы коэффициентов
def generate_coefficients_table(order: int = 10):
    # Построение рядов для таблицы, включая деление рядов для mu
    a, b, d, lam, b2, d2, db = compute_base_coefficients(order)

    x = np.zeros(order + 1)
    x[1] = 1.0
    x2 = np.zeros(order + 1)
    x2[2] = 1.0

    b_minus_1 = b.copy()
    b_minus_1[0] -= 1.0
    a_minus_1 = a.copy()
    a_minus_1[0] -= 1.0

    db_minus_b2 = series_sub(db, b2)

    term1 = series_mul(x2, b_minus_1, order)
    term2 = series_mul(x, series_mul(d, b_minus_1, order), order)
    Num = series_sub(series_add(term1, term2), db_minus_b2)

    term3 = series_add(x2, series_mul(x, d, order))
    term4 = series_add(term3, db_minus_b2)
    Den = series_mul(a_minus_1, term4, order)

    mu = series_div(Num, Den, order)

    return a, b, d, mu, lam


def print_coefficients_table(order: int = 10):
    # Форматированный консольный вывод рассчитанной таблицы коэффициентов
    a, b, d, mu, lam = generate_coefficients_table(order)

    def fmt(val):
        if np.isclose(val, round(val), atol=1e-9):
            return str(int(round(val)))
        return f"{val:g}"

    rows = {'a': a, 'b': b, 'd': d, 'μ': mu, 'λ': lam}
    col_widths = [6] + [8] * (order + 1)

    header_top = "┌" + "┬".join("─" * w for w in col_widths) + "┐"
    header_mid = "├" + "┼".join("─" * w for w in col_widths) + "┤"
    header_bot = "└" + "┴".join("─" * w for w in col_widths) + "┘"

    print("\n" + " " * 15 +
          f"Table of expansion coefficients (Order {order})")
    print(header_top)
    print(f"│ {'N':^4} │ " + " │ ".join(f"{fmt(i):^6}" for i in range(order + 1)) + " │")
    print(header_mid)
    for name, values in rows.items():
        row_cells = [f"{fmt(v):^6}" for v in values]
        print(f"│ {name:^4} │ " + " │ ".join(row_cells) + " │")
    print(header_bot)


# 8. Численный оптимизатор для произвольного отрезка
def solve_numerical_arbitrary(r1: float, r2: float):
    # Поиск плана на произвольном отрезке методом последовательного квадратичного программирования
    r, omega, is_critical = calculate_geometry(r1, r2)

    # Сдвиг симметричного плана для построения начального приближения
    if is_critical:
        p_sym, w_sym, _ = solve_symmetric_critical(r)
    else:
        p_sym, w_sym, _ = solve_symmetric_subcritical(r)

    x1_init = p_sym[1] + omega
    x2_init = p_sym[2] + omega
    w0_init, w1_init, w2_init = w_sym[0], w_sym[1], w_sym[2]

    # Целевая функция (минимизация отрицательного собственного числа)
    def objective(v):
        x1, x2, w0, w1, w2 = v
        w3 = 1.0 - w0 - w1 - w2

        pts = np.array([r1, x1, x2, r2])
        wts = np.array([w0, w1, w2, w3])

        M = build_information_matrix(pts, wts)
        return -get_min_eigenvalue(M)

    # Ограничения на неотрицательность весов и упорядоченность точек
    def constraint_weights(v):
        return 1.0 - (v[2] + v[3] + v[4])

    def constraint_order(v):
        return v[1] - v[0]

    bounds = [(r1, r2), (r1, r2), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]

    constraints = [
        {'type': 'ineq', 'fun': constraint_weights},
        {'type': 'ineq', 'fun': constraint_order}
    ]

    v0 = [x1_init, x2_init, w0_init, w1_init, w2_init]

    res = minimize(objective, v0, bounds=bounds,
                   constraints=constraints, method='SLSQP', tol=1e-10)

    if not res.success:
        print("Warning: optimizer did not converge successfully")

    x1_opt, x2_opt, w0_opt, w1_opt, w2_opt = res.x
    w3_opt = 1.0 - w0_opt - w1_opt - w2_opt

    points = np.array([r1, x1_opt, x2_opt, r2])
    weights = np.array([w0_opt, w1_opt, w2_opt, w3_opt])
    lam_star = -res.fun

    return points, weights, lam_star


# 9. Расчет стартового вектора в точке симметрии
def get_symmetric_theta_0_vector(r: float):
    # Расчет 11-мерного вектора theta_0 при омега = 0
    pts, wts, lam_star = solve_symmetric_critical(r)

    x2_norm = pts[1] / r
    x3_norm = pts[2] / r

    # Построение матрицы Фишера
    M_real = np.zeros((4, 4))
    for i in range(4):
        f = np.array([1, pts[i], pts[i]**2, pts[i]**3])
        M_real += wts[i] * np.outer(f, f)

    # Собственные векторы четного и нечетного блоков матрицы Фишера
    M13 = M_real[np.ix_([0, 2], [0, 2])]
    M24 = M_real[np.ix_([1, 3], [1, 3])]

    vals1, evecs1 = np.linalg.eigh(M13)
    u_even = evecs1[:, np.argmin(vals1)]

    vals2, evecs2 = np.linalg.eigh(M24)
    v_odd = evecs2[:, np.argmin(vals2)]

    q1_val = v_odd[0] / v_odd[1]
    q0_val, q2_val = 0.0, 0.0
    q_at_r = q1_val * r + r**3

    # Определение масштаба многочлена p(x) из уравнений двойственности
    u0, u2 = u_even[0], u_even[1]
    p_at_r_base = u0 + u2 * r**2

    Cp2 = (lam_star - q_at_r**2) / (p_at_r_base**2)
    Cp = np.sqrt(max(0.0, Cp2))

    p0_val, p2_val = Cp * u0, Cp * u2
    p1_val = 0.0

    theta_0 = [
        p0_val, p1_val, p2_val,
        q0_val, q1_val, q2_val,
        wts[1], wts[2], wts[3],
        x2_norm, x3_norm
    ]
    return theta_0


# 10. Аналитический метод сдвига плана
def solve_shifted_design_analytical(r_val: float, omega_val: float, order: int = 1):
    # Расчет плана на произвольном отрезке методом дифференцирования по параметру сдвига
    theta_0_numeric = get_symmetric_theta_0_vector(r_val)

    # Символьные переменные и реальные координаты точек плана
    omega = sp.Symbol('omega', real=True)
    p0, p1, p2 = sp.symbols('p0 p1 p2', real=True)
    q0, q1, q3 = sp.symbols('q0 q1 q3', real=True)
    mu2, mu3, mu4 = sp.symbols('mu2 mu3 mu4', real=True)
    x2, x3 = sp.symbols('x2 x3', real=True)

    theta_vars = [p0, p1, p2, q0, q1, q3, mu2, mu3, mu4, x2, x3]

    x1_norm, x4_norm = -1.0, 1.0
    mu1 = 1.0 - mu2 - mu3 - mu4
    p3, q2 = 0.0, 0.0

    X_pts = [
        x1_norm * r_val + omega,
        x2 * r_val + omega,
        x3 * r_val + omega,
        x4_norm * r_val + omega
    ]
    Weights = [mu1, mu2, mu3, mu4]

    x_sym = sp.Symbol('x', real=True)
    p_x = p0 + p1*x_sym + p2*x_sym**2
    q_x = q0 + q1*x_sym + q2*x_sym**2 + q3*x_sym**3

    g_x = p_x**2 + q_x**2
    dg_dx = sp.diff(g_x, x_sym)

    lam = g_x.subs(x_sym, X_pts[3])

    p_coeffs = [p0, p1, p2, p3]
    q_coeffs = [q0, q1, q2, q3]

    # Система из 11 уравнений на основе свойств экстремального многочлена и моментов
    equations = []
    equations.append(g_x.subs(x_sym, X_pts[0]) - lam)
    equations.append(g_x.subs(x_sym, X_pts[1]) - lam)
    equations.append(g_x.subs(x_sym, X_pts[2]) - lam)

    equations.append(dg_dx.subs(x_sym, X_pts[1]))
    equations.append(dg_dx.subs(x_sym, X_pts[2]))

    for k in range(4):
        moment_p = sum(Weights[i] * p_x.subs(x_sym, X_pts[i])
                       * (X_pts[i]**k) for i in range(4))
        equations.append(moment_p - lam * p_coeffs[k])

    for k in range(2):
        moment_q = sum(Weights[i] * q_x.subs(x_sym, X_pts[i])
                       * (X_pts[i]**k) for i in range(4))
        equations.append(moment_q - lam * q_coeffs[k])

    F = sp.Matrix(equations)

    # Символьные Якобиан и вектор явных производных по смещению
    G_sym = F.jacobian(theta_vars)
    G_omega_sym = -F.diff(omega)

    subs_list: list[tuple[sp.Basic, float]] = [(omega, 0.0)]
    for i in range(11):
        subs_list.append((theta_vars[i], float(theta_0_numeric[i])))

    # Псевдообращение вырожденной матрицы Якоби в точке омега = 0
    G_0_num = np.array(G_sym.subs(subs_list), dtype=np.float64)
    G_0_inv = np.linalg.pinv(G_0_num)

    G_omega_0_num = np.array(G_omega_sym.subs(subs_list), dtype=np.float64).flatten()

    # Редукция Якобиана (индексы: p1 -> 1, q0 -> 3, q3 -> 5)
    active_indices = [1, 3, 5]
    G_reduced = G_0_num[:, active_indices]

    theta_active_dot, _, _, _ = np.linalg.lstsq(G_reduced, G_omega_0_num, rcond=None)

    theta_1_numeric = np.zeros(11)
    for i, idx in enumerate(active_indices):
        theta_1_numeric[idx] = theta_active_dot[i]

    theta_final = np.array(theta_0_numeric) + theta_1_numeric * omega_val

    # Извлекаем физические параметры
    mu2_val, mu3_val, mu4_val = theta_final[6], theta_final[7], theta_final[8]
    mu1_val = 1.0 - mu2_val - mu3_val - mu4_val

    x2_norm_val, x3_norm_val = theta_final[9], theta_final[10]

    points = np.array([-r_val, x2_norm_val * r_val,
                      x3_norm_val * r_val, r_val]) + omega_val
    weights = np.array([mu1_val, mu2_val, mu3_val, mu4_val])
    lam_val = float(lam.subs(subs_list))

    return points, weights, lam_val


# 11. Построение информационной матрицы и расчет собственного числа
def build_information_matrix(points, weights):
    # Построение информационной матрицы Фишера
    M = np.zeros((4, 4))
    for i in range(len(points)):
        f = np.array([1, points[i], points[i]**2, points[i]**3])
        M += weights[i] * np.outer(f, f)
    return M


def get_min_eigenvalue(matrix):
    # Нахождение минимального собственного числа матрицы
    eigenvalues, _ = np.linalg.eigh(matrix)
    return np.min(eigenvalues)


# 12. Диспетчер выбора оптимального метода
def get_optimal_design(r1: float, r2: float, order: int = 15):
    # Анализ геометрии и выбор метода решения
    r, omega, is_critical = calculate_geometry(r1, r2)

    if abs(omega) > 1e-10:
        points, weights, l_star = solve_numerical_arbitrary(r1, r2)
        return points, weights, l_star

    if is_critical:
        points, weights, l_star = solve_symmetric_critical(r, order=order)
    else:
        points, weights, l_star = solve_symmetric_subcritical(r)

    return points, weights, l_star
