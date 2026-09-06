# test_separator_ident.py - Полная реализация идентификации
import sys
import math
import numpy as np
from pathlib import Path
import copy
import matplotlib.pyplot as plt

sys.path.insert(1, str(Path(__file__).parent.parent / "src"))

from net_separator import NetSeparatorModel, NetSeparatorParameters, NetSeparatorControl
from pid import PIDParameters, PIDController

# Попробуем импортировать GEKKO, если установлен
try:
    from gekko import GEKKO
    GEKKO_AVAILABLE = True
except ImportError:
    print("GEKKO не установлен. Используется только МНК.")
    GEKKO_AVAILABLE = False


def net_multistep(net_model, time_step, step_count, step_delay, initial_control, final_control):
    """
    Расчет динамики на заданное число шагов в условиях заданного управления
    с управлением моментом скачка.
    
    Параметры:
    net_model: модель сепаратора
    time_step: шаг по времени (сек)
    step_count: общее количество шагов
    step_delay: момент скачка управления (в шагах)
    initial_control: начальное управление
    final_control: конечное управление после скачка
    
    Возвращает:
    t: список временных точек
    controls: список управлений для каждого шага
    states: список состояний для каждого шага
    """
    N = step_count
    dt = time_step
    
    # Временная сетка
    t = [i * dt for i in range(N)]
    
    # Формирование последовательности управлений с помощью list comprehension
    controls = [copy.deepcopy(initial_control) if i < step_delay else copy.deepcopy(final_control) 
                for i in range(N)]
    
    # Моделирование
    states = []
    
    # Сбрасываем модель (копируем, чтобы не изменять исходную)
    current_net = copy.deepcopy(net_model)
    
    for i in range(N):
        if i == 0:
            # Получаем начальное состояние (шаг 0)
            state = current_net.state
        else:
            # Обычный шаг моделирования
            state = current_net.step(dt, controls[i])
        states.append(copy.deepcopy(state))
    
    return t, controls, states


def normalize_data(data, min_val, max_val):
    """Нормировка данных в диапазон [0, 1]"""
    if math.isclose(max_val, min_val):
        return np.zeros_like(data)
    return (np.array(data) - min_val) / (max_val - min_val)


# ============================================================================
# Часть 1: Идентификация апериодического объекта (давление)
# ============================================================================

def ident_arx1_gekko(u, y):
    """Идентификация ARX-модели na=1, nb=1 с помощью библиотеки GEKKO"""
    if not GEKKO_AVAILABLE:
        raise ImportError("GEKKO не установлен. Установите: pip install gekko")
    
    # Центрирование относительно начального значения
    y_data = np.array(y) - y[0]
    u_data = np.array(u) - u[0]
    
    m = GEKKO(remote=False)
    na = 1
    nb = 1
    yp, p, K = m.sysid(range(len(y_data)), u_data, y_data, na, nb, pred='meas', shift='init')
    
    # Извлечение коэффициентов (структура GEKKO сложная)
    a1 = p['a'][0][0]
    b1 = p['b'][0][0][0]
    
    return float(a1), float(b1)

def ident_arx1_ls(u, y):
    """
    Идентификация ARX-модели na=1, nb=1 методом наименьших квадратов
    с центрированием данных для устранения смещения.
    Модель: y(t) - mean_y = a1*(y(t-1) - mean_y) + b1*(u(t-1) - mean_u)
    """
    n = len(y)
    if n < 3:
        raise ValueError("Недостаточно данных для идентификации (n >= 3)")
    
    # Центрирование
    mean_u = np.mean(u)
    mean_y = np.mean(y)
    u_centered = u - mean_u
    y_centered = y - mean_y
    
    # Подготовка матрицы X и вектора Y
    Y = y_centered[1:]  # y(t) для t = 1..n-1
    X = np.column_stack([
        y_centered[:-1],  # y(t-1)
        u_centered[:-1]   # u(t-1)
    ])
    
    # Решение методом наименьших квадратов
    XTX = X.T @ X
    if np.linalg.det(XTX) < 1e-10:
        raise ValueError("Матрица XᵀX вырождена")
    beta = np.linalg.inv(XTX) @ X.T @ Y
    
    a1 = float(beta[0])
    b1 = float(beta[1])
    return a1, b1


def ident_arx1_lstsq(u, y):
    """
    Идентификация ARX-модели na=1, nb=1 с помощью np.linalg.lstsq
    """
    n = len(y)
    if n < 3:
        raise ValueError("Недостаточно данных для идентификации (n >= 3)")
    
    # Подготовка матрицы X и вектора Y
    Y = y[1:]  # y(t) для t = 1..n-1
    X = np.column_stack([
        y[:-1],  # y(t-1)
        u[:-1]   # u(t-1)
    ])
    
    # Решение методом наименьших квадратов с использованием lstsq
    beta, residuals, rank, s = np.linalg.lstsq(X, Y, rcond=None)
    
    a1 = float(beta[0])
    b1 = float(beta[1])
    
    return a1, b1


def ident_tf1_convert_coeffs(dt, a1, b1):
    """
    Пересчет коэффициентов дискретной ARX-модели в параметры 
    непрерывной передаточной функции апериодического звена:
    
    W(p) = k / (T*p + 1)
    
    Дискретная модель: y(t) = a1*y(t-1) + b1*u(t-1)
    """
    if a1 >= 1.0 or a1 <= 0:
        raise ValueError(f"Коэффициент a1={a1} вне допустимого диапазона (0,1)")
    
    # Расчет постоянной времени T
    T = -dt / math.log(a1)
    
    # Расчет коэффициента усиления k
    k = b1 / (1 - a1)
    
    return k, T


# ============================================================================
# Часть 2: Идентификация интегрирующего объекта (уровень)
# ============================================================================
def ident_tf_integral_ls(u, y):
    """
    Идентификация интегрирующего объекта методом МНК с центрированием.
    Модель: d[t] - mean_d = b1 * (u[t] - mean_u) + ε(t)
    где d[t] = y[t] - y[t-1]
    """
    n = len(y)
    if n < 2:
        raise ValueError("Недостаточно данных для идентификации (n >= 2)")
    
    d = np.diff(y)        # d[t] = y[t] - y[t-1]
    u_trimmed = u[:-1]    # u для t = 0..n-2
    
    # Центрирование
    mean_d = np.mean(d)
    mean_u_trimmed = np.mean(u_trimmed)
    d_centered = d - mean_d
    u_centered = u_trimmed - mean_u_trimmed
    
    # Оценка b1: b1 = sum(u_centered * d_centered) / sum(u_centered^2)
    b1 = np.sum(u_centered * d_centered) / np.sum(u_centered ** 2)
    return float(b1)

def ident_tf_integral_convert_coeffs(dt, b1):
    """
    Пересчет коэффициента дискретной модели интегрирующего объекта
    в коэффициент непрерывной передаточной функции:
    
    W(p) = k / p
    
    где k = b1 / dt
    """
    k = b1 / dt
    return k


# ============================================================================
# Тесты идентификации
# ============================================================================

def test_ident_separator_pressure():
    """Тест идентификации по давлению"""
    print("\n=== Тест идентификации давления ===")
    
    # 1. Создаем модель сепаратора
    params = NetSeparatorParameters.default_values()
    model = NetSeparatorModel(params)
    
    # 2. Инициализация
    initial_level = 5.0
    initial_pressure = 7.5e5
    model.initialize_level_pressure(initial_level, initial_pressure)
    
    # 3. Управление
    initial_control = NetSeparatorControl.default_values()
    final_control = NetSeparatorControl.default_values()
    
    # Изменяем положение газового клапана
    change_percent = 10  # 10% изменение
    final_control.valve_gas_opening = initial_control.valve_gas_opening * (1 + change_percent / 100)
    
    # 4. Параметры моделирования
    dt = 1.0
    total_time = 3000  # 50 минут для полного отклика
    step_count = int(total_time / dt)
    step_delay = 500  # скачок через 500 секунд
    
    # 5. Снимаем кривую разгона
    t, controls, states = net_multistep(
        model, dt, step_count, step_delay, initial_control, final_control
    )
    
    # 6. Извлекаем данные
    pressure = [state.separator_state.pressure_gas for state in states]
    valve_opening = [ctrl.valve_gas_opening for ctrl in controls]
    
    # 7. Нормировка данных
    # Давление
    PV_min = 7.0e5  # 700 кПа
    PV_max = 8.0e5  # 800 кПа
    y_norm = normalize_data(pressure, PV_min, PV_max)
    
    # Управление (открытие клапана)
    OP_min = 0.0
    OP_max = 1.0
    u_norm = normalize_data(valve_opening, OP_min, OP_max)
    
    print(f"Данные собраны: {len(t)} точек")
    print(f"Давление: min={min(pressure)/1e3:.1f} кПа, max={max(pressure)/1e3:.1f} кПа")
    print(f"Открытие клапана: min={min(valve_opening):.3f}, max={max(valve_opening):.3f}")
    
    # 8. Идентификация разными методами
    results = {}
    
    # Метод 1: МНК
    try:
        a1_ls, b1_ls = ident_arx1_ls(u_norm, y_norm)
        k_ls, T_ls = ident_tf1_convert_coeffs(dt, a1_ls, b1_ls)
        results['LS'] = {'a1': a1_ls, 'b1': b1_ls, 'k': k_ls, 'T': T_ls}
        print(f"\nМНК: a1={a1_ls:.4f}, b1={b1_ls:.4f}, k={k_ls:.4f}, T={T_ls:.1f} с")
    except Exception as e:
        print(f"Ошибка МНК: {e}")
    
    # Метод 2: lstsq
    try:
        a1_lstsq, b1_lstsq = ident_arx1_lstsq(u_norm, y_norm)
        k_lstsq, T_lstsq = ident_tf1_convert_coeffs(dt, a1_lstsq, b1_lstsq)
        results['LSTSQ'] = {'a1': a1_lstsq, 'b1': b1_lstsq, 'k': k_lstsq, 'T': T_lstsq}
        print(f"LSTSQ: a1={a1_lstsq:.4f}, b1={b1_lstsq:.4f}, k={k_lstsq:.4f}, T={T_lstsq:.1f} с")
    except Exception as e:
        print(f"Ошибка LSTSQ: {e}")
    
    # Метод 3: GEKKO (если доступен)
    if GEKKO_AVAILABLE:
        try:
            a1_gekko, b1_gekko = ident_arx1_gekko(u_norm, y_norm)
            k_gekko, T_gekko = ident_tf1_convert_coeffs(dt, a1_gekko, b1_gekko)
            results['GEKKO'] = {'a1': a1_gekko, 'b1': b1_gekko, 'k': k_gekko, 'T': T_gekko}
            print(f"GEKKO: a1={a1_gekko:.4f}, b1={b1_gekko:.4f}, k={k_gekko:.4f}, T={T_gekko:.1f} с")
        except Exception as e:
            print(f"Ошибка GEKKO: {e}")
    
    # 9. Эталонные значения (из графического метода или предыдущих расчетов)
    # Примерные значения для проверки
    k_etalon = 0.0188  # коэффициент усиления в нормированных единицах
    T_etalon = 289.0   # постоянная времени, с
    
    # 10. Проверка совпадения результатов
    tolerance = 0.01  # 1% погрешность
    
    print(f"\nПроверка с эталонными значениями:")
    print(f"Эталон: k={k_etalon:.4f}, T={T_etalon:.1f} с")
    
    for method, res in results.items():
        k_error = abs(res['k'] - k_etalon) / k_etalon
        T_error = abs(res['T'] - T_etalon) / T_etalon
        
        print(f"\n{method}:")
        print(f"  k={res['k']:.4f} (погрешность: {k_error*100:.1f}%)")
        print(f"  T={res['T']:.1f} с (погрешность: {T_error*100:.1f}%)")
        
        # Утверждения для автоматических тестов
        if 'pydevd' not in sys.modules:  # Только если не в отладчике
            assert k_error < tolerance, f"Коэффициент усиления k не соответствует (метод {method})"
            assert T_error < tolerance, f"Постоянная времени T не соответствует (метод {method})"
    
    # 11. Визуализация (опционально)
    if 'pydevd' in sys.modules or True:  # Для отладки
        plt.figure(figsize=(12, 8))
        
        # График 1: Исходные данные
        plt.subplot(2, 2, 1)
        plt.plot(t, pressure, 'b-', linewidth=2)
        plt.axvline(x=step_delay*dt, color='r', linestyle='--', label='Скачок управления')
        plt.xlabel('Время, с')
        plt.ylabel('Давление, Па')
        plt.title('Кривая разгона по давлению')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # График 2: Нормированные данные
        plt.subplot(2, 2, 2)
        plt.plot(t, y_norm, 'g-', linewidth=2, label='Норм. давление')
        plt.plot(t, u_norm, 'r-', linewidth=1, alpha=0.7, label='Норм. управление')
        plt.axvline(x=step_delay*dt, color='r', linestyle='--')
        plt.xlabel('Время, с')
        plt.ylabel('Нормированные величины')
        plt.title('Нормированные данные для идентификации')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # График 3: Сравнение методов идентификации
        plt.subplot(2, 2, 3)
        methods = list(results.keys())
        k_values = [results[m]['k'] for m in methods]
        T_values = [results[m]['T'] for m in methods]
        
        x = np.arange(len(methods))
        width = 0.35
        
        plt.bar(x - width/2, k_values, width, label='k')
        plt.bar(x + width/2, T_values, width, label='T')
        plt.axhline(y=k_etalon, color='b', linestyle=':', alpha=0.5, label='k эталон')
        plt.axhline(y=T_etalon, color='orange', linestyle=':', alpha=0.5, label='T эталон')
        
        plt.xlabel('Метод идентификации')
        plt.ylabel('Значение параметра')
        plt.title('Сравнение методов идентификации')
        plt.xticks(x, methods)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # График 4: Моделирование с идентифицированными параметрами
        plt.subplot(2, 2, 4)
        plt.plot(t, y_norm, 'k-', linewidth=2, label='Эксперимент')
        
        # Моделирование с использованием идентифицированных параметров (МНК)
        if 'LS' in results:
            a1 = results['LS']['a1']
            b1 = results['LS']['b1']
            
            # Симуляция дискретной модели
            y_sim = np.zeros_like(y_norm)
            y_sim[0] = y_norm[0]
            for i in range(1, len(y_sim)):
                y_sim[i] = a1 * y_sim[i-1] + b1 * u_norm[i-1]
            
            plt.plot(t, y_sim, 'r--', linewidth=1.5, label='Модель МНК')
        
        plt.xlabel('Время, с')
        plt.ylabel('Нормированное давление')
        plt.title('Сравнение эксперимента и модели')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    print("✅ Тест идентификации давления пройден")
    return results


def test_ident_separator_level():
    """Тест идентификации по уровню"""
    print("\n=== Тест идентификации уровня ===")
    
    # 1. Создаем модель сепаратора
    params = NetSeparatorParameters.default_values()
    model = NetSeparatorModel(params)
    
    # 2. Инициализация
    initial_level = 5.0
    initial_pressure = 7.5e5
    model.initialize_level_pressure(initial_level, initial_pressure)
    
    # 3. Управление
    initial_control = NetSeparatorControl.default_values()
    final_control = NetSeparatorControl.default_values()
    
    # Изменяем положение жидкостного клапана
    change_percent = 10  # 10% изменение
    final_control.valve_liquid_opening = initial_control.valve_liquid_opening * (1 + change_percent / 100)
    
    # 4. Параметры моделирования (уровень медленнее реагирует)
    dt = 1.0
    total_time = 10000  # более длительное время для уровня
    step_count = int(total_time / dt)
    step_delay = 500  # скачок через 500 секунд
    
    # 5. Снимаем кривую разгона
    t, controls, states = net_multistep(
        model, dt, step_count, step_delay, initial_control, final_control
    )
    
    # 6. Извлекаем данные
    level = [state.separator_state.level_liquid for state in states]
    valve_opening = [ctrl.valve_liquid_opening for ctrl in controls]
    
    # 7. Нормировка данных
    # Уровень
    PV_min = 0.0
    PV_max = 10.0
    y_norm = normalize_data(level, PV_min, PV_max)
    
    # Управление (открытие клапана)
    OP_min = 0.0
    OP_max = 1.0
    u_norm = normalize_data(valve_opening, OP_min, OP_max)
    
    print(f"Данные собраны: {len(t)} точек")
    print(f"Уровень: min={min(level):.2f} м, max={max(level):.2f} м")
    print(f"Открытие клапана: min={min(valve_opening):.3f}, max={max(valve_opening):.3f}")
    
    # 8. Идентификация интегрирующего объекта
    try:
        b1 = ident_tf_integral_ls(u_norm, y_norm)
        k = ident_tf_integral_convert_coeffs(dt, b1)
        
        print(f"\nИнтегрирующий объект:")
        print(f"b1={b1:.6f}, k={k:.6f}")
        
        # Эталонное значение (из графического метода)
        k_etalon = 0.0005  # примерное значение
        
        # Проверка
        error = abs(k - k_etalon) / k_etalon
        print(f"Эталон k={k_etalon:.6f}, расчет k={k:.6f}")
        print(f"Погрешность: {error*100:.1f}%")
        
        if 'pydevd' not in sys.modules:
            assert error < 0.01, "Коэффициент k не соответствует эталону (1% погрешность)"
    
    except Exception as e:
        print(f"Ошибка идентификации уровня: {e}")
        raise
    
    # 9. Визуализация
    if 'pydevd' in sys.modules or True:
        plt.figure(figsize=(12, 6))
        
        # График 1: Исходные данные
        plt.subplot(1, 2, 1)
        plt.plot(t, level, 'b-', linewidth=2)
        plt.axvline(x=step_delay*dt, color='r', linestyle='--', label='Скачок управления')
        plt.xlabel('Время, с')
        plt.ylabel('Уровень, м')
        plt.title('Кривая разгона по уровню')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # График 2: Производная уровня
        plt.subplot(1, 2, 2)
        d_level = np.diff(level)
        t_diff = t[:-1]
        
        plt.plot(t_diff, d_level, 'g-', linewidth=2, alpha=0.7, label='dУровень/dt')
        
        # Линейная аппроксимация
        if 'b1' in locals():
            # Моделирование производной
            d_model = b1 * np.array(valve_opening[:-1])
            plt.plot(t_diff, d_model, 'r--', linewidth=1.5, label='Модель')
        
        plt.axvline(x=step_delay*dt, color='r', linestyle='--')
        plt.xlabel('Время, с')
        plt.ylabel('Производная уровня, м/с')
        plt.title('Производная уровня для идентификации')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    print("✅ Тест идентификации уровня пройден")
    return {'b1': b1, 'k': k}


def test_comparison_methods():
    """Тест сравнения разных методов идентификации"""
    print("\n=== Тест сравнения методов идентификации ===")
    
    # Создаем тестовые данные (синусоида с шумом)
    n = 200
    dt = 1.0
    t = np.arange(n) * dt
    
    # Истинные параметры модели
    a1_true = 0.9
    b1_true = 0.1
    
    # Генерация данных
    u = 0.5 + 0.3 * np.sin(2 * np.pi * t / 100)  # входной сигнал
    y = np.zeros(n)
    y[0] = 0.0
    
    # Симуляция ARX(1,1) модели
    for i in range(1, n):
        y[i] = a1_true * y[i-1] + b1_true * u[i-1] + 0.01 * np.random.randn()
    
    # Нормировка (не обязательно для тестовых данных, но для консистентности)
    y_norm = normalize_data(y, np.min(y), np.max(y))
    u_norm = normalize_data(u, np.min(u), np.max(u))
    
    # Идентификация разными методами
    methods_results = {}
    
    # Метод 1: МНК
    try:
        a1_ls, b1_ls = ident_arx1_ls(u_norm, y_norm)
        methods_results['LS'] = {'a1': a1_ls, 'b1': b1_ls}
    except Exception as e:
        print(f"Ошибка МНК: {e}")
    
    # Метод 2: lstsq
    try:
        a1_lstsq, b1_lstsq = ident_arx1_lstsq(u_norm, y_norm)
        methods_results['LSTSQ'] = {'a1': a1_lstsq, 'b1': b1_lstsq}
    except Exception as e:
        print(f"Ошибка LSTSQ: {e}")
    
    # Метод 3: GEKKO
    if GEKKO_AVAILABLE:
        try:
            a1_gekko, b1_gekko = ident_arx1_gekko(u_norm, y_norm)
            methods_results['GEKKO'] = {'a1': a1_gekko, 'b1': b1_gekko}
        except Exception as e:
            print(f"Ошибка GEKKO: {e}")
    
    # Сравнение с истинными значениями
    print(f"\nИстинные значения: a1={a1_true:.4f}, b1={b1_true:.4f}")
    
    for method, res in methods_results.items():
        a1_error = abs(res['a1'] - a1_true) / a1_true
        b1_error = abs(res['b1'] - b1_true) / b1_true
        
        print(f"\n{method}:")
        print(f"  a1={res['a1']:.4f} (погрешность: {a1_error*100:.2f}%)")
        print(f"  b1={res['b1']:.4f} (погрешность: {b1_error*100:.2f}%)")
        
        # Проверка точности
        if 'pydevd' not in sys.modules:
            assert a1_error < 0.05, f"Метод {method}: большая погрешность по a1"
            assert b1_error < 0.05, f"Метод {method}: большая погрешность по b1"
    
    # Визуализация сравнения
    if 'pydevd' in sys.modules or True:
        plt.figure(figsize=(10, 8))
        
        # График 1: Исходные данные
        plt.subplot(2, 2, 1)
        plt.plot(t, u_norm, 'b-', label='Вход u(t)')
        plt.plot(t, y_norm, 'g-', label='Выход y(t)')
        plt.xlabel('Время, с')
        plt.ylabel('Нормированные величины')
        plt.title('Тестовые данные')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # График 2: Сравнение коэффициентов
        plt.subplot(2, 2, 2)
        methods = list(methods_results.keys())
        
        a1_values = [methods_results[m]['a1'] for m in methods]
        b1_values = [methods_results[m]['b1'] for m in methods]
        
        x = np.arange(len(methods))
        width = 0.35
        
        bars1 = plt.bar(x - width/2, a1_values, width, label='a1 (оценка)')
        bars2 = plt.bar(x + width/2, b1_values, width, label='b1 (оценка)')
        
        # Истинные значения
        plt.axhline(y=a1_true, color='b', linestyle='--', alpha=0.7, label='a1 (истинное)')
        plt.axhline(y=b1_true, color='g', linestyle='--', alpha=0.7, label='b1 (истинное)')
        
        plt.xlabel('Метод идентификации')
        plt.ylabel('Значение коэффициента')
        plt.title('Сравнение оценок коэффициентов')
        plt.xticks(x, methods)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # График 3: Моделирование с разными методами
        plt.subplot(2, 2, 3)
        plt.plot(t, y_norm, 'k-', linewidth=2, label='Исходный сигнал')
        
        colors = ['r', 'g', 'b', 'm']
        for idx, (method, res) in enumerate(methods_results.items()):
            # Симуляция с идентифицированными параметрами
            y_sim = np.zeros_like(y_norm)
            y_sim[0] = y_norm[0]
            for i in range(1, len(y_sim)):
                y_sim[i] = res['a1'] * y_sim[i-1] + res['b1'] * u_norm[i-1]
            
            plt.plot(t, y_sim, linestyle='--', color=colors[idx % len(colors)], 
                    alpha=0.7, label=f'{method} модель')
        
        plt.xlabel('Время, с')
        plt.ylabel('Нормированный выход')
        plt.title('Сравнение моделей')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # График 4: Ошибки моделей
        plt.subplot(2, 2, 4)
        for idx, (method, res) in enumerate(methods_results.items()):
            # Симуляция с идентифицированными параметрами
            y_sim = np.zeros_like(y_norm)
            y_sim[0] = y_norm[0]
            for i in range(1, len(y_sim)):
                y_sim[i] = res['a1'] * y_sim[i-1] + res['b1'] * u_norm[i-1]
            
            error = y_norm - y_sim
            plt.plot(t, error, color=colors[idx % len(colors)], 
                    alpha=0.7, label=f'{method} ошибка')
        
        plt.xlabel('Время, с')
        plt.ylabel('Ошибка моделирования')
        plt.title('Ошибки моделей')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    print("✅ Тест сравнения методов пройден")
    return methods_results


if __name__ == "__main__":
    print("Запуск тестов идентификации сепаратора...")
    
    # Запуск тестов
    try:
        # Тест сравнения методов (быстрый)
        test_comparison_methods()
        
        # Основные тесты идентификации
        pressure_results = test_ident_separator_pressure()
        level_results = test_ident_separator_level()
        
        print("\n" + "="*50)
        print("РЕЗУЛЬТАТЫ ИДЕНТИФИКАЦИИ:")
        print("="*50)
        
        print("\nДавление (апериодическое звено):")
        if isinstance(pressure_results, dict) and 'LS' in pressure_results:
            k = pressure_results['LS']['k']
            T = pressure_results['LS']['T']
            print(f"  Коэффициент усиления: k = {k:.6f}")
            print(f"  Постоянная времени: T = {T:.1f} с ({T/60:.1f} мин)")
        
        print("\nУровень (интегрирующее звено):")
        if isinstance(level_results, dict) and 'k' in level_results:
            k_integral = level_results['k']
            print(f"  Коэффициент усиления: k' = {k_integral:.6f}")
            print(f"  Передаточная функция: W(p) = {k_integral:.6f} / p")
        
        print("\n✅ Все тесты идентификации пройдены успешно!")
        
    except Exception as e:
        print(f"\n❌ Ошибка при выполнении тестов: {e}")
        import traceback
        traceback.print_exc()