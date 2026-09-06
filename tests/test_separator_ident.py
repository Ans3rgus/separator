# test_separator_ident.py
import sys
import math
import numpy as np
from pathlib import Path
import copy
import matplotlib.pyplot as plt

sys.path.insert(1, str(Path(__file__).parent.parent / "src"))

from net_separator import NetSeparatorModel, NetSeparatorParameters, NetSeparatorControl
from separator import SeparatorModel
from pid import PIDParameters, PIDController

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
    
    # Формирование последовательности управлений
    controls = []
    for i in range(N):
        if i < step_delay:
            controls.append(copy.deepcopy(initial_control))
        else:
            controls.append(copy.deepcopy(final_control))
    
    # Моделирование
    states = []
    
    # Сбрасываем модель
    current_net = copy.deepcopy(net_model)
    
    for i in range(N):
        if i == 0:
            # Получаем начальное состояние
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

def ident_arx1_ls(u, y):
    """
    Идентификация ARX-модели na=1, nb=1 методом наименьших квадратов
    Модель: y(t) = a1 * y(t-1) + b1 * u(t-1) + ε(t)
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
    
    # Решение методом наименьших квадратов: β = (XᵀX)⁻¹ XᵀY
    XTX = X.T @ X
    if np.linalg.det(XTX) < 1e-10:
        raise ValueError("Матрица XᵀX вырождена")
    
    XTX_inv = np.linalg.inv(XTX)
    beta = XTX_inv @ X.T @ Y
    
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

def test_pressure_identification():
    """Тест идентификации динамики давления"""
    print("\n=== Тест идентификации давления ===")
    
    # Создаем модель сепаратора
    params = NetSeparatorParameters.default_values()
    model = NetSeparatorModel(params)
    
    # Инициализация
    initial_level = 5.0
    initial_pressure = 7.5e5
    model.initialize_level_pressure(initial_level, initial_pressure)
    
    # Управление
    initial_control = NetSeparatorControl.default_values()
    final_control = NetSeparatorControl.default_values()
    
    # Изменяем положение газового клапана
    change_percent = 10
    final_control.valve_gas_opening = initial_control.valve_gas_opening * (1 + change_percent / 100)
    
    # Параметры моделирования
    dt = 1.0
    total_time = 3000
    step_count = int(total_time / dt)
    step_delay = 500
    
    # Снимаем кривую разгона
    t, controls, states = net_multistep(
        model, dt, step_count, step_delay, initial_control, final_control
    )
    
    # Извлекаем данные
    pressure = [state.separator_state.pressure_gas for state in states]
    valve_opening = [ctrl.valve_gas_opening for ctrl in controls]
    
    # Нормировка
    y_min = min(pressure)
    y_max = max(pressure)
    y_norm = normalize_data(pressure, y_min, y_max)
    
    u_min = min(valve_opening)
    u_max = max(valve_opening)
    u_norm = normalize_data(valve_opening, u_min, u_max)
    
    # Идентификация
    a1, b1 = ident_arx1_ls(u_norm, y_norm)
    k, T = ident_tf1_convert_coeffs(dt, a1, b1)
    
    print(f"Дискретная модель: a1 = {a1:.4f}, b1 = {b1:.4f}")
    print(f"Непрерывная модель: k = {k:.4f}, T = {T:.1f} с")
    
    # Проверка (примерные значения)
    assert 0 < a1 < 1, "Коэффициент a1 должен быть в диапазоне (0,1)"
    assert T > 0, "Постоянная времени должна быть положительной"
    
    if 'pydevd' in sys.modules:
        # Визуализация
        plt.figure(figsize=(10, 6))
        plt.plot(t, pressure, 'b-', linewidth=2, label='Давление')
        plt.axvline(x=step_delay*dt, color='r', linestyle='--', label='Скачок управления')
        plt.xlabel('Время, с')
        plt.ylabel('Давление, Па')
        plt.title('Кривая разгона по давлению')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()
    
    print("✅ Тест идентификации давления пройден")

def test_level_identification():
    """Тест идентификации динамики уровня"""
    print("\n=== Тест идентификации уровня ===")
    
    # Создаем модель сепаратора
    params = NetSeparatorParameters.default_values()
    model = NetSeparatorModel(params)
    
    # Инициализация
    initial_level = 5.0
    initial_pressure = 7.5e5
    model.initialize_level_pressure(initial_level, initial_pressure)
    
    # Управление
    initial_control = NetSeparatorControl.default_values()
    final_control = NetSeparatorControl.default_values()
    
    # Изменяем положение жидкостного клапана
    change_percent = 10
    final_control.valve_liquid_opening = initial_control.valve_liquid_opening * (1 + change_percent / 100)
    
    # Параметры моделирования (уровень медленнее реагирует)
    dt = 1.0
    total_time = 10000
    step_count = int(total_time / dt)
    step_delay = 500
    
    # Снимаем кривую разгона
    t, controls, states = net_multistep(
        model, dt, step_count, step_delay, initial_control, final_control
    )
    
    # Извлекаем данные
    level = [state.separator_state.level_liquid for state in states]
    valve_opening = [ctrl.valve_liquid_opening for ctrl in controls]
    
    # Простая проверка
    assert len(level) == step_count
    assert len(valve_opening) == step_count
    
    print(f"Уровень: min={min(level):.2f} м, max={max(level):.2f} м")
    print(f"Клапан: min={min(valve_opening):.3f}, max={max(valve_opening):.3f}")
    
    if 'pydevd' in sys.modules:
        # Визуализация
        plt.figure(figsize=(10, 6))
        plt.plot(t, level, 'b-', linewidth=2, label='Уровень')
        plt.axvline(x=step_delay*dt, color='r', linestyle='--', label='Скачок управления')
        plt.xlabel('Время, с')
        plt.ylabel('Уровень, м')
        plt.title('Кривая разгона по уровню')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()
    
    print("✅ Тест идентификации уровня пройден")

if __name__ == "__main__":
    print("Запуск тестов идентификации сепаратора...")
    test_pressure_identification()
    test_level_identification()
    print("\n✅ Все тесты идентификации пройдены")