import pytest
import sys
import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Добавляем путь к src для импорта
sys.path.insert(1, str(Path(__file__).parent.parent / "src")) 

from separator import *

def test_separator_default_parameters():
    """Тест создания сепаратора с параметрами по умолчанию"""
    fluid = FluidParameters.default_values()
    separator_state = SeparatorState.default_values()
    separator_parameters = SeparatorParameters.default_values()
    
    separator = SeparatorModel(fluid, separator_parameters, separator_state)
    
    # Проверяем, что объект создан успешно
    assert separator is not None
    assert isinstance(separator, SeparatorModel)
    assert separator.parameters.volume == 100
    assert separator.parameters.area == 10

def test_gas_only_step():
    """Тест подачи только газа в сепаратор"""
    
    fluid = FluidParameters.default_values()
    separator_state = SeparatorState.default_values()
    separator_parameters = SeparatorParameters.default_values()
    separator = SeparatorModel(fluid, separator_parameters, separator_state)

    # Исходные параметры
    initial_level = 0
    initial_pressure = 0
    initial_mass_gas = 0

    """Параметры шага"""
    # Общий расход смеси
    Gin_mix = 100.0
    # Только газ
    omega = 1.0
    # Отбор газа
    Ggas_out = 50.0
    # Нет отбора жидкости
    Gliquid_out = 0.0
    # Временной шаг
    dt = 1.0
    
    # Выполняем шаг с подачей только газа
    step_result = separator.step(dt, omega, Gin_mix, Ggas_out, Gliquid_out)

    """Проверяем результаты"""
    # Уровень жидкости должен остаться неизменным
    assert math.isclose(step_result.level_liquid, initial_level)
    
    # Масса газа должна увеличиться на разницу между притоком и отбором
    expected_gas_mass = initial_mass_gas + (Gin_mix * omega - Ggas_out) * dt
    assert math.isclose(step_result.mass_gas, expected_gas_mass)

def test_liquid_only_step_empty_separator():
    """Тест подачи только жидкости в пустой сепаратор"""
    
    fluid = FluidParameters.default_values()
    separator_state = SeparatorState.default_values()
    separator_parameters = SeparatorParameters.default_values()
    separator = SeparatorModel(fluid, separator_parameters, separator_state)

    # Параметры шага
    Gin_mix = 100.0  # кг/с
    omega = 0.0      # только жидкость
    Ggas_out = 0.0
    Gliquid_out = 30.0  # кг/с
    dt = 1.0
    
    # Выполняем шаг с подачей только жидкости
    step_result = separator.step(dt, omega, Gin_mix, Ggas_out, Gliquid_out)

    # Проверяем результаты
    # Масса жидкости должна увеличиться на 70 кг (100 - 30)
    expected_liquid_mass = 70.0
    assert math.isclose(step_result.mass_liquid, expected_liquid_mass)
    
    # Объем жидкости
    expected_volume_liquid = expected_liquid_mass / 1000
    assert math.isclose(step_result.volume_liquid, expected_volume_liquid)
    
    # Объем газа
    assert math.isclose(step_result.volume_gas, 0)
    
    # Уровень жидкости
    expected_level = expected_volume_liquid / 10  # 0.07 / 10 = 0.007 м
    assert math.isclose(step_result.level_liquid, expected_level)
    
    # Давление газа должно остаться 0 (газа нет)
    assert step_result.pressure_gas == 0
    
    # Давление жидкости должно быть только от гидростатики
    expected_pressure_liquid = 1000 * expected_level * 10  # 1000 * 0.007 * 9.81 ≈ 68.67 Па
    assert math.isclose(step_result.pressure_liquid, expected_pressure_liquid)

def test_mixed_flow_step_empty_separator():
    """Тест подачи смеси в пустой сепаратор"""
    
    fluid = FluidParameters.default_values()
    separator_state = SeparatorState.default_values()  # Все = 0
    separator_parameters = SeparatorParameters.default_values()
    separator = SeparatorModel(fluid, separator_parameters, separator_state)

    # Параметры шага
    Gin_mix = 100.0  # кг/с
    omega = 0.7      # 70% газа, 30% жидкости
    Ggas_out = 20.0  # отбор газа
    Gliquid_out = 10.0  # отбор жидкости
    dt = 1.0
    
    step_result = separator.step(dt, omega, Gin_mix, Ggas_out, Gliquid_out)

    # Проверяем материальный баланс
    expected_gas_mass = (Gin_mix * omega - Ggas_out) * dt
    expected_liquid_mass = (Gin_mix * (1 - omega) - Gliquid_out) * dt
    
    assert math.isclose(step_result.mass_gas, expected_gas_mass)
    assert math.isclose(step_result.mass_liquid, expected_liquid_mass)
    
    # Проверяем объемы
    expected_volume_liquid = expected_liquid_mass / 1000
    expected_volume_gas = separator_parameters.volume - expected_volume_liquid
    
    assert math.isclose(step_result.volume_liquid, expected_volume_liquid)
    assert math.isclose(step_result.volume_gas, expected_volume_gas)
    
    # Проверяем давления
    if expected_gas_mass > 0 and expected_volume_gas > 0:
        expected_pressure_gas = (expected_gas_mass / (fluid.gas_molar_mass * expected_volume_gas)) * (8.314 * fluid.temperature)
        assert math.isclose(step_result.pressure_gas, expected_pressure_gas)
    
    expected_pressure_liquid = step_result.pressure_gas + 1000 * step_result.level_liquid * 10
    assert math.isclose(step_result.pressure_liquid, expected_pressure_liquid)

def test_transient_process():
    """Тест переходного процесса заполнения сепаратора двухфазной смесью"""
    
    fluid = FluidParameters.default_values()
    
    # Задание состояния в рчуную
    separator_state = SeparatorState()
    separator_state.mass_gas = 0
    separator_state.mass_liquid = 0
    separator_state.volume_liquid = 0
    separator_state.volume_gas = 0
    separator_state.pressure_gas = 0
    separator_state.pressure_liquid = 0
    separator_state.level_liquid = 0

    separator_parameters = SeparatorParameters.default_values()
    separator = SeparatorModel(fluid, separator_parameters, separator_state)

    # Параметры симуляции
    dt = 1.0  # шаг времени, с
    total_time = 10000  # общее время симуляции, с
    steps = int(total_time / dt)
    
    # Параметры потока
    Gin_mix = 5.64  # кг/с
    omega = 0.33     # 33% газа, 67% жидкости
    flap_gas = 0.5 # степень открытия клапана газа
    Ggas_out = Gin_mix * omega * flap_gas
    flap_liq = 0.5 # степень открытия клапана жидкости
    Gliquid_out = Gin_mix * (1-omega) * flap_liq
    
    # Массивы для хранения результатов
    times = np.zeros(steps)
    pressures = np.zeros(steps)
    levels = np.zeros(steps)
    mass_gas = np.zeros(steps)
    mass_liquid = np.zeros(steps)
    
    # Сохраняем первую итерацию
    times[0] = 0
    pressures[0] = separator.state.pressure_gas
    levels[0] = separator.state.level_liquid
    mass_gas[0] = separator.state.mass_gas
    mass_liquid[0] = separator.state.mass_liquid
    
    # Выполняем симуляцию (начинаем с 1, т.к. 0-й шаг уже записан)
    for i in range(1, steps):
        current_state = separator.step(dt, omega, Gin_mix, Ggas_out, Gliquid_out)
        
        times[i] = i * dt
        pressures[i] = current_state.pressure_gas
        levels[i] = current_state.level_liquid
        mass_gas[i] = current_state.mass_gas
        mass_liquid[i] = current_state.mass_liquid
    
    # Строим графики
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(times, pressures)
    plt.xlabel('Время, с')
    plt.ylabel('Давление газа, Па')
    plt.title('Давление в сепараторе')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(times, levels)
    plt.xlabel('Время, с')
    plt.ylabel('Уровень жидкости, м')
    plt.title('Уровень жидкости в сепараторе')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(times, mass_gas, label='Масса газа')
    plt.plot(times, mass_liquid, label='Масса жидкости')
    plt.xlabel('Время, с')
    plt.ylabel('Масса, кг')
    plt.title('Массы компонентов')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    total_mass = mass_gas + mass_liquid
    plt.plot(times, total_mass)
    plt.xlabel('Время, с')
    plt.ylabel('Общая масса, кг')
    plt.title('Общая масса в сепараторе')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('separator_transient_process.png')
    plt.show()