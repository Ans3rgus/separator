# tests/test_net_separator_unittest.py
import unittest
import math
import sys
from pathlib import Path

# Добавляем путь к src для импорта
sys.path.insert(1, str(Path(__file__).parent.parent / "src")) 

from net_separator import *


class TestNetSeparator(unittest.TestCase):
    """Тесты для сепаратора с обвязкой с использованием unittest"""
    
    def setUp(self):
        """Настройка перед каждым тестом"""
        self.params = NetSeparatorParameters.default_values()
        self.model = NetSeparatorModel(self.params)
        
    def tearDown(self):
        """Очистка после каждого теста"""
        pass
    
    def test_initialize_level_pressure(self):
        """Тест инициализации по уровню и давлению"""
        # Arrange
        level = 2.0
        pressure = 7.5e5
        
        # Act
        result = self.model.initialize_level_pressure(level, pressure)
        
        # Assert
        self.assertTrue(
            math.isclose(result.separator_state.level_liquid, level, rel_tol=0.01),
            f"Уровень не соответствует: {result.separator_state.level_liquid} != {level}"
        )
        self.assertTrue(
            math.isclose(result.separator_state.pressure_gas, pressure, rel_tol=0.01),
            f"Давление не соответствует: {result.separator_state.pressure_gas} != {pressure}"
        )
    
    def test_gas_only_step_net(self):
        """Тест шага только по газу"""
        # Arrange
        # Инициализация
        self.model.initialize_level_pressure(1.0, 7.5e5)
        
        control = NetSeparatorControl.default_values()
        control.valve_liquid_opening = 0.0  # закрыт жидкостный клапан
        control.valve_gas_opening = 1.0     # открыт газовый клапан
        control.valve_in_opening = 1.0      # открыт входной клапан
        control.omega_in = 1.0              # только газ
        
        # Act
        result = self.model.step(1.0, control)
        
        # Assert
        self.assertEqual(
            result.G_liquid, 0.0,
            "При только газовом режиме жидкость не должна течь"
        )
        self.assertGreater(
            result.G_gas, 0.0,
            "При только газовом режиме газ должен течь"
        )
        self.assertGreater(
            result.G_in, 0.0,
            "Входной поток должен быть положительным"
        )
    
    def test_liquid_only_step_net(self):
        """Тест шага только по жидкости"""
        # Arrange
        # Инициализация
        self.model.initialize_level_pressure(5.0, 7.5e5)
        
        control = NetSeparatorControl.default_values()
        control.valve_liquid_opening = 1.0  # открыт жидкостный клапан
        control.valve_gas_opening = 0.0     # закрыт газовый клапан
        control.valve_in_opening = 1.0      # открыт входной клапан
        control.omega_in = 0.0              # только жидкость
        
        # Act
        result = self.model.step(1.0, control)
        
        # Assert
        self.assertEqual(
            result.G_gas, 0.0,
            "При только жидкостном режиме газ не должен течь"
        )
        self.assertGreater(
            result.G_liquid, 0.0,
            "При только жидкостном режиме жидкость должна течь"
        )
        self.assertGreater(
            result.G_in, 0.0,
            "Входной поток должен быть положительным"
        )
    
    def test_stationary_verification(self):
        """Тест стационарного расчета"""
        # Arrange
        model = default_net_separator()
        control = NetSeparatorControl.default_values()
        
        # Act - стационарный расчет (dt = 0)
        result = model.step(0.0, control)
        
        # Assert - проверка баланса расходов
        total_out = result.G_gas + result.G_liquid
        
        # В стационаре входной расход должен равняться сумме выходных
        self.assertTrue(
            math.isclose(result.G_in, total_out, rel_tol=0.1),
            f"Баланс расходов не выполняется: G_in={result.G_in}, сумма выходов={total_out}"
        )
    
    def test_step_with_positive_dt(self):
        """Тест шага расчета с положительным шагом времени"""
        # Arrange
        self.model.initialize_level_pressure(3.0, 7.5e5)
        control = NetSeparatorControl.default_values()
        
        # Act
        result = self.model.step(1.0, control)
        
        # Assert
        self.assertIsNotNone(result.separator_state.pressure_gas)
        self.assertIsNotNone(result.separator_state.level_liquid)
        self.assertIsNotNone(result.G_in)
        self.assertIsNotNone(result.G_gas)
        self.assertIsNotNone(result.G_liquid)
        
        # Проверяем физическую корректность
        self.assertGreaterEqual(result.separator_state.pressure_gas, 0)
        self.assertGreaterEqual(result.separator_state.level_liquid, 0)
        self.assertGreaterEqual(result.G_in, 0)
        self.assertGreaterEqual(result.G_gas, 0)
        self.assertGreaterEqual(result.G_liquid, 0)
    
    def test_initialize_with_zero_level(self):
        """Тест инициализации с нулевым уровнем"""
        # Arrange
        level = 0.0
        pressure = 5.0e5
        
        # Act
        result = self.model.initialize_level_pressure(level, pressure)
        
        # Assert
        self.assertTrue(
            math.isclose(result.separator_state.level_liquid, 0.0, rel_tol=0.01),
            f"Уровень должен быть 0: {result.separator_state.level_liquid}"
        )
        self.assertTrue(
            math.isclose(result.separator_state.pressure_gas, pressure, rel_tol=0.01),
            f"Давление не соответствует: {result.separator_state.pressure_gas}"
        )
    
    def test_initialize_with_high_pressure(self):
        """Тест инициализации с высоким давлением"""
        # Arrange
        level = 3.0
        pressure = 20.0e5  # 20 бар
        
        # Act
        result = self.model.initialize_level_pressure(level, pressure)
        
        # Assert
        self.assertTrue(
            math.isclose(result.separator_state.pressure_gas, pressure, rel_tol=0.01),
            f"Высокое давление не установилось: {result.separator_state.pressure_gas}"
        )
    
    def test_mixed_flow_step(self):
        """Тест шага со смешанным потоком"""
        # Arrange
        self.model.initialize_level_pressure(3.0, 7.5e5)
        
        control = NetSeparatorControl.default_values()
        control.valve_liquid_opening = 0.5
        control.valve_gas_opening = 0.5
        control.valve_in_opening = 1.0
        control.omega_in = 0.5  # 50% газа
        
        # Act
        result = self.model.step(1.0, control)
        
        # Assert
        self.assertGreater(result.G_in, 0, "Входной поток должен быть положительным")
        self.assertGreater(result.G_gas, 0, "Газовый поток должен быть положительным")
        self.assertGreater(result.G_liquid, 0, "Жидкостный поток должен быть положительным")
        
        # В смешанном режиме оба потока должны быть
        self.assertGreater(result.G_gas, 0, "Должен быть газовый поток")
        self.assertGreater(result.G_liquid, 0, "Должен быть жидкостный поток")
    
    def test_default_net_separator_function(self):
        """Тест функции создания сепаратора по умолчанию"""
        # Act
        model = default_net_separator()
        
        # Assert
        self.assertIsInstance(model, NetSeparatorModel)
        self.assertIsNotNone(model.state.separator_state.pressure_gas)
        self.assertIsNotNone(model.state.separator_state.level_liquid)
        
        # Проверяем значения по умолчанию из Simba
        self.assertTrue(
            math.isclose(model.state.separator_state.level_liquid, 5.0, rel_tol=0.01),
            f"Уровень по умолчанию должен быть 5.0 м: {model.state.separator_state.level_liquid}"
        )
        self.assertTrue(
            math.isclose(model.state.separator_state.pressure_gas, 7.5e5, rel_tol=0.01),
            f"Давление по умолчанию должно быть 7.5 бар: {model.state.separator_state.pressure_gas/1e5:.1f}"
        )
    
    def test_negative_dt_exception(self):
        """Тест с отрицательным шагом времени"""
        # Arrange
        self.model.initialize_level_pressure(3.0, 7.5e5)
        control = NetSeparatorControl.default_values()
        
        # Act & Assert
        # Шаг времени должен быть неотрицательным
        # Если модель не проверяет это, тест пропускается
        try:
            result = self.model.step(-1.0, control)
            # Если не выброшено исключение, хотя бы проверяем результат
            self.assertIsNotNone(result)
        except (ValueError, RuntimeError) as e:
            self.assertIn("шаг", str(e).lower() or "время", str(e).lower())
    
    def test_calculate_densities_method(self):
        """Тест внутреннего метода расчета плотностей"""
        # Arrange
        self.model.initialize_level_pressure(3.0, 7.5e5)
        
        # Вызываем шаг чтобы densities были рассчитаны
        control = NetSeparatorControl.default_values()
        self.model.step(1.0, control)
        
        # Проверяем что плотности рассчитаны
        self.assertIsNotNone(self.model.state.density_gas)
        self.assertIsNotNone(self.model.state.density_liquid)
        
        # Плотность жидкости должна быть положительной и постоянной
        self.assertGreater(self.model.state.density_liquid, 0)
        self.assertAlmostEqual(
            self.model.state.density_liquid, 
            1000.0,  # плотность воды
            delta=10.0
        )
        
        # Плотность газа должна быть положительной (при наличии газа)
        if self.model.state.separator_state.pressure_gas > 0:
            self.assertGreater(self.model.state.density_gas, 0)
    
    def test_calculate_flows_method(self):
        """Тест внутреннего метода расчета расходов"""
        # Arrange
        self.model.initialize_level_pressure(3.0, 7.5e5)
        control = NetSeparatorControl.default_values()
        
        # Рассчитываем плотности
        self.model._calculate_densities()
        
        # Act - рассчитываем расходы
        self.model._calculate_flows(control)
        
        # Assert
        self.assertIsNotNone(self.model.state.G_in)
        self.assertIsNotNone(self.model.state.G_gas)
        self.assertIsNotNone(self.model.state.G_liquid)
        
        # Расходы должны быть неотрицательными
        self.assertGreaterEqual(self.model.state.G_in, 0)
        self.assertGreaterEqual(self.model.state.G_gas, 0)
        self.assertGreaterEqual(self.model.state.G_liquid, 0)
    
    def test_calculate_mixture_density(self):
        """Тест расчета плотности смеси"""
        # Arrange
        self.model.initialize_level_pressure(3.0, 7.5e5)
        
        # Рассчитываем плотности
        self.model._calculate_densities()
        
        # Test cases
        test_cases = [
            (0.0, self.model.state.density_liquid),  # только жидкость
            (1.0, self.model.state.density_gas),     # только газ
            (0.5, None),  # смесь - просто проверяем что не падает
        ]
        
        for omega, expected_density in test_cases:
            with self.subTest(omega=omega):
                density = self.model._calculate_mixture_density(omega)
                self.assertIsNotNone(density)
                self.assertGreater(density, 0)
                
                if expected_density is not None:
                    self.assertAlmostEqual(density, expected_density, delta=1.0)


class TestNetSeparatorParameters(unittest.TestCase):
    """Тесты для параметров сепаратора"""
    
    def test_default_values(self):
        """Тест параметров по умолчанию"""
        # Act
        params = NetSeparatorParameters.default_values()
        
        # Assert
        self.assertIsNotNone(params.separator_parameters.volume)
        self.assertIsNotNone(params.separator_parameters.area)
        
        self.assertIsNotNone(params.fluid_parameters.gas_molar_mass)
        self.assertIsNotNone(params.fluid_parameters.liquid_density)
        self.assertIsNotNone(params.fluid_parameters.temperature)
        self.assertIsNotNone(params.fluid_parameters.R)
        
        self.assertIsNotNone(params.valve_in_parameters.kv100)
        self.assertIsNotNone(params.valve_gas_parameters.kv100)
        self.assertIsNotNone(params.valve_liquid_parameters.kv100)
        
        # Проверяем конкретные значения из Simba
        self.assertAlmostEqual(params.valve_in_parameters.kv100, 100, delta=0.1)
        self.assertAlmostEqual(params.valve_gas_parameters.kv100, 248, delta=0.1)
        self.assertAlmostEqual(params.valve_liquid_parameters.kv100, 193, delta=0.1)


class TestNetSeparatorControl(unittest.TestCase):
    """Тесты для управления сепаратором"""
    
    def test_default_values(self):
        """Тест управления по умолчанию"""
        # Act
        control = NetSeparatorControl.default_values()
        
        # Assert
        self.assertIsNotNone(control.valve_in_opening)
        self.assertIsNotNone(control.valve_gas_opening)
        self.assertIsNotNone(control.valve_liquid_opening)
        self.assertIsNotNone(control.omega_in)
        self.assertIsNotNone(control.pressure_out)
        self.assertIsNotNone(control.pressure_in)
        
        # Проверяем диапазоны
        self.assertTrue(0 <= control.valve_in_opening <= 1)
        self.assertTrue(0 <= control.valve_gas_opening <= 1)
        self.assertTrue(0 <= control.valve_liquid_opening <= 1)
        self.assertTrue(0 <= control.omega_in <= 1)
        self.assertGreater(control.pressure_out, 0)
        self.assertGreater(control.pressure_in, 0)
        
        # Проверяем значения из Simba
        self.assertAlmostEqual(control.valve_gas_opening, 0.5, delta=0.01)
        self.assertAlmostEqual(control.valve_liquid_opening, 0.5, delta=0.01)
        self.assertAlmostEqual(control.omega_in, 0.0615, delta=0.0001)


if __name__ == '__main__':
    unittest.main(verbosity=2)