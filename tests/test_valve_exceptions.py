# tests/test_valve_exceptions_unittest.py
import unittest
import sys
import os
import math

# Добавляем путь к src для импорта
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from valve import ValveModel, ValveParameters, KvType


class TestValveExceptions(unittest.TestCase):
    """Тесты для проверки исключений в модели клапана с использованием unittest"""
    
    def setUp(self):
        """Настройка перед каждым тестом"""
        # Создаем параметры клапана по умолчанию
        self.valve_params = ValveParameters()
        self.valve_params.kv0 = 1.0
        self.valve_params.kv100 = 10.0
        self.valve_params.type = KvType.Linear
        self.valve_params.cutoff = True
        
        self.valve = ValveModel(self.valve_params)
        
        # Параметры среды для тестов расхода
        self.density = 1000.0  # кг/м³
        self.pressure_in = 1e6  # 1 МПа
        self.pressure_out = 9e5  # 0.9 МПа
    
    def test_calc_kv_negative_opening(self):
        """Тест: отрицательное открытие должно вызывать исключение"""
        with self.assertRaises(ValueError) as context:
            self.valve.calc_kv(-0.1)
        
        # Проверяем сообщение исключения
        error_msg = str(context.exception)
        self.assertIn("должна быть в диапазоне", error_msg)
        self.assertIn("-0.1", error_msg)
    
    def test_calc_kv_opening_greater_than_one(self):
        """Тест: открытие больше 1 должно вызывать исключение"""
        with self.assertRaises(ValueError) as context:
            self.valve.calc_kv(1.1)
        
        error_msg = str(context.exception)
        self.assertIn("должна быть в диапазоне", error_msg)
        self.assertIn("1.1", error_msg)
    
    def test_calc_kv_boundary_values_valid(self):
        """Тест: граничные значения 0 и 1 должны быть валидными"""
        # Открытие 0 должно работать
        kv0 = self.valve.calc_kv(0.0)
        self.assertEqual(kv0, 0.0)  # Из-за cutoff=True
        
        # Открытие 1 должно работать
        kv1 = self.valve.calc_kv(1.0)
        self.assertAlmostEqual(kv1, 10.0, delta=1e-6)  # kv100
        
        # Промежуточное значение должно работать
        kv_half = self.valve.calc_kv(0.5)
        self.assertAlmostEqual(kv_half, 5.5, delta=1e-6)  # kv0 + 0.5*(kv100-kv0)
    
    def test_get_volumetric_flow_negative_opening(self):
        """Тест: объемный расход с отрицательным открытием"""
        with self.assertRaises(ValueError) as context:
            self.valve.get_volumetric_flow(-0.5, self.density, self.pressure_in, self.pressure_out)
        
        error_msg = str(context.exception)
        self.assertIn("должна быть в диапазоне", error_msg)
    
    def test_get_volumetric_flow_opening_greater_than_one(self):
        """Тест: объемный расход с открытием > 1"""
        with self.assertRaises(ValueError) as context:
            self.valve.get_volumetric_flow(1.5, self.density, self.pressure_in, self.pressure_out)
        
        error_msg = str(context.exception)
        self.assertIn("должна быть в диапазоне", error_msg)
    
    def test_get_mass_flow_negative_opening(self):
        """Тест: массовый расход с отрицательным открытием"""
        with self.assertRaises(ValueError) as context:
            self.valve.get_mass_flow(-0.2, self.density, self.pressure_in, self.pressure_out)
        
        error_msg = str(context.exception)
        self.assertIn("должна быть в диапазоне", error_msg)
    
    def test_get_mass_flow_opening_greater_than_one(self):
        """Тест: массовый расход с открытием > 1"""
        with self.assertRaises(ValueError) as context:
            self.valve.get_mass_flow(2.0, self.density, self.pressure_in, self.pressure_out)
        
        error_msg = str(context.exception)
        self.assertIn("должна быть в диапазоне", error_msg)
    
    def test_valid_values_work_correctly(self):
        """Тест: корректные значения должны работать без исключений"""
        # Проверяем диапазон корректных значений
        for opening in [0.0, 0.1, 0.5, 0.9, 1.0]:
            try:
                kv = self.valve.calc_kv(opening)
                self.assertIsInstance(kv, float)
                self.assertGreaterEqual(kv, 0.0)
                
                # Проверяем методы расхода
                Q = self.valve.get_volumetric_flow(
                    opening, self.density, self.pressure_in, self.pressure_out
                )
                self.assertIsInstance(Q, float)
                
                G = self.valve.get_mass_flow(
                    opening, self.density, self.pressure_in, self.pressure_out
                )
                self.assertIsInstance(G, float)
                
            except Exception as e:
                self.fail(f"Корректное значение {opening} вызвало исключение: {e}")
    
    def test_negative_density(self):
        """Тест: отрицательная плотность"""
        with self.assertRaises(ValueError):
            self.valve.get_volumetric_flow(0.5, -1000.0, self.pressure_in, self.pressure_out)
    
    def test_negative_pressure(self):
        """Тест: отрицательное давление"""
        with self.assertRaises(ValueError):
            self.valve.get_volumetric_flow(0.5, self.density, -1e6, self.pressure_out)
        
        with self.assertRaises(ValueError):
            self.valve.get_volumetric_flow(0.5, self.density, self.pressure_in, -1e6)
    
    def test_zero_density(self):
        """Тест: нулевая плотность"""
        with self.assertRaises(ValueError):
            self.valve.get_volumetric_flow(0.5, 0.0, self.pressure_in, self.pressure_out)
    
    def test_reverse_pressure_flow(self):
        """Тест: обратный поток (давление на выходе больше)"""
        # При обратном давлении расход должен быть 0, но не исключение
        Q = self.valve.get_volumetric_flow(0.5, self.density, 1e5, 2e5)
        self.assertEqual(Q, 0.0)
    
    def test_different_valve_types(self):
        """Тест: разные типы клапанов с некорректными открытиями"""
        # Тестируем равнопроцентную характеристику
        self.valve_params.type = KvType.EqualPercent
        self.valve_params.kv0 = 0.1
        valve_eq = ValveModel(self.valve_params)
        
        with self.assertRaises(ValueError):
            valve_eq.calc_kv(-0.1)
        
        with self.assertRaises(ValueError):
            valve_eq.calc_kv(1.1)
        
        # Тестируем параболическую характеристику
        self.valve_params.type = KvType.Parabolic
        valve_par = ValveModel(self.valve_params)
        
        with self.assertRaises(ValueError):
            valve_par.calc_kv(-0.2)
        
        with self.assertRaises(ValueError):
            valve_par.calc_kv(1.2)
    
    def test_cutoff_disabled_validation(self):
        """Тест: валидация при отключенной отсечке"""
        self.valve_params.cutoff = False
        valve = ValveModel(self.valve_params)
        
        # Отрицательное открытие все равно должно вызывать исключение
        with self.assertRaises(ValueError):
            valve.calc_kv(-0.1)
        
        # Открытие > 1 тоже должно вызывать исключение
        with self.assertRaises(ValueError):
            valve.calc_kv(1.1)
        
        # Граничные значения должны работать
        kv0 = valve.calc_kv(0.0)
        kv1 = valve.calc_kv(1.0)
        self.assertAlmostEqual(kv0, 1.0, delta=1e-6)  # kv0
        self.assertAlmostEqual(kv1, 10.0, delta=1e-6)  # kv100


class TestValveIntegrationWithPID(unittest.TestCase):
    """Интеграционные тесты клапана с регулятором"""
    
    def setUp(self):
        """Настройка перед каждым тестом"""
        # Параметры клапана
        self.valve_params = ValveParameters()
        self.valve_params.kv0 = 1.0
        self.valve_params.kv100 = 10.0
        self.valve_params.type = KvType.Linear
        self.valve_params.cutoff = True
        self.valve = ValveModel(self.valve_params)
        
        # Параметры среды
        self.density = 1000.0
        self.pin = 1e6
        self.pout = 9e5
        
        # Импортируем ПИД-регулятор
        from pid import PIDParameters
        
        # Параметры ПИД-регулятора
        self.pid_params = PIDParameters.default_values()
        self.pid_params.Kp = 2.0
        self.pid_params.Ti = 5.0
        self.pid_params.action = 1
        self.pid_params.OP_bias = 0.0
        self.pid_params.OP_min = 0.0
        self.pid_params.OP_max = 100.0
        self.pid_params.PV_min = 0.0
        self.pid_params.PV_max = 100.0
    
    def test_pid_valve_integration_valid(self):
        """Тест интеграции ПИД-регулятора с клапаном (валидные значения)"""
        from pid import PIDController
        
        pid = PIDController(self.pid_params)
        
        # Симулируем работу системы
        for i in range(10):
            # ПИД рассчитывает управляющее воздействие
            pid_output = pid.step(PV=30.0 + i*5, SP=50.0, dt=1.0)
            
            # Конвертируем в открытие клапана (0-100% -> 0-1)
            opening = pid_output / 100.0
            
            # Проверяем что открытие корректное
            self.assertGreaterEqual(opening, 0.0)
            self.assertLessEqual(opening, 1.0)
            
            # Клапан должен работать без исключений
            try:
                flow = self.valve.get_volumetric_flow(opening, self.density, self.pin, self.pout)
                self.assertGreaterEqual(flow, 0.0)
            except ValueError as e:
                self.fail(f"ПИД выдал некорректное открытие {opening}: {e}")
    
    def test_pid_valve_saturation(self):
        """Тест насыщения ПИД-регулятора и его влияние на клапан"""
        from pid import PIDController
        
        # ПИД с большим усилением
        self.pid_params.Kp = 10.0
        pid = PIDController(self.pid_params)
        
        # Большая ошибка
        pid_output = pid.step(PV=0.0, SP=100.0, dt=1.0)
        
        # Выход должен быть ограничен
        self.assertLessEqual(pid_output, 100.0)
        
        # Конвертируем в открытие
        opening = pid_output / 100.0
        
        # Клапан должен принимать это значение
        self.assertTrue(0.0 <= opening <= 1.0)
        
        # Проверяем что клапан работает
        flow = self.valve.get_volumetric_flow(opening, self.density, self.pin, self.pout)
        self.assertGreaterEqual(flow, 0.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)