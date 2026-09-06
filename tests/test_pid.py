# tests/test_pid_unittest.py
import sys
import math
import unittest
from pathlib import Path
sys.path.insert(1, str(Path(__file__).parent.parent / "src"))

from pid import PIDController, PIDParameters, PIDState


class TestPIDController(unittest.TestCase):
    """Тесты для ПИД-регулятора с использованием unittest"""
    
    def setUp(self):
        """Настройка перед каждым тестом"""
        self.params = PIDParameters.default_values()
        self.params.Kp = 1.0
        self.params.Ti = 0.0  # по умолчанию только П-составляющая
        self.params.action = 1
        self.params.OP_bias = 0.0
        self.params.OP_min = 0.0
        self.params.OP_max = 1.0
        self.params.PV_min = 0.0
        self.params.PV_max = 100.0
        self.params.Kn = 0.0
        self.params.K0 = 0.0
    
    def test_p_controller_basic(self):
        """Тест базовой работы П-регулятора"""
        # Arrange
        controller = PIDController(self.params)
        
        # Act
        pv = 40.0  # 40% от шкалы
        sp = 50.0  # 50% от шкалы
        dt = 1.0
        op = controller.step(pv, sp, dt)
        
        # Assert
        self.assertAlmostEqual(op, 0.0)
        self.assertAlmostEqual(controller.state.error, -0.1)
        self.assertAlmostEqual(controller.state.P_component, -0.1)
        self.assertAlmostEqual(controller.state.I_component, 0.0)
    
    def test_p_controller_reverse_action(self):
        """Тест П-регулятора с обратным действием"""
        # Arrange
        self.params.action = -1  # обратное действие
        controller = PIDController(self.params)
        
        # Act
        pv = 40.0
        sp = 50.0
        dt = 1.0
        op = controller.step(pv, sp, dt)
        
        # Assert
        self.assertAlmostEqual(op, 0.1, places=6)
        self.assertAlmostEqual(controller.state.error, 0.1)
        self.assertAlmostEqual(controller.state.P_component, 0.1)
    
    def test_pi_controller_basic(self):
        """Тест базовой работы ПИ-регулятора"""
        # Arrange
        self.params.Ti = 2.0  # включаем интеграл
        controller = PIDController(self.params)
        
        # Act - первый шаг
        pv = 40.0
        sp = 50.0
        dt = 1.0
        op1 = controller.step(pv, sp, dt)
        
        # Assert первого шага
        self.assertAlmostEqual(op1, 0.0)
        self.assertAlmostEqual(controller.state.error, -0.1)
        self.assertAlmostEqual(controller.state.P_component, -0.1)
        self.assertAlmostEqual(controller.state.I_component, -0.05)
        self.assertAlmostEqual(controller.state.I, -0.1)
        
        # Act - второй шаг (та же ошибка)
        op2 = controller.step(pv, sp, dt)
        
        # Assert второго шага
        self.assertAlmostEqual(op2, 0.0)
        self.assertAlmostEqual(controller.state.I, -0.2)
        self.assertAlmostEqual(controller.state.I_component, -0.1)
    
    def test_pi_controller_with_saturation(self):
        """Тест ПИ-регулятора с контролем насыщения"""
        # Arrange
        self.params.Ti = 2.0
        self.params.Kn = 0.5  # с насыщением
        controller = PIDController(self.params)
        
        # Act
        pv = 40.0
        sp = 50.0
        dt = 1.0
        op = controller.step(pv, sp, dt)
        
        # Assert
        self.assertAlmostEqual(op, 0.0)
    
    def test_controller_scaling(self):
        """Тест правильности нормировки"""
        # Arrange
        self.params.OP_min = 4.0
        self.params.OP_max = 20.0
        self.params.PV_max = 200.0
        controller = PIDController(self.params)
        
        # Act
        pv = 100.0  # 50% от шкалы
        sp = 150.0  # 75% от шкалы
        dt = 1.0
        op = controller.step(pv, sp, dt)
        
        # Assert
        self.assertAlmostEqual(op, 4.0)
    
    def test_controller_reset(self):
        """Тест сброса состояния регулятора"""
        # Arrange
        self.params.Ti = 2.0
        controller = PIDController(self.params)
        
        # Act - несколько шагов для накопления состояния
        for _ in range(3):
            controller.step(50.0, 60.0, 1.0)
        
        state_before = controller.state
        self.assertNotAlmostEqual(state_before.I, 0.0)
        
        # Сброс
        controller.reset()
        
        # Assert
        state_after = controller.state
        self.assertAlmostEqual(state_after.I, 0.0)
        self.assertAlmostEqual(state_after.error, 0.0)
        self.assertAlmostEqual(state_after.P_component, 0.0)
    
    def test_op_bias_effect(self):
        """Тест влияния смещения OP_bias"""
        # Arrange
        self.params.OP_bias = 0.5  # смещение 50%
        controller = PIDController(self.params)
        
        # Act - нулевая ошибка
        pv = 50.0
        sp = 50.0
        dt = 1.0
        op = controller.step(pv, sp, dt)
        
        # Assert
        self.assertAlmostEqual(op, 0.5)
    
    def test_p_only_mode(self):
        """Тест режима только П-составляющей (Ti = 0)"""
        # Arrange
        self.params.Kp = 2.0
        self.params.Ti = 0.0  # только П
        controller = PIDController(self.params)
        
        # Act
        op1 = controller.step(40.0, 50.0, 1.0)
        op2 = controller.step(40.0, 50.0, 1.0)  # тот же шаг
        
        # Assert - интегральная составляющая не должна накапливаться
        self.assertAlmostEqual(op1, op2)
        self.assertAlmostEqual(controller.state.I, 0.0)
        self.assertAlmostEqual(controller.state.I_component, 0.0)
    
    def test_i_only_mode(self):
        """Тест режима только И-составляющей (Kp = 0)"""
        self.params.Kp = 0.0
        self.params.Ti = 2.0
        controller = PIDController(self.params)
        
        # Положительная ошибка: PV > SP
        op1 = controller.step(60.0, 50.0, 1.0)
        op2 = controller.step(60.0, 50.0, 1.0)
        
        self.assertAlmostEqual(controller.state.P_component, 0.0)
        self.assertNotAlmostEqual(op1, op2)

    def test_output_saturation(self):
        """Тест ограничения выходного сигнала"""
        self.params.Kp = 10.0
        controller = PIDController(self.params)
        
        # Положительная ошибка: PV > SP
        op = controller.step(PV=100.0, SP=0.0, dt=1.0)
        
        self.assertTrue(0.0 <= op <= 1.0)
        self.assertAlmostEqual(op, 1.0)   # теперь будет 1.0
        
    def test_integral_windup_prevention(self):
        """Тест предотвращения насыщения интегральной составляющей"""
        # Arrange
        self.params.Ti = 1.0
        self.params.Kn = 1.0  # сильное насыщение
        controller = PIDController(self.params)
        
        # Act - много шагов с постоянной ошибкой
        ops = []
        for _ in range(10):
            op = controller.step(0.0, 100.0, 1.0)
            ops.append(op)
        
        # Assert - интеграл не должен уходить в бесконечность
        for op in ops:
            self.assertTrue(0.0 <= op <= 1.0)
    
    def test_parameter_validation(self):
        """Тест валидации параметров"""
        # Arrange
        params = PIDParameters()
        
        # Act & Assert - должны быть исключения
        with self.assertRaises(ValueError):
            controller = PIDController(params)
        
        # Проверка отдельных параметров
        params.Kp = 1.0
        params.Ti = -1.0  # отрицательное время
        params.action = 1
        params.OP_bias = 0.0
        params.OP_min = 0.0
        params.OP_max = 1.0
        params.PV_min = 0.0
        params.PV_max = 100.0
        params.Kn = 0.0
        params.K0 = 0.0
        
        with self.assertRaises(ValueError):
            controller = PIDController(params)
    
    def test_dt_zero(self):
        """Тест с нулевым шагом времени"""
        # Arrange
        controller = PIDController(self.params)
        
        # Act
        op = controller.step(40.0, 50.0, 0.0)
        
        # Assert - должен работать без ошибок
        self.assertIsInstance(op, float)


if __name__ == '__main__':
    unittest.main(verbosity=2)