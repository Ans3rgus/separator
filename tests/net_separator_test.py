import pytest
import math
import sys
from pathlib import Path
# Добавляем путь к src для импорта
sys.path.insert(1, str(Path(__file__).parent.parent / "src")) 

from net_separator import *

class TestNetSeparator:
    """Тесты для сепаратора с обвязкой"""
    
    def test_initialize_level_pressure(self):
        """Тест инициализации по уровню и давлению"""
        # Arrange
        params = NetSeparatorParameters.default_values()
        model = NetSeparatorModel(params)
        level = 2.0
        pressure = 7.5e5
        
        # Act
        result = model.initialize_level_pressure(level, pressure)
        
        # Assert
        assert math.isclose(result.separator_state.level_liquid, level, rel_tol=0.01)
        assert math.isclose(result.separator_state.pressure_gas, pressure, rel_tol=0.01)
    
    def test_gas_only_step_net(self):
        """Тест шага только по газу"""
        # Arrange
        params = NetSeparatorParameters.default_values()
        model = NetSeparatorModel(params)
        
        # Инициализация
        model.initialize_level_pressure(1.0, 7.5e5)
        
        control = NetSeparatorControl.default_values()
        control.valve_liquid_opening = 0.0  # закрыт жидкостный клапан
        control.valve_gas_opening = 1.0  # открыт газовый клапан
        control.valve_in_opening = 1.0  # открыт входной клапан
        control.omega_in = 1.0  # только газ
        
        # Act
        result = model.step(1.0, control)
        
        # Assert
        assert result.G_liquid == 0.0  # жидкость не течет
        assert result.G_gas > 0.0  # газ течет
        assert result.G_in > 0.0  # входной поток есть
    
    def test_liquid_only_step_net(self):
        """Тест шага только по жидкости"""
        # Arrange
        params = NetSeparatorParameters.default_values()
        model = NetSeparatorModel(params)
        
        # Инициализация
        model.initialize_level_pressure(5.0, 7.5e5)
        
        control = NetSeparatorControl.default_values()
        control.valve_liquid_opening = 1.0  # открыт жидкостный клапан
        control.valve_gas_opening = 0.0  # закрыт газовый клапан
        control.valve_in_opening = 1.0  # открыт входной клапан
        control.omega_in = 0.0  # только жидкость
        
        # Act
        result = model.step(1.0, control)
        
        # Assert
        assert result.G_gas == 0.0  # газ не течет
        assert result.G_liquid > 0.0  # жидкость течет
        assert result.G_in > 0.0  # входной поток есть
    
    def test_stationary_verification(self):
        # Arrange
        model = default_net_separator()
        control = NetSeparatorControl.default_values()
        
        # Act - стационарный расчет (dt = 0)
        result = model.step(0.0, control)
        
        # Assert - проверка баланса расходов
        total_out = result.G_gas + result.G_liquid
        
        # В стационаре входной расход должен равняться сумме выходных
        assert math.isclose(result.G_in, total_out, rel_tol=0.1)