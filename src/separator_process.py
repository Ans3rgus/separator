# src/separator_process.py
import math
from typing import Optional
from net_separator import NetSeparatorModel, NetSeparatorParameters, NetSeparatorControl, NetSeparatorState
from pid import PIDController, PIDParameters


class SeparatorProcessControl:
    """Управление процессом сепарации с автоматикой"""
    
    def __init__(self):
        self.setpoint_pressure: Optional[float] = None  # уставка давления
        self.setpoint_level: Optional[float] = None     # уставка уровня
        self.omega_in: Optional[float] = None           # доля газа на входе
        self.pressure_out: Optional[float] = None       # давление на выходе
        self.pressure_in: Optional[float] = None        # давление на входе
    
    @staticmethod
    def default_values():
        """Значения по умолчанию"""
        control = SeparatorProcessControl()
        control.setpoint_pressure = 7.5e5
        control.setpoint_level = 5.0
        control.omega_in = 0.0615
        control.pressure_out = 7.0e5
        control.pressure_in = 8.0e5
        return control


class SeparatorProcessState:
    """Состояние процесса сепарации"""
    
    def __init__(self):
        self.net_separator_state: Optional[NetSeparatorState] = None
        self.reg_pressure_op: Optional[float] = None  # выход регулятора давления
        self.reg_level_op: Optional[float] = None     # выход регулятора уровня
        self.pressure_error: Optional[float] = None   # ошибка давления
        self.level_error: Optional[float] = None      # ошибка уровня


class SeparatorProcess:
    """Модель процесса сепарации с автоматикой"""
    
    def __init__(self, separator_params: NetSeparatorParameters,
                 pid_pressure_params: PIDParameters,
                 pid_level_params: PIDParameters):
        
        # Модель сепаратора
        self.separator_model = NetSeparatorModel(separator_params)
        
        # Регуляторы
        self.reg_pressure = PIDController(pid_pressure_params)
        self.reg_level = PIDController(pid_level_params)
        
        # Текущее состояние
        self.state = SeparatorProcessState()
        
        # Флаг инициализации
        self.initialized = False
    
    def get_separator_state(self):
        """Геттер для получения состояния сепаратора"""
        # Проверяем, есть ли метод get_state() в сепараторе
        # Если нет, возвращаем state напрямую
        if hasattr(self.separator_model, 'get_state'):
            return self.separator_model.get_state()
        else:
            # Временное решение: прямое обращение к полю
            return self.separator_model.state
    
    def _set_controller_op_bias(self, controller, bias):
        """Вспомогательный метод для установки OP_bias с проверкой наличия метода"""
        # Проверяем, есть ли метод set_op_bias в контроллере
        if hasattr(controller, 'set_op_bias'):
            controller.set_op_bias(bias)
        else:
            # Временное решение: прямое обращение к полю
            controller.parameters.OP_bias = bias
    
    def initialize(self, level_liquid: float, pressure_gas: float, 
                   control: SeparatorProcessControl):
        """
        Инициализация процесса с безударным запуском
        """
        # Инициализация сепаратора
        self.separator_model.initialize_level_pressure(level_liquid, pressure_gas)
        
        # Настройка регуляторов на безударный запуск
        # Для этого нужно рассчитать начальные положения клапанов,
        # которые соответствуют текущему состоянию сепаратора
        
        # Получаем состояние сепаратора через геттер
        sep_state = self.get_separator_state()
        
        # Расчет требуемых начальных положений клапанов
        # Для простоты используем стационарное решение
        # В реальной системе здесь должен быть расчет на основе текущего состояния
        initial_gas_valve = 0.5  # можно рассчитать точнее
        initial_liq_valve = 0.5
        
        # Настраиваем регуляторы через вспомогательный метод
        self._set_controller_op_bias(self.reg_pressure, initial_gas_valve)
        self._set_controller_op_bias(self.reg_level, initial_liq_valve)
        
        # Сбрасываем интегральные составляющие
        self.reg_pressure.reset()
        self.reg_level.reset()
        
        # Устанавливаем начальные выходы регуляторов
        # Это важно для безударного запуска
        if hasattr(self.reg_pressure, 'output'):
            self.reg_pressure.output = initial_gas_valve
        if hasattr(self.reg_level, 'output'):
            self.reg_level.output = initial_liq_valve
        
        self.initialized = True
    
    def step(self, dt: float, control: SeparatorProcessControl) -> SeparatorProcessState:
        """
        Шаг расчета процесса сепарации
        """
        if not self.initialized:
            raise RuntimeError("Процесс не инициализирован. Вызовите initialize()")
        
        # Проверяем корректность входных данных
        if dt <= 0:
            raise ValueError(f"Некорректный шаг времени: dt={dt}")
        
        # 1. Получаем текущие PV из сепаратора через геттер
        current_state = self.get_separator_state()
        pressure_pv = current_state.separator_state.pressure_gas
        level_pv = current_state.separator_state.level_liquid
        
        # 2. Вызываем регуляторы
        try:
            # Регулятор давления (управляет газовым клапаном)
            gas_opening = self.reg_pressure.step(
                PV=pressure_pv,
                SP=control.setpoint_pressure,
                dt=dt
            )
            
            # Регулятор уровня (управляет жидкостным клапаном)
            liq_opening = self.reg_level.step(
                PV=level_pv,
                SP=control.setpoint_level,
                dt=dt
            )
        except Exception as e:
            raise RuntimeError(f"Ошибка в регуляторе: {e}")
        
        # 3. Формируем управление для сепаратора
        net_control = NetSeparatorControl()
        net_control.valve_gas_opening = gas_opening
        net_control.valve_liquid_opening = liq_opening
        net_control.valve_in_opening = 1.0  # входной клапан полностью открыт
        net_control.omega_in = control.omega_in
        net_control.pressure_out = control.pressure_out
        net_control.pressure_in = control.pressure_in
        
        # 4. Вызываем шаг сепаратора
        try:
            net_state = self.separator_model.step(dt, net_control)
        except Exception as e:
            raise RuntimeError(f"Ошибка в модели сепаратора: {e}")
        
        # 5. Формируем состояние процесса
        self.state.net_separator_state = net_state
        self.state.reg_pressure_op = gas_opening
        self.state.reg_level_op = liq_opening
        self.state.pressure_error = pressure_pv - control.setpoint_pressure
        self.state.level_error = level_pv - control.setpoint_level
        
        return self.state
    
    def reset(self):
        """Сброс процесса в исходное состояние"""
        # Сбрасываем состояние процесса
        self.state = SeparatorProcessState()
        
        # Сбрасываем регуляторы
        self.reg_pressure.reset()
        self.reg_level.reset()
        
        # Сбрасываем сепаратор, если есть метод reset
        if hasattr(self.separator_model, 'reset'):
            self.separator_model.reset()
        
        # Сбрасываем флаг инициализации
        self.initialized = False