from separator import SeparatorModel, SeparatorParameters, SeparatorState
from fluid import FluidParameters
from valve import ValveModel, ValveParameters, KvType


class NetSeparatorParameters:
    """Постоянные параметры сепаратора с обвязкой"""
    
    def __init__(self):
        self.separator_parameters = SeparatorParameters()
        self.fluid_parameters = FluidParameters()
        self.valve_in_parameters = ValveParameters()
        self.valve_gas_parameters = ValveParameters()
        self.valve_liquid_parameters = ValveParameters()
    
    @staticmethod
    def default_values():
        """Параметры по умолчанию согласно данным из Simba"""
        params = NetSeparatorParameters()
        
        # Параметры сепаратора
        params.separator_parameters.volume = 100
        params.separator_parameters.area = 10
        
        # Параметры флюида
        params.fluid_parameters.gas_molar_mass = 16e-3
        params.fluid_parameters.liquid_density = 1000
        params.fluid_parameters.temperature = 293  # 20°C
        params.fluid_parameters.R = 8.314
        
        # Параметры клапанов из данных:
        # Входной клапан (смесь)
        params.valve_in_parameters.kv100 = 100  # из данных для смеси
        params.valve_in_parameters.kv0 = 100 * 0.01
        params.valve_in_parameters.type = KvType.EqualPercent
        params.valve_in_parameters.cutoff = True
        
        # Газовый клапан
        params.valve_gas_parameters.kv100 = 248  # из данных для газа
        params.valve_gas_parameters.kv0 = 248 * 0.01
        params.valve_gas_parameters.type = KvType.EqualPercent
        params.valve_gas_parameters.cutoff = True
        
        # Жидкостный клапан
        params.valve_liquid_parameters.kv100 = 193  # из данных для жидкости
        params.valve_liquid_parameters.kv0 = 193 * 0.01
        params.valve_liquid_parameters.type = KvType.EqualPercent
        params.valve_liquid_parameters.cutoff = True
        
        return params


class NetSeparatorState:
    """Состояние сепаратора с обвязкой"""
    
    def __init__(self):
        self.separator_state = SeparatorState()
        # Дополнительные результаты расчетов
        self.density_gas = None
        self.density_liquid = None
        self.G_in = None  # расход на входе
        self.G_gas = None  # расход газа
        self.G_liquid = None  # расход жидкости


class NetSeparatorControl:
    """Управление сепаратором с обвязкой"""
    
    def __init__(self):
        self.valve_in_opening = None  # положение входного клапана
        self.valve_gas_opening = None  # положение газового клапана
        self.valve_liquid_opening = None  # положение жидкостного клапана
        self.omega_in = None  # доля газа на входе
        self.pressure_out = None  # давление на выходе клапанов
        self.pressure_in = None  # давление на входе входного клапана
    
    @staticmethod
    def default_values():
        """Параметры управления по умолчанию из данных Simba"""
        control = NetSeparatorControl()
        control.valve_in_opening = 1.0  # полностью открыт для смеси
        control.valve_gas_opening = 0.5  # 50% открытия
        control.valve_liquid_opening = 0.5  # 50% открытия
        control.omega_in = 0.0615  # 6.15% газа на входе (из данных)
        control.pressure_out = 7.0e5  # 700 кПа на выходе
        control.pressure_in = 8.0e5  # 800 кПа на входе (из данных)
        return control


class NetSeparatorModel:
    """Модель сепаратора с обвязкой"""
    
    def __init__(self, parameters: NetSeparatorParameters):
        self.parameters = parameters
        
        # Создаем экземпляры моделей
        initial_state = SeparatorState.default_values()
        self.separator_model = SeparatorModel(
            parameters.fluid_parameters,
            parameters.separator_parameters,
            initial_state
        )
        
        # Создаем модели клапанов
        self.valve_in_model = ValveModel(parameters.valve_in_parameters)
        self.valve_gas_model = ValveModel(parameters.valve_gas_parameters)
        self.valve_liquid_model = ValveModel(parameters.valve_liquid_parameters)
        
        self.state = NetSeparatorState()
        self.state.separator_state = initial_state
    
    def initialize_level_pressure(self, level_liquid: float, pressure_gas: float):
        """Инициализация по уровню и давлению"""
        separator_state = self.separator_model.initialize_level_pressure(level_liquid, pressure_gas)
        self.state.separator_state = separator_state
        return self.state
    
    def step(self, dt: float, control: NetSeparatorControl):
        """Шаг расчета сепаратора с обвязкой"""
        # Расчет плотностей
        self._calculate_densities()
        
        # Расчет расходов через клапаны
        self._calculate_flows(control)
        
        # Шаг расчета сепаратора
        self.separator_model.step(
            dt,
            control.omega_in,
            self.state.G_in,
            self.state.G_gas,
            self.state.G_liquid
        )
        
        # Обновляем состояние
        self.state.separator_state = self.separator_model.state
        
        return self.state
    
    def _calculate_densities(self):
        """Расчет плотностей газа и жидкости"""
        # Плотность жидкости постоянная
        self.state.density_liquid = self.parameters.fluid_parameters.liquid_density
        
        # Плотность газа из уравнения состояния
        if self.state.separator_state.volume_gas > 0 and self.state.separator_state.pressure_gas > 0:
            pressure = self.state.separator_state.pressure_gas
            temperature = self.parameters.fluid_parameters.temperature
            molar_mass = self.parameters.fluid_parameters.gas_molar_mass
            R = self.parameters.fluid_parameters.R
            
            self.state.density_gas = pressure * molar_mass / (R * temperature)
        else:
            self.state.density_gas = 0
    
    def _calculate_flows(self, control: NetSeparatorControl):
        """Расчет расходов через клапаны"""
        # Давление в сепараторе
        p_sep_gas = self.state.separator_state.pressure_gas
        p_sep_liquid = self.state.separator_state.pressure_liquid
        
        print(f"DEBUG: p_sep_gas={p_sep_gas}, p_sep_liquid={p_sep_liquid}")
        print(f"DEBUG: density_gas={self.state.density_gas}, density_liquid={self.state.density_liquid}")
        
        # Расход через газовый клапан
        self.state.G_gas = self.valve_gas_model.get_mass_flow(
            control.valve_gas_opening,
            self.state.density_gas,
            p_sep_gas,
            control.pressure_out
        )
        
        # Расход через жидкостный клапан
        self.state.G_liquid = self.valve_liquid_model.get_mass_flow(
            control.valve_liquid_opening,
            self.state.density_liquid,
            p_sep_liquid,
            control.pressure_out
        )
        
        # Расход через входной клапан
        density_mix = self._calculate_mixture_density(control.omega_in)
        
        self.state.G_in = self.valve_in_model.get_mass_flow(
            control.valve_in_opening,
            density_mix,
            control.pressure_in,  # давление на входе входного клапана
            p_sep_gas  # давление в сепараторе
        )
        
        print(f"DEBUG: G_in={self.state.G_in}, G_gas={self.state.G_gas}, G_liquid={self.state.G_liquid}")
    
    def _calculate_mixture_density(self, omega: float):
        """Расчет плотности смеси"""
        if omega == 0:
            return self.state.density_liquid
        elif omega == 1:
            return self.state.density_gas
        else:
            # Для смеси используем формулу: 1/ρ_см = ω/ρ_г + (1-ω)/ρ_ж
            if self.state.density_gas > 0 and self.state.density_liquid > 0:
                return 1.0 / (omega / self.state.density_gas + (1 - omega) / self.state.density_liquid)
            else:
                return self.state.density_liquid


def default_net_separator():
    """Создание и инициализация сепаратора с обвязкой по умолчанию"""
    params = NetSeparatorParameters.default_values()
    model = NetSeparatorModel(params)
    
    # Инициализация по данным из Simba
    level_liquid = 5.0  # 5000 мм = 5.0 м
    pressure_gas = 7.5e5  # 750 кПа
    
    model.initialize_level_pressure(level_liquid, pressure_gas)
    
    return model