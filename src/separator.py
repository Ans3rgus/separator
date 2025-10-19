class SeparatorParameters:
    """Постоянные параметры сепаратора"""

    def __init__(self):
        self.volume = None
        self.area = None

    def default_values():
        separator_parameters = SeparatorParameters()
        separator_parameters.volume = 100
        separator_parameters.area = 10
        return separator_parameters
    
class SeparatorState:
    """Состояние сепаратора"""

    def __init__(self):
        # Масса газа (кг)
        self.mass_gas = None
        # Масса жидкости (кг)
        self.mass_liquid = None
        # Объём жидкости м^3
        self.volume_liquid = None
        # Объём газа м^3
        self.volume_gas = None
        # Давление газа (Па)
        self.pressure_gas = None
        # Давление жидкости (Па)
        self.pressure_liquid = None
        # Уровень жидкости (м)
        self.level_liquid = None

    def default_values():
        state = SeparatorState()
        state.mass_gas = 0
        state.mass_liquid = 0
        state.volume_liquid = 0
        state.volume_gas = 0
        state.pressure_gas = 0
        state.pressure_liquid = 0
        state.level_liquid = 0
        return state

class FluidParameters:
    """Параметры флюида"""
    def __init__(self):
        # Молярная масса газа
        self.gas_molar_mass = None
        # Плотность жидкости
        self.liquid_density = None
        # Температура жидкости
        self.temperature = None
        # Универсальная газовая постоянная
        self.R = None
    
    def default_values():
        fluid = FluidParameters()
        fluid.gas_molar_mass = 16e-3
        fluid.liquid_density = 1000
        fluid.temperature = 300
        fluid.R = 8.314
        return fluid
    
class SeparatorModel:
    """Модель сепаратора"""
    def __init__(self, fluid, parameters, initial_state):
        """Запоминает параметры флюид и сепаратора, задает начальное состояние"""
        self.fluid = fluid
        self.parameters = parameters
        self.state = initial_state

    def step(self, dt, omegamix, Gin_mix, Ggas, Gliquid):
        """Вычисляет шаг расчёта сепаратора, обновляет состояние self.state, возвращает его как результат"""
        # Материальный баланс сепаратора
        Gin_gas = Gin_mix * omegamix
        Gin_liquid = Gin_mix * (1 - omegamix)

        # Обновляем массы
        delta_mass_gas = (Gin_gas - Ggas) * dt
        self.state.mass_gas = max(0, self.state.mass_gas + delta_mass_gas)

        delta_mass_liquid = (Gin_liquid - Gliquid) * dt
        self.state.mass_liquid = max(0, self.state.mass_liquid + delta_mass_liquid)

        # Обновление объема жидкости (плотность постоянная)
        self.state.volume_liquid = self.state.mass_liquid / self.fluid.liquid_density
        
        # Объём газа
        if self.state.mass_gas > 0:
            # Есть газ - рассчитываем объем
            self.state.volume_gas = max(0, self.parameters.volume - self.state.volume_liquid)
        else:
            # Нет газа - объем газа 0
            self.state.volume_gas = 0
        
        # Случай переполнения
        if self.state.volume_liquid > self.parameters.volume:
            # Жидкость переполняет сепаратор
            self.state.volume_liquid = self.parameters.volume
            self.state.volume_gas = 0
            self.state.mass_liquid = self.state.volume_liquid * self.fluid.liquid_density
            self.state.mass_gas = 0
            self.state.pressure_gas = 0
        
        # Давление газа
        if self.state.volume_gas > 0 and self.state.mass_gas > 0:
            self.state.pressure_gas = (self.state.mass_gas / (self.fluid.gas_molar_mass * self.state.volume_gas)) * (self.fluid.R * self.fluid.temperature)
        else:
            self.state.pressure_gas = 0
        
        # Уровень жидкости
        if self.parameters.area > 0:
            self.state.level_liquid = self.state.volume_liquid / self.parameters.area
        else:
            self.state.level_liquid = 0

        # Давление жидкости
        self.state.pressure_liquid = self.state.pressure_gas + self.fluid.liquid_density * self.state.level_liquid * 10

        return self.state