class FluidParameters:
    """Параметры флюида для расчетов гидродинамики"""
    
    def __init__(self):
        # Молярная масса газа, кг/моль
        self.gas_molar_mass = None
        # Плотность жидкости, кг/м³
        self.liquid_density = None
        # Температура, К
        self.temperature = None
        # Универсальная газовая постоянная, Дж/(моль·К)
        self.R = None
    
    @staticmethod
    def default_values():
        """Создает объект с параметрами по умолчанию"""
        fluid = FluidParameters()
        fluid.gas_molar_mass = 16e-3  # метан
        fluid.liquid_density = 1000   # вода
        fluid.temperature = 293       # 20°C
        fluid.R = 8.314               # универсальная газовая постоянная
        return fluid

    @staticmethod
    def calc_density_mix(G1, G2, density1, density2):
        """
        Расчет плотности смеси двух флюидов
        
        Args:
            G1: массовый расход первого флюида, кг/с
            G2: массовый расход второго флюида, кг/с  
            density1: плотность первого флюида, кг/м³
            density2: плотность второго флюида, кг/м³
            
        Returns:
            Плотность смеси, кг/м³
        """
        if G1 + G2 == 0:
            return 0.0
        
        # Проверка корректности входных параметров
        if density1 <= 0 or density2 <= 0:
            raise ValueError("Плотности флюидов должны быть положительными")
        if G1 < 0 or G2 < 0:
            raise ValueError("Массовые расходы не могут быть отрицательными")
        
        # Формула для плотности смеси: ρ_см = (G1 + G2) / (G1/ρ1 + G2/ρ2)
        density_mix = (G1 + G2) / (G1 / density1 + G2 / density2)
        return density_mix

    @staticmethod
    def calc_density_gas(pressure, temperature, molar_mass, R=8.314):
        """
        Расчет плотности газа по уравнению Менделеева-Клапейрона
        
        Args:
            pressure: давление газа, Па
            temperature: температура газа, К
            molar_mass: молярная масса газа, кг/моль
            R: газовая постоянная, Дж/(моль·К)
            
        Returns:
            Плотность газа, кг/м³
            
        Raises:
            ValueError: при некорректных входных параметрах
        """
        # Валидация входных параметров
        if temperature <= 0:
            raise ValueError("Температура должна быть положительной")
        if R <= 0:
            raise ValueError("Газовая постоянная должна быть положительной")
        if molar_mass <= 0:
            raise ValueError("Молярная масса должна быть положительной")
        if pressure < 0:
            raise ValueError("Давление не может быть отрицательным")
        
        # Уравнение Менделеева-Клапейрона: ρ = P * M / (R * T)
        density = pressure * molar_mass / (R * temperature)
        return density

    def validate(self):
        """Проверка корректности установленных параметров"""
        errors = []
        
        if self.gas_molar_mass is not None and self.gas_molar_mass <= 0:
            errors.append("Молярная масса газа должна быть положительной")
        
        if self.liquid_density is not None and self.liquid_density <= 0:
            errors.append("Плотность жидкости должна быть положительной")
            
        if self.temperature is not None and self.temperature <= 0:
            errors.append("Температура должна быть положительной")
            
        if self.R is not None and self.R <= 0:
            errors.append("Газовая постоянная должна быть положительной")
        
        if errors:
            raise ValueError("Ошибки в параметрах флюида: " + "; ".join(errors))
        
        return True