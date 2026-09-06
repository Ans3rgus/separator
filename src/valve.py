import math
from enum import Enum
from typing import Optional


class KvType(Enum):
    """Тип расходных характеристик клапанов"""
    Linear = 1
    EqualPercent = 2
    Parabolic = 3


class ValveParameters:
    """Параметры арматуры для гидравлических расчетов"""
    
    def __init__(self):
        # Пропускная способность при нулевом открытии, м³/ч
        self.kv0: Optional[float] = None
        # Пропускная способность при полном открытии, м³/ч
        self.kv100: Optional[float] = None
        # Тип характеристики (enum KvType)
        self.type: Optional[KvType] = None
        # Отсечка расхода при нулевом открытии
        self.cutoff: Optional[bool] = None

    @staticmethod
    def default_values():
        """Создает объект с параметрами клапана по умолчанию"""
        valve_parameters = ValveParameters()
        valve_parameters.kv0 = 1e-3
        valve_parameters.kv100 = 10
        valve_parameters.type = KvType.EqualPercent
        valve_parameters.cutoff = True
        return valve_parameters

    def validate(self):
        """Проверка корректности параметров клапана"""
        errors = []
        
        if self.kv0 is None or self.kv100 is None:
            errors.append("Kv0 и Kv100 должны быть заданы")
        elif self.kv0 < 0 or self.kv100 < 0:
            errors.append("Kv0 и Kv100 должны быть неотрицательными")
        elif self.kv0 > self.kv100:
            errors.append("Kv0 не может быть больше Kv100")
            
        if self.type is None:
            errors.append("Тип характеристики должен быть задан")
            
        if self.cutoff is None:
            errors.append("Параметр cutoff должен быть задан")
        
        if errors:
            raise ValueError("Ошибки в параметрах клапана: " + "; ".join(errors))
        
        return True


class ValveModel:
    """Модель клапана для расчета расходных характеристик"""
    
    def __init__(self, valve_parameters: ValveParameters):
        """
        Args:
            valve_parameters: параметры клапана
        """
        valve_parameters.validate()
        self.parameters = valve_parameters
    
    def calc_kv(self, opening: float) -> float:
        """
        Расчет Kv при заданном открытии клапана
        
        Args:
            opening: степень открытия клапана [0, 1]
            
        Returns:
            Пропускная способность Kv, м³/ч
            
        Raises:
            ValueError: при некорректных параметрах
        """
        # Валидация входных параметров
        if opening < 0 or opening > 1:
            raise ValueError("Степень открытия должна быть в диапазоне [0, 1]")
        
        # Нормализация открытия в диапазон [0, 1]
        opening = max(0.0, min(1.0, opening))
        
        # Проверка отсечки
        if math.isclose(opening, 0.0) and self.parameters.cutoff:
            return 0.0
        
        kv_type = self.parameters.type
        kv0 = self.parameters.kv0
        kv100 = self.parameters.kv100
        
        if kv_type == KvType.Linear:
            # Линейная характеристика: Kv = kv0 + opening * (kv100 - kv0)
            return kv0 + opening * (kv100 - kv0)
        
        elif kv_type == KvType.EqualPercent:
            # Равнопроцентная характеристика: Kv = kv0 * R^(opening)
            # где R - диапазон регулирования (kv100/kv0)
            if kv0 <= 0:
                raise ValueError("kv0 должен быть положительным для равнопроцентной характеристики")
            R = kv100 / kv0
            return kv0 * (R ** opening)
        
        elif kv_type == KvType.Parabolic:
            # Параболическая характеристика: Kv = kv0 + (kv100 - kv0) * opening^2
            return kv0 + (kv100 - kv0) * (opening ** 2)
        
        else:
            raise NotImplementedError(f'Не поддерживаемый тип расходной характеристики: {kv_type}')
    
    def get_volumetric_flow(self, opening: float, density: float, 
                           pressure_in: float, pressure_out: float) -> float:
        """
        Расчет объемного расхода по ГОСТ Р 55508-2013
        
        Args:
            opening: степень открытия клапана [0, 1]
            density: плотность среды, кг/м³
            pressure_in: давление на входе, Па
            pressure_out: давление на выходе, Па
            
        Returns:
            Объемный расход, м³/с
        """
        # Валидация входных параметров
        if density <= 0:
            raise ValueError("Плотность должна быть положительной")
        if pressure_in < 0 or pressure_out < 0:
            raise ValueError("Давления не могут быть отрицательными")
        
        # Расчет перепада давления
        dp = pressure_in - pressure_out
        
        # Проверка на обратный поток
        if dp < 0:
            return 0.0
        
        # Расчет Kv для текущего открытия
        kv = self.calc_kv(opening)
        
        # Формула из ГОСТ Р 55508-2013: Q = (Kv / 35700) * sqrt(dp / density)
        # Гост не действует, изменено на международный стандарт
        # где Kv в м³/ч, Q в м³/с
        volumetric_flow = (kv / 35700.0) * math.sqrt(dp / density)
        return volumetric_flow
    
    def get_mass_flow(self, opening: float, density: float, 
                     pressure_in: float, pressure_out: float) -> float:
        """
        Расчет массового расхода: G = ρ * Q
        
        Args:
            opening: степень открытия клапана [0, 1]
            density: плотность среды, кг/м³
            pressure_in: давление на входе, Па
            pressure_out: давление на выходе, Па
            
        Returns:
            Массовый расход, кг/с
        """
        volumetric_flow = self.get_volumetric_flow(opening, density, pressure_in, pressure_out)
        mass_flow = density * volumetric_flow
        return mass_flow


class ValveTestData:
    """Данные для тестирования модели клапана"""
    
    def __init__(self):
        self.Pin: Optional[float] = None  # Давление на входе, Па
        self.Pout: Optional[float] = None  # Давление на выходе, Па
        self.valve_opening: Optional[float] = None  # Степень открытия [0, 1]
        self.T: Optional[float] = None  # Температура, К
        self.fluid_gas_mass_fraction: Optional[float] = None  # Массовая доля газа
        self.fluid_density_simba: Optional[float] = None  # Плотность из Simba, кг/м³
        self.kv100_simba: Optional[float] = None  # Kv100 из Simba
        self.kv0_simba: Optional[float] = None  # Kv0 из Simba
        self.Q_simba: Optional[float] = None  # Объемный расход из Simba, м³/с
        self.G_simba: Optional[float] = None  # Массовый расход из Simba, кг/с


class ValveTestDataFactory:
    """Фабрика для создания тестовых данных клапана"""
    
    @staticmethod
    def mix_valve() -> ValveTestData:
        """Тестовые данные для клапана со смесью газ-жидкость"""
        result = ValveTestData()
        result.Pin = 8e5
        result.Pout = 7.5e5
        result.valve_opening = 1.0
        result.T = 293
        result.fluid_density_simba = 79.011004328319
        result.fluid_gas_mass_fraction = 0.061521056745191
        
        Q0_simba = 0.01
        result.kv100_simba = 100
        result.kv0_simba = result.kv100_simba * Q0_simba
        result.Q_simba = 0.06984731668432
        result.G_simba = 5.5187066408666

        return result

    @staticmethod
    def gas_valve() -> ValveTestData:
        """Тестовые данные для газового клапана"""
        result = ValveTestData()
        result.Pin = 7.5e5
        result.Pout = 7.0e5
        result.valve_opening = 0.5
        result.T = 293
        result.fluid_density_simba = 4.9383603347308
        result.fluid_gas_mass_fraction = 1.0
        
        Q0_simba = 0.01
        result.kv100_simba = 248
        result.kv0_simba = result.kv100_simba * Q0_simba
        result.Q_simba = 0.06928815740051
        result.G_simba = 0.34216988817328

        return result
    
    @staticmethod
    def liq_valve() -> ValveTestData:
        result = ValveTestData()
        result.Pin = 797231.92990662
        result.Pout = 700000.0
        result.valve_opening = 0.5
        result.T = 293
        result.fluid_density_simba = 1000
        result.fluid_gas_mass_fraction = 0.0
        
        result.kv100_simba = 192.5
        result.kv0_simba = 192.5 * 0.01
        
        # Правильные значения из Simba:
        result.Q_simba = 0.005317003
        result.G_simba = 5.317003

        return result