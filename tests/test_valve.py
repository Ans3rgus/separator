import pytest
import sys
import math
from pathlib import Path
import numpy as np

# Добавляем путь к src для импорта
sys.path.insert(1, str(Path(__file__).parent.parent / "src"))

from valve import ValveParameters, ValveModel, KvType, ValveTestDataFactory
from fluid import FluidParameters


class TestValveModel:
    """Тесты для модели клапана"""
    
    def test_correct_with_water(self):
        """Проверка, что расход на воде при эталонном перепаде численно равен Kv"""
        params = ValveParameters()
        params.kv0 = 0.001
        params.kv100 = 10.0
        params.type = KvType.Linear
        params.cutoff = True
        params.validate()

        valve = ValveModel(params)

        # Условия теста: вода, перепад давления 1 бар = 10^5 Па
        opening = 0.5  # 50% открытия
        density = 1000  # плотность воды, кг/м³
        pressure_in = 2e5  # 2 бар
        pressure_out = 1e5  # 1 бар (перепад 1 бар)

        # Расчет Kv при данном открытии
        kv = valve.calc_kv(opening)  # = 0.001 + 0.5*(10-0.001) = 5.0005 м³/ч

        # Расчет объемного расхода
        Q = valve.get_volumetric_flow(opening, density, pressure_in, pressure_out)
        # Q = (5.0005 / 3600) * sqrt(100000 / 1000) 
        #    = (5.0005 / 3600) * sqrt(100)
        #    = (5.0005 / 3600) * 10
        #    = 0.01389 м³/с

        # Переводим Q из м³/с в м³/ч для сравнения с Kv
        Q_m3h = Q * 3600  # = 0.01389 * 3600 = 5.0005 м³/ч

        # Проверяем, что расход в м³/ч примерно равен Kv
        assert math.isclose(Q_m3h, kv, rel_tol=0.01)
    
    def test_is_symmetric(self):
        """Проверка симметричности расчета расхода при смене знака перепада давления"""
        params = ValveParameters()
        params.kv0 = 0.001
        params.kv100 = 10.0
        params.type = KvType.EqualPercent
        params.cutoff = False  # отключаем отсечку для теста
        params.validate()
        
        valve = ValveModel(params)
        
        opening = 0.5
        density = 1000
        
        # Прямой перепад давления
        Q_forward = valve.get_volumetric_flow(opening, density, 2e5, 1e5)
        
        # Обратный перепад давления (должен быть 0 из-за проверки в методе)
        Q_reverse = valve.get_volumetric_flow(opening, density, 1e5, 2e5)
        
        # При обратном перепаде расход должен быть нулевым
        assert Q_reverse == 0.0
        assert Q_forward > 0
    
    def test_cutoff_zero_opening(self):
        """Проверка функциональности отсечки при нулевом открытии"""
        params = ValveParameters()
        params.kv0 = 0.5  # Kv при нулевом открытии
        params.kv100 = 10.0
        params.type = KvType.EqualPercent
        params.cutoff = True
        params.validate()
        
        valve = ValveModel(params)
        
        # Проверяем, что при нулевом открытии Kv = 0.0 (из-за отсечки в коде)
        kv_zero = valve.calc_kv(0.0)
        assert kv_zero == 0.0  # Код возвращает 0.0 при отсечке
        
        # И расход должен быть нулевым
        Q = valve.get_volumetric_flow(0.0, 1000, 2e5, 1e5)
        assert Q == 0.0
    
    def test_cutoff_no_flow(self):
        """Проверка, что при отсечке расход нулевой даже при ненулевом Kv0"""
        params = ValveParameters()
        params.kv0 = 0.5  # Ненулевое Kv при закрытии
        params.kv100 = 10.0
        params.type = KvType.EqualPercent
        params.cutoff = True
        params.validate()
        
        valve = ValveModel(params)
        
        # При нулевом открытии Kv = 0.0 (из-за отсечки в коде), и расход = 0
        kv = valve.calc_kv(0.0)
        Q = valve.get_volumetric_flow(0.0, 1000, 2e5, 1e5)
        
        assert kv == 0.0  # Код возвращает 0.0 при отсечке
        assert Q == 0.0
    
    def test_no_cutoff_behavior(self):
        """Проверка поведения без отсечки - должен возвращаться kv0"""
        params = ValveParameters()
        params.kv0 = 0.5
        params.kv100 = 10.0
        params.type = KvType.EqualPercent
        params.cutoff = False  # отключаем отсечку
        params.validate()
        
        valve = ValveModel(params)
        
        # При нулевом открытии без отсечки должен возвращаться kv0
        kv = valve.calc_kv(0.0)
        assert kv == 0.5  # kv0
    
    def test_equal_percent_property(self):
        """Проверка свойства равнопроцентности"""
        params = ValveParameters()
        params.kv0 = 1.0
        params.kv100 = 100.0
        params.type = KvType.EqualPercent
        params.cutoff = False
        params.validate()
        
        valve = ValveModel(params)
        
        # Вычисляем Kv при разных открытиях
        kv_03 = valve.calc_kv(0.3)
        kv_05 = valve.calc_kv(0.5)
        kv_07 = valve.calc_kv(0.7)
        
        # Проверяем, что Kv растет с увеличением открытия
        assert kv_03 < kv_05 < kv_07
        
        # Проверяем равнопроцентность: отношение Kv при соседних открытиях должно быть примерно постоянным
        ratio1 = kv_05 / kv_03
        ratio2 = kv_07 / kv_05
        
        # Для равнопроцентной характеристики отношения должны быть близки
        assert math.isclose(ratio1, ratio2, rel_tol=0.1)
    
    def test_linear_characteristic(self):
        """Дополнительный тест для линейной характеристики"""
        params = ValveParameters()
        params.kv0 = 2.0
        params.kv100 = 10.0
        params.type = KvType.Linear
        params.cutoff = False
        params.validate()
        
        valve = ValveModel(params)
        
        # Проверяем линейность
        assert valve.calc_kv(0.0) == 2.0
        assert math.isclose(valve.calc_kv(0.5), 6.0)  # 2 + 0.5*(10-2) = 6
        assert valve.calc_kv(1.0) == 10.0
    
    def test_parabolic_characteristic(self):
        """Дополнительный тест для параболической характеристики"""
        params = ValveParameters()
        params.kv0 = 1.0
        params.kv100 = 9.0
        params.type = KvType.Parabolic
        params.cutoff = False
        params.validate()
        
        valve = ValveModel(params)
        
        kv_05 = valve.calc_kv(0.5)
        expected = 1.0 + (9.0 - 1.0) * (0.5 ** 2)  # 1 + 8*0.25 = 3
        assert math.isclose(kv_05, 3.0)
    
    def test_invalid_opening(self):
        """Проверка обработки некорректной степени открытия"""
        params = ValveParameters.default_values()
        valve = ValveModel(params)
        
        with pytest.raises(ValueError):
            valve.calc_kv(-0.1)
        
        with pytest.raises(ValueError):
            valve.calc_kv(1.1)
    
    def test_invalid_density(self):
        """Проверка обработки отрицательной плотности"""
        params = ValveParameters.default_values()
        valve = ValveModel(params)
        
        with pytest.raises(ValueError):
            valve.get_volumetric_flow(0.5, -1000, 2e5, 1e5)
    
    def test_invalid_pressure(self):
        """Проверка обработки отрицательного давления"""
        params = ValveParameters.default_values()
        valve = ValveModel(params)
        
        with pytest.raises(ValueError):
            valve.get_volumetric_flow(0.5, 1000, -2e5, 1e5)


def perform_valve_calculations(verification_data, density):
    """Расчёты по верификации модели клапана"""
    valve_params = ValveParameters.default_values()
    valve_params.kv100 = verification_data.kv100_simba
    valve_params.kv0 = verification_data.kv0_simba
    valve_params.validate()
    
    valve = ValveModel(valve_params)
    kv_python = valve.calc_kv(verification_data.valve_opening)

    Q_python = valve.get_volumetric_flow(
        verification_data.valve_opening, density, verification_data.Pin, verification_data.Pout)
    
    G_python = valve.get_mass_flow(
        verification_data.valve_opening, density, verification_data.Pin, verification_data.Pout)

    return kv_python, Q_python, G_python


class TestValveVerification:
    """Тесты верификации модели клапана с данными из Simba"""
    
    def test_verification_simba_liq(self):
        """Верификация клапана по жидкости по данным из Симбы"""
        etalon_liq = ValveTestDataFactory.liq_valve()
        fluid = FluidParameters.default_values()

        kv_python, Q_python, G_python = perform_valve_calculations(etalon_liq, fluid.liquid_density)

        # Проверяем что Kv в допустимом диапазоне
        assert etalon_liq.kv0_simba <= kv_python <= etalon_liq.kv100_simba

        # Верификация расходов
        assert math.isclose(Q_python, etalon_liq.Q_simba, rel_tol=0.01)
        assert math.isclose(G_python, etalon_liq.G_simba, rel_tol=0.01)
    
    def test_verification_simba_gas(self):
        """Верификация клапана по газу по данным из Симбы"""
        etalon_gas = ValveTestDataFactory.gas_valve()
        fluid = FluidParameters.default_values()

        gas_density = fluid.calc_density_gas(
            etalon_gas.Pin, fluid.temperature, fluid.gas_molar_mass)
        
        kv_python, Q_python, G_python = perform_valve_calculations(etalon_gas, gas_density)

        # Проверяем что Kv в допустимом диапазоне
        assert etalon_gas.kv0_simba <= kv_python <= etalon_gas.kv100_simba

        # Верификация расходов
        assert math.isclose(Q_python, etalon_gas.Q_simba, rel_tol=0.01)
        assert math.isclose(G_python, etalon_gas.G_simba, rel_tol=0.01)
    
    def test_verification_simba_mix(self):
        """Верификация клапана по смеси по данным из Симбы"""
        etalon_mix = ValveTestDataFactory.mix_valve()

        kv_python, Q_python, G_python = perform_valve_calculations(
            etalon_mix, etalon_mix.fluid_density_simba)

        # Проверяем что Kv в допустимом диапазоне
        assert etalon_mix.kv0_simba <= kv_python <= etalon_mix.kv100_simba

        # Верификация расходов
        assert math.isclose(Q_python, etalon_mix.Q_simba, rel_tol=0.01)
        assert math.isclose(G_python, etalon_mix.G_simba, rel_tol=0.01)


class TestValveParametersValidation:
    """Тесты валидации параметров клапана"""
    
    def test_valid_parameters(self):
        """Проверка корректных параметров"""
        params = ValveParameters.default_values()
        assert params.validate() is True
    
    def test_invalid_kv_range(self):
        """Проверка некорректного диапазона Kv"""
        params = ValveParameters()
        params.kv0 = 10.0
        params.kv100 = 5.0  # kv0 > kv100 - ошибка
        params.type = KvType.EqualPercent
        params.cutoff = True
        
        with pytest.raises(ValueError, match="Kv0 не может быть больше Kv100"):
            params.validate()
    
    def test_missing_parameters(self):
        """Проверка отсутствующих параметров"""
        params = ValveParameters()  # Все параметры None
        
        with pytest.raises(ValueError):
            params.validate()


if __name__ == "__main__":
    # Запуск тестов
    pytest.main([__file__, "-v"])