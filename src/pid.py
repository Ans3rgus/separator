# src/pid.py
import math


class PIDParameters:
    """Класс параметров ПИД-регулятора"""
    
    def __init__(self):
        self.Kp = None  # коэффициент усиления П-составляющей (K_e)
        self.Ti = None  # время интегрирования И-составляющей (сек) (T_u)
        
        self.action = None  # 1 - прямое действие, -1 - обратное действие
        
        # начальное значение управляющего воздействия (OP_bias)
        self.OP_bias = None 
        
        # настройки ограничений
        self.OP_min = None  # минимальное значение OP
        self.OP_max = None  # максимальное значение OP
        
        # коэффициенты насыщения (из задания)
        self.Kn = None  # коэффициент насыщения
        self.K0 = None  # коэффициент для расчета ограничений
        
        # нормировочные коэффициенты
        self.PV_min = None
        self.PV_max = None

    def normalize_PV(self, pv):
        """Нормировка PV в диапазон [0, 1]"""
        if math.isclose(self.PV_max, self.PV_min):
            return 0.0
        return (pv - self.PV_min) / (self.PV_max - self.PV_min)
    
    def denormalize_OP(self, op_norm):
        """Денормировка OP из диапазона [0, 1]"""
        return self.OP_min + op_norm * (self.OP_max - self.OP_min)
    
    def normalize_OP(self, op):
        """Нормировка OP в диапазон [0, 1]"""
        if math.isclose(self.OP_max, self.OP_min):
            return 0.0
        return (op - self.OP_min) / (self.OP_max - self.OP_min)
    
    @staticmethod
    def default_values():
        """Создание параметров по умолчанию"""
        params = PIDParameters()
        params.Kp = 1.0
        params.Ti = 1.0
        params.action = 1
        params.OP_bias = 0.0
        params.OP_min = 0.0
        params.OP_max = 1.0  # Изменено на 1.0 для нормировки
        params.Kn = 0.0
        params.K0 = 0.0
        params.PV_min = 0.0
        params.PV_max = 100.0
        return params
    
    def validate(self):
        """Проверка корректности параметров"""
        errors = []
        
        if self.Kp is None:
            errors.append("Kp должен быть задан")
        
        if self.Ti is None:
            errors.append("Ti должен быть задан")
        elif self.Ti < 0:
            errors.append("Ti не может быть отрицательным")
            
        if self.action is None:
            errors.append("action должен быть задан")
        elif self.action not in [1, -1]:
            errors.append("action должен быть 1 или -1")
            
        if self.OP_bias is None:
            errors.append("OP_bias должен быть задан")
            
        if self.OP_min is None or self.OP_max is None:
            errors.append("OP_min и OP_max должны быть заданы")
        elif self.OP_min >= self.OP_max:
            errors.append("OP_min должен быть меньше OP_max")
            
        if self.Kn is None:
            errors.append("Kn должен быть задан")
        elif self.Kn < 0:
            errors.append("Kn не может быть отрицательным")
            
        if self.K0 is None:
            errors.append("K0 должен быть задан")
        elif self.K0 < 0:
            errors.append("K0 не может быть отрицательным")
            
        if self.PV_min is None or self.PV_max is None:
            errors.append("PV_min и PV_max должны быть заданы")
        elif self.PV_min >= self.PV_max:
            errors.append("PV_min должен быть меньше PV_max")
        
        if errors:
            raise ValueError("Ошибки в параметрах ПИД-регулятора: " + "; ".join(errors))
        
        return True


class PIDState:
    """Класс состояния ПИД-регулятора"""
    
    def __init__(self):
        self.I = None  # интегральная сумма (нормированная)
        self.OP = None  # текущее значение OP (денормированное)
        self.error = None  # текущая ошибка (нормированная)
        
        # Для отладки
        self.P_component = None  # пропорциональная составляющая (нормированная)
        self.I_component = None  # интегральная составляющая (нормированная)
        self.OP_before_saturation = None  # значение OP до ограничения (нормированное)

    def reset(self):
        """Сброс состояния регулятора"""
        self.I = 0.0
        self.OP = 0.0
        self.error = 0.0
        self.P_component = 0.0
        self.I_component = 0.0
        self.OP_before_saturation = 0.0

    @staticmethod
    def default_values():
        """Создание состояния регулятора по умолчанию"""
        state = PIDState()
        state.I = 0.0
        state.OP = 0.0
        state.error = 0.0
        state.P_component = 0.0
        state.I_component = 0.0
        state.OP_before_saturation = 0.0
        return state


class PIDController:
    """Класс ПИД-регулятора с контролем насыщения"""
    
    def __init__(self, parameters: PIDParameters):
        self.parameters = parameters
        self.parameters.validate()
        self.state = PIDState.default_values()
    
    def _calculate_integral_limits(self, error_norm, p_component):
        """
        Расчет ограничений для интегральной составляющей по формулам из задания:
        
        I_max(i) = [1.0 - (K_n * e_i + OP_max)] * T_u
        I_min(i) = -(K_n * e_i + OP_max) * T_u
        
        ВАЖНО: Без D-составляющей (K_0 * D_i = 0)
        ВАЖНО: OP_max в нормированном виде всегда 1.0
        """
        if self.parameters.Ti <= 0 or math.isclose(self.parameters.Ti, 0.0):
            return -float('inf'), float('inf')
        
        # OP_max в нормированном виде всегда 1.0
        OP_max_norm = 1.0
        
        # term = K_n * e_i + OP_max (без D-составляющей)
        term = self.parameters.Kn * error_norm + OP_max_norm
        
        # Расчет ограничений
        I_max = (1.0 - term) * self.parameters.Ti
        I_min = -term * self.parameters.Ti
        
        return I_min, I_max
    
    def step(self, PV, SP, dt):
        """
        Один шаг работы ПИД-регулятора.
        
        Args:
            PV: текущее значение регулируемого параметра
            SP: заданное значение (уставка)
            dt: шаг по времени (сек)
            
        Returns:
            OP: управляющее воздействие регулятора (денормированное)
            
        Реализует параллельную дискретную форму ПИ-регулятора:
        OP_i = K_e * e_i + (1/T_u) * I_i + OP_bias
        """
        # Нормировка PV и SP
        PV_norm = self.parameters.normalize_PV(PV)
        SP_norm = self.parameters.normalize_PV(SP)
        
        # Расчет ошибки с учетом действия регулятора
        # e_i = action * (PV_i - SP)
        error_norm = self.parameters.action * (PV_norm - SP_norm)
        self.state.error = error_norm
        
        # Пропорциональная составляющая (нормированная)
        # P = K_e * e_i
        P_norm = self.parameters.Kp * error_norm
        self.state.P_component = P_norm
        
        # Интегральная составляющая
        I_norm = 0.0
        
        # Проверяем, включена ли интегральная составляющая (Ti > 0)
        if self.parameters.Ti > 0 and not math.isclose(self.parameters.Ti, 0.0):
            # Расчет нового значения интеграла
            # I_i = I_{i-1} + e_i * dt
            I_new = self.state.I + error_norm * dt
            
            # Расчет ограничений для интеграла
            I_min, I_max = self._calculate_integral_limits(error_norm, P_norm)
            
            # Применение ограничений насыщения
            # I_i = min(max(I_i, I_min(i)), I_max(i))
            if I_new < I_min:
                I_new = I_min
            elif I_new > I_max:
                I_new = I_max
            
            self.state.I = I_new
            
            # Интегральная составляющая в выходе (нормированная)
            # I_component = (1/T_u) * I_i
            I_norm = self.state.I / self.parameters.Ti
            self.state.I_component = I_norm
        else:
            # Если интеграл отключен (Ti = 0), сбрасываем интегральную сумму
            self.state.I = 0.0
            self.state.I_component = 0.0
        
        # Смещение выходного сигнала (нормированное)
        OP_bias_norm = self.parameters.normalize_OP(self.parameters.OP_bias)
        
        # Суммарное выходное значение (нормированное)
        # OP_i = K_e * e_i + (1/T_u) * I_i + OP_bias
        OP_norm = P_norm + I_norm + OP_bias_norm
        self.state.OP_before_saturation = OP_norm
        
        # Ограничение выходного значения [0, 1]
        if OP_norm < 0.0:
            OP_norm = 0.0
        elif OP_norm > 1.0:
            OP_norm = 1.0
        
        # Денормировка выходного значения
        OP = self.parameters.denormalize_OP(OP_norm)
        
        # Дополнительное ограничение (на всякий случай)
        if OP < self.parameters.OP_min:
            OP = self.parameters.OP_min
        elif OP > self.parameters.OP_max:
            OP = self.parameters.OP_max
        
        self.state.OP = OP
        
        return OP
    
    def reset(self):
        """Сброс состояния регулятора"""
        self.state.reset()
    
    def set_parameters(self, parameters):
        """Обновление параметров регулятора"""
        self.parameters = parameters
        self.parameters.validate()
        self.reset()