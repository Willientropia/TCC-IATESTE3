"""
environment.py — Ambiente de simulação para o DQN.

Simula o sistema de bateria solar com dados reais.
O agente toma uma decisão por slot de 15 minutos: STANDBY ou DISCHARGE.
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional
from .planner import WeeklyPlanner, WeeklyPlan, SLOTS_PER_DAY, SLOT_H, CAPACITY_WH, SOC_FLOOR
from .reward import compute_reward, STANDBY, DISCHARGE


# Constantes físicas
MAX_DISCHARGE_SLOT_WH = 1485.0  # 27A × 220V × 0.25h
MAX_CHARGE_SLOT_WH = 742.5      # 13.5A × 220V × 0.25h


@dataclass
class StepResult:
    """Resultado de um passo da simulação."""
    state: np.ndarray
    reward: float
    done: bool
    info: dict


class BatteryEnv:
    """
    Ambiente de simulação de bateria solar.
    
    Segue o padrão Gym simplificado:
    - reset(episode_data, plan) → state
    - step(action) → (next_state, reward, done, info)
    
    O agente vê 12 features normalizadas e escolhe entre 2 ações.
    """
    
    N_FEATURES = 12
    N_ACTIONS = 2
    
    def __init__(
        self,
        outage_prob: float = 0.15,
        outage_duration_slots: tuple[int, int] = (12, 20),  # 3-5 horas
        outage_start_range: tuple[int, int] = (72, 92),      # 18:00-23:00
    ):
        """
        Args:
            outage_prob: Probabilidade de queda por episódio no treino.
            outage_duration_slots: (min, max) slots de duração da queda.
            outage_start_range: (min, max) slot de início da queda.
        """
        self.outage_prob = outage_prob
        self.outage_duration = outage_duration_slots
        self.outage_start_range = outage_start_range
        
        # Estado interno
        self.load: Optional[np.ndarray] = None
        self.gen: Optional[np.ndarray] = None
        self.plan: Optional[WeeklyPlan] = None
        self.n_slots: int = 0
        self.current_slot: int = 0
        self.battery_wh: float = 0
        self.prev_action: int = STANDBY
        self.total_saved: float = 0
        self.outage_slots: set = set()
        
        # Para tracking
        self.soc_at_solar_end: float = 1.0
    
    def reset(
        self,
        load: np.ndarray,
        gen: np.ndarray,
        plan: WeeklyPlan,
        initial_soc: float = 0.99,
        force_outage_slots: Optional[set] = None,
    ) -> np.ndarray:
        """
        Reinicia o ambiente para um novo episódio.
        
        Args:
            load: Array de carga (W) por slot.
            gen: Array de geração (W) por slot.
            plan: Plano semanal do Agente 1.
            initial_soc: SOC inicial (0-1).
            force_outage_slots: Slots com queda forçada (para cenários de teste).
        
        Returns:
            Estado inicial (12 features).
        """
        self.load = load.astype(np.float32)
        self.gen = gen.astype(np.float32)
        self.plan = plan
        self.n_slots = len(load)
        self.current_slot = 0
        self.battery_wh = initial_soc * CAPACITY_WH
        self.prev_action = STANDBY
        self.total_saved = 0.0
        self.soc_at_solar_end = initial_soc
        
        # Gerar queda aleatória (treino) ou usar forçada (teste)
        if force_outage_slots is not None:
            self.outage_slots = force_outage_slots
        elif np.random.random() < self.outage_prob:
            self._generate_random_outage()
        else:
            self.outage_slots = set()
        
        return self._get_state()
    
    def step(self, action: int) -> StepResult:
        """
        Executa um passo da simulação.
        
        Args:
            action: 0 = STANDBY, 1 = DISCHARGE
        
        Returns:
            StepResult com (next_state, reward, done, info)
        """
        assert 0 <= action <= 1, f"Ação inválida: {action}"
        
        slot = self.current_slot
        slot_in_day = slot % SLOTS_PER_DAY
        load_w = max(0, self.load[slot])
        gen_w = max(0, self.gen[slot])
        grid_ok = slot not in self.outage_slots
        soc = self.battery_wh / CAPACITY_WH
        
        in_solar = self.plan is not None and self._is_solar(slot_in_day)
        in_discharge = self.plan is not None and self._is_discharge_window(slot_in_day)
        
        energy_saved = 0.0
        
        # ======== FÍSICA DA BATERIA ========
        if not grid_ok:
            # QUEDA: bateria cobre carga independente da ação
            needed = load_w * SLOT_H
            available = max(0, self.battery_wh - SOC_FLOOR * CAPACITY_WH)
            provide = min(needed, available, MAX_DISCHARGE_SLOT_WH)
            self.battery_wh -= provide
            
        elif in_solar:
            # SOLAR: carregar bateria com excedente
            surplus = gen_w - load_w
            if surplus > 0:
                charge = min(surplus * SLOT_H, CAPACITY_WH - self.battery_wh, MAX_CHARGE_SLOT_WH)
                self.battery_wh += charge
            # Registrar SOC ao fim do solar
            if slot_in_day == self.plan.solar_end_slot:
                self.soc_at_solar_end = self.battery_wh / CAPACITY_WH
            
        elif action == DISCHARGE:
            # DESCARGA: bateria cobre a carga
            available = max(0, self.battery_wh - SOC_FLOOR * CAPACITY_WH)
            provide = min(load_w * SLOT_H, available, MAX_DISCHARGE_SLOT_WH)
            self.battery_wh -= provide
            energy_saved = provide
            self.total_saved += provide
            
        # else: STANDBY — nada acontece com a bateria
        
        # Clamp
        self.battery_wh = np.clip(self.battery_wh, 0, CAPACITY_WH)
        
        # ======== RECOMPENSA ========
        new_soc = self.battery_wh / CAPACITY_WH
        is_sunrise = (
            slot_in_day == self.plan.solar_start_slot if self.plan else False
        )
        
        reward = compute_reward(
            soc=new_soc,
            action=action,
            prev_action=self.prev_action,
            in_solar=in_solar,
            in_discharge_window=in_discharge,
            grid_ok=grid_ok,
            is_sunrise=is_sunrise,
            energy_saved_wh=energy_saved,
        )
        
        # ======== AVANÇAR ========
        self.prev_action = action
        self.current_slot += 1
        done = self.current_slot >= self.n_slots
        
        info = {
            "slot": slot,
            "slot_in_day": slot_in_day,
            "soc": new_soc,
            "load_w": load_w,
            "gen_w": gen_w,
            "grid_ok": grid_ok,
            "in_solar": in_solar,
            "in_discharge": in_discharge,
            "energy_saved": energy_saved,
            "total_saved": self.total_saved,
            "action": action,
        }
        
        next_state = self._get_state() if not done else np.zeros(self.N_FEATURES)
        
        return StepResult(
            state=next_state,
            reward=reward,
            done=done,
            info=info,
        )
    
    def _get_state(self) -> np.ndarray:
        """Constrói vetor de 12 features normalizadas."""
        slot = self.current_slot
        slot_in_day = slot % SLOTS_PER_DAY
        soc = self.battery_wh / CAPACITY_WH
        
        # Features temporais (cíclicas)
        hora_sin = np.sin(2 * np.pi * slot_in_day / SLOTS_PER_DAY)
        hora_cos = np.cos(2 * np.pi * slot_in_day / SLOTS_PER_DAY)
        
        # Slots até solar (normalizado)
        if self.plan:
            slots_solar = self._slots_until_solar(slot_in_day) / SLOTS_PER_DAY
        else:
            slots_solar = 0.5
        
        # SOC vs alvo do plano
        soc_vs_alvo = soc - SOC_FLOOR
        
        # Na janela de descarga?
        na_janela = 1.0 if (self.plan and self._is_discharge_window(slot_in_day)) else 0.0
        
        # Solar ratio: geração recente vs geração esperada
        if self.plan and self.plan.avg_day_gen_wh > 0:
            recent_gen = self._recent_avg(self.gen, slot, window=4)
            # Normalizar pela geração esperada por slot
            expected_per_slot = self.plan.avg_day_gen_wh / (
                self.plan.solar_end_slot - self.plan.solar_start_slot + 1
            ) / SLOT_H
            solar_ratio = min(2.0, recent_gen / max(1, expected_per_slot))
        else:
            solar_ratio = 0.0
        
        # Grid OK
        grid_ok = 1.0 if slot not in self.outage_slots else 0.0
        
        # Razão de cobertura
        if self.plan and self.plan.avg_night_load_w > 0:
            usable = max(0, self.battery_wh - SOC_FLOOR * CAPACITY_WH)
            slots_remaining = self._slots_until_solar(slot_in_day)
            deficit_remaining = self.plan.avg_night_load_w * slots_remaining * SLOT_H
            razao_cobertura = min(2.0, usable / max(1, deficit_remaining))
        else:
            razao_cobertura = 1.0
        
        # SOC ao fim do solar ontem
        soc_fim_solar = self.soc_at_solar_end
        
        # Carga recente (média dos últimos 4 slots / 2000)
        carga_recente = self._recent_avg(self.load, slot, window=4) / 2000.0
        
        # Última ação
        ultima_acao = float(self.prev_action)
        
        state = np.array([
            soc,                # 0: SOC normalizado
            hora_sin,           # 1: sin(hora)
            hora_cos,           # 2: cos(hora)
            slots_solar,        # 3: slots até solar (norm.)
            soc_vs_alvo,        # 4: SOC - floor
            na_janela,          # 5: na janela de descarga?
            solar_ratio,        # 6: geração vs esperado
            grid_ok,            # 7: rede OK?
            razao_cobertura,    # 8: energia / déficit restante
            soc_fim_solar,      # 9: SOC fim do solar ontem
            carga_recente,      # 10: carga recente normalizada
            ultima_acao,        # 11: última ação
        ], dtype=np.float32)
        
        return state
    
    def _is_solar(self, slot_in_day: int) -> bool:
        if not self.plan:
            return False
        return self.plan.solar_start_slot <= slot_in_day <= self.plan.solar_end_slot
    
    def _is_discharge_window(self, slot_in_day: int) -> bool:
        if not self.plan:
            return False
        start = self.plan.discharge_start_slot
        end = self.plan.solar_start_slot
        if start < end:
            return start <= slot_in_day < end
        else:
            return slot_in_day >= start or slot_in_day < end
    
    def _slots_until_solar(self, slot_in_day: int) -> int:
        if not self.plan:
            return 48
        if slot_in_day < self.plan.solar_start_slot:
            return self.plan.solar_start_slot - slot_in_day
        else:
            return (SLOTS_PER_DAY - slot_in_day) + self.plan.solar_start_slot
    
    def _recent_avg(self, arr: np.ndarray, slot: int, window: int = 4) -> float:
        start = max(0, slot - window)
        if start >= slot:
            return 0.0
        return float(np.mean(arr[start:slot]))
    
    def _generate_random_outage(self):
        """Gera queda aleatória para treino."""
        self.outage_slots = set()
        start = np.random.randint(*self.outage_start_range)
        duration = np.random.randint(*self.outage_duration)
        for s in range(start, min(start + duration, self.n_slots)):
            self.outage_slots.add(s)
