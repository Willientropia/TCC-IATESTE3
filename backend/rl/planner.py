"""
planner.py — Agente 1: Planejador Semanal (estatístico).

Calcula perfis semanais a partir de dados históricos:
- Horário de nascer/pôr do sol (período solar)
- Déficit noturno típico
- Carga noturna média
- Slot ideal para início da descarga

NENHUM machine learning aqui — puro cálculo estatístico.
Recalcula automaticamente com dados novos.
"""
import numpy as np
from dataclasses import dataclass


SLOTS_PER_DAY = 96
SLOT_H = 0.25  # 15 min em horas
CAPACITY_WH = 5000
SOC_TARGET_SUNRISE = 0.12  # Alvo: 12% ao amanhecer
SOC_FLOOR = 0.10           # Mínimo absoluto


@dataclass
class WeeklyPlan:
    """Plano semanal gerado pelo Agente 1."""
    solar_start_slot: int     # Slot do nascer do sol (0-95)
    solar_end_slot: int       # Slot do pôr do sol (0-95)
    avg_night_load_w: float   # Carga noturna média (W)
    avg_day_gen_wh: float     # Geração diária média (Wh)
    discharge_start_slot: int # Slot ideal para iniciar descarga (0-95)
    discharge_slots: int      # Quantos slots de descarga
    usable_wh: float          # Energia utilizável ao anoitecer
    night_deficit_wh: float   # Déficit noturno total (Wh)
    
    def describe(self) -> str:
        """Descrição legível do plano."""
        def slot_to_time(s):
            h = (s * 15) // 60
            m = (s * 15) % 60
            return f"{h:02d}:{m:02d}"
        
        return (
            f"=== Plano Semanal ===\n"
            f"Sol: {slot_to_time(self.solar_start_slot)} - {slot_to_time(self.solar_end_slot)}\n"
            f"Carga noturna média: {self.avg_night_load_w:.0f} W\n"
            f"Geração diária: {self.avg_day_gen_wh:.0f} Wh\n"
            f"Energia utilizável: {self.usable_wh:.0f} Wh\n"
            f"Déficit noturno: {self.night_deficit_wh:.0f} Wh\n"
            f"Descarga: {slot_to_time(self.discharge_start_slot)} → {slot_to_time(self.solar_start_slot)} "
            f"({self.discharge_slots} slots = {self.discharge_slots * 0.25:.1f}h)\n"
        )


class WeeklyPlanner:
    """
    Agente 1 — Calcula plano semanal baseado em dados históricos.
    
    Não usa ML. Calcula estatísticas simples dos últimos N dias
    para orientar o Agente 2 (DQN).
    """
    
    def __init__(self, solar_threshold_w: float = 200.0):
        """
        Args:
            solar_threshold_w: Geração mínima (W) para considerar como período solar.
        """
        self.solar_threshold = solar_threshold_w
    
    def compute_plan(
        self,
        days: list[dict],
        soc_at_solar_end: float = 1.0,
    ) -> WeeklyPlan:
        """
        Calcula plano para a próxima semana baseado nos últimos dias.
        
        Args:
            days: Lista de dicts com {"load": ndarray(96,), "gen": ndarray(96,)}.
            soc_at_solar_end: SOC esperado ao fim do período solar (0-1).
        
        Returns:
            WeeklyPlan com parâmetros calculados.
        """
        if not days:
            return self._default_plan()
        
        # 1. Detectar período solar médio
        solar_start, solar_end = self._detect_solar_period(days)
        
        # 2. Calcular estatísticas noturnas
        night_loads = []
        day_gens = []
        
        for d in days:
            load = d["load"]
            gen = d["gen"]
            
            # Geração diária total
            day_gens.append(np.sum(gen) * SLOT_H)
            
            # Carga noturna: slots fora do período solar
            night_mask = np.ones(SLOTS_PER_DAY, dtype=bool)
            night_mask[solar_start:solar_end + 1] = False
            night_loads.append(np.mean(load[night_mask]))
        
        avg_night_load = np.mean(night_loads)
        avg_day_gen = np.mean(day_gens)
        
        # 3. Calcular energia utilizável e janela de descarga
        battery_at_end = soc_at_solar_end * CAPACITY_WH
        usable = max(0, battery_at_end - SOC_TARGET_SUNRISE * CAPACITY_WH)
        
        # Déficit noturno total
        night_slots = SLOTS_PER_DAY - (solar_end - solar_start + 1)
        night_deficit = avg_night_load * night_slots * SLOT_H
        
        # Slots de descarga (quantos slots a bateria aguenta) + 2 slots de "Gordura" a pedido do usuário
        wh_per_slot = avg_night_load * SLOT_H
        discharge_slots = int(usable / wh_per_slot) if wh_per_slot > 0 else 0
        discharge_slots = min(discharge_slots + 2, night_slots)
        
        # Slot de início: retrocede a partir do amanhecer
        discharge_start = solar_start - discharge_slots
        if discharge_start < 0:
            discharge_start += SLOTS_PER_DAY  # Volta para noite anterior
        
        return WeeklyPlan(
            solar_start_slot=solar_start,
            solar_end_slot=solar_end,
            avg_night_load_w=avg_night_load,
            avg_day_gen_wh=avg_day_gen,
            discharge_start_slot=discharge_start,
            discharge_slots=discharge_slots,
            usable_wh=usable,
            night_deficit_wh=night_deficit,
        )
    
    def _detect_solar_period(self, days: list[dict]) -> tuple[int, int]:
        """Detecta período solar médio usando geração acumulada."""
        # Perfil médio de geração
        gen_profile = np.mean([d["gen"] for d in days], axis=0)
        
        # Suavizar com média móvel (janela de 5 slots = 75 min)
        kernel = np.ones(5) / 5
        smoothed = np.convolve(gen_profile, kernel, mode="same")
        
        # Encontrar primeiro e último slot acima do threshold
        above = smoothed >= self.solar_threshold
        if not above.any():
            return 27, 70  # Fallback: ~06:45 a ~17:30
        
        indices = np.where(above)[0]
        return int(indices[0]), int(indices[-1])
    
    def _default_plan(self) -> WeeklyPlan:
        """Plano padrão quando não há dados históricos."""
        return WeeklyPlan(
            solar_start_slot=27,  # 06:45
            solar_end_slot=70,     # 17:30
            avg_night_load_w=500,
            avg_day_gen_wh=10000,
            discharge_start_slot=3,  # 00:45
            discharge_slots=24,
            usable_wh=4400,
            night_deficit_wh=7500,
        )
    
    def is_solar_slot(self, slot: int, plan: WeeklyPlan) -> bool:
        """Verifica se um slot está no período solar."""
        return plan.solar_start_slot <= slot <= plan.solar_end_slot
    
    def is_discharge_window(self, slot: int, plan: WeeklyPlan) -> bool:
        """
        Verifica se um slot está na janela de descarga.
        Lida com wrap-around (ex: 23:00 → 06:00).
        """
        start = plan.discharge_start_slot
        end_slot = plan.solar_start_slot  # Descarga vai até o sol nascer
        
        if start < end_slot:
            # Janela normal: ex slot 4 a slot 27
            return start <= slot < end_slot
        else:
            # Wrap-around: ex slot 88 a slot 27 (passa pela meia-noite)
            return slot >= start or slot < end_slot
    
    def slots_until_solar(self, current_slot: int, plan: WeeklyPlan) -> int:
        """Quantos slots até o período solar."""
        if current_slot < plan.solar_start_slot:
            return plan.solar_start_slot - current_slot
        else:
            return (SLOTS_PER_DAY - current_slot) + plan.solar_start_slot


if __name__ == "__main__":
    from data_loader import load_and_process, get_all_days
    import sys
    
    path = sys.argv[1] if len(sys.argv) > 1 else "dados/leituras.json"
    
    print(f"Carregando {path}...")
    df = load_and_process(path)
    days = get_all_days(df)
    
    print(f"Usando últimos 7 dias de {len(days)} disponíveis...")
    planner = WeeklyPlanner()
    plan = planner.compute_plan(days[-7:])
    print(plan.describe())
