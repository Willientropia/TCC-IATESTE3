"""
simulate_30d.py — Roda 28 dias contínuos de simulação gerando quedas de energia.

A cada virada de semana (7 dias), o planejamento estatístico é atualizado
usando os últimos 7 dias. Salva os 4 pedaços (4 Semanas) num JSON master.
"""
import sys
import json
import random
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from backend.rl.data_loader import load_data
from backend.rl.planner import WeeklyPlanner
from backend.rl.environment import BatteryEnv
from backend.rl.dqn_agent import DQNAgent
from backend.rl.reward import DISCHARGE

random.seed(42)  # Manter blackouts consistentes para debug visual

def main():
    print("⏳ Iniciando Maratona 30 dias (DQN)...")
    
    data_path = ROOT / "dados/training_data_real.json"
    days = load_data(data_path)
    
    if len(days) < 28:
        print("❌ Dataset muito pequeno. Precisamos de pelo menos 28 dias.")
        sys.exit(1)
        
    # Extrai 28 dias diretos
    days_to_sim = days[:28]
    load_28d = np.concatenate([d["load"] for d in days_to_sim])
    gen_28d = np.concatenate([d["gen"] for d in days_to_sim])
    
    # Gerar Apagões (1 a 3 por semana)
    outage_slots = set()
    for week in range(4):
        num_outages = random.randint(1, 3)
        days_in_week = random.sample(range(7), num_outages)
        for day in days_in_week:
            global_day = week * 7 + day
            duration = random.randint(8, 20)  # 2 a 5 horas
            start_slot = random.randint(0, 20)  # Queda entre 00:00 e 05:00
            
            start_idx = global_day * 96 + start_slot
            for i in range(duration):
                outage_slots.add(start_idx + i)
                
    print(f"🌩️ Quedas programadas no mês: {len(outage_slots)} slots ({len(outage_slots)*15/60:.1f} horas totais)")

    planner = WeeklyPlanner()
    env = BatteryEnv()
    agent = DQNAgent()
    
    model_path = ROOT / "backend/rl/model/dqn_battery.pth"
    if model_path.exists():
        agent.load(model_path)
    else:
        print("❌ Modelo não encontrado.")
        sys.exit(1)
        
    # Inicializa plan com a própria semana 1 para bootstrap
    initial_plan = planner.compute_plan(days[:7])
    
    state = env.reset(
        load_28d, 
        gen_28d, 
        initial_plan, 
        initial_soc=1.0, 
        force_outage_slots=outage_slots
    )
    
    sim_battery, sim_grid, sim_decisions = [], [], []
    sim_load, sim_gen = [], []
    
    for week in range(4):
        print(f"🗓️ Rodando Semana {week+1}...")
        if week > 0:
            # Ao ligar a nova semana, o planejador olha pra SEMANA ANTERIOR
            prev_week_days = days[(week-1)*7 : week*7]
            env.plan = planner.compute_plan(prev_week_days)
            
        for step_in_week in range(96 * 7):
            global_step = week * 96 * 7 + step_in_week
            
            action = agent.select_action(state, training=False)
            prev_battery = env.battery_wh
            
            result = env.step(action)
            info = result.info
            
            # Coleta métricas
            curr_battery = env.battery_wh
            sim_battery.append(curr_battery)
            sim_load.append(info["load_w"])
            sim_gen.append(info["gen_w"])
            
            is_outage = info["slot"] in outage_slots
            if is_outage: dec_str = "OUTAGE"
            elif info["in_solar"]: dec_str = "SOLAR_CHARGE"
            elif action == DISCHARGE and info["energy_saved"] > 0: dec_str = "DISCHARGE"
            else: dec_str = "STANDBY"
            sim_decisions.append(dec_str)
            
            # Impacto Grid
            if is_outage:
                grid_w = 0
            elif info["in_solar"]:
                charge_wh = curr_battery - prev_battery
                grid_w = info["load_w"] - info["gen_w"] + (charge_wh / 0.25)
            else:
                grid_w = max(0, info["load_w"] - (info["energy_saved"] / 0.25))
                
            sim_grid.append(grid_w)
            state = result.state

    # Empacota em 4 Semanas Separadas para o Frontend desenhar os 4 Gráficos
    out_data = {"semanas": []}
    for week in range(4):
        start = week * 96 * 7
        end = start + 96 * 7
        
        outages_bool = [(start + i) in outage_slots for i in range(96*7)]
        
        sem_data = {
            "semana": week + 1,
            "simBattery": [round(float(b), 2) for b in sim_battery[start:end]],
            "simGrid": [round(float(g), 2) for g in sim_grid[start:end]],
            "simLoad": [round(float(l), 2) for l in sim_load[start:end]],
            "simGen": [round(float(g), 2) for g in sim_gen[start:end]],
            "simDecisions": sim_decisions[start:end],
            "outages": outages_bool
        }
        out_data["semanas"].append(sem_data)
        
    out_data["totalSaved"] = round(float(env.total_saved), 2)
    
    out_path = ROOT / "simulador/30d_dqn.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False)
        
    print(f"✅ Geração finalizada! Dados salvos em: {out_path}")
    print(f"💰 Economia Efetiva no Mês: {env.total_saved:.0f} Wh")

if __name__ == "__main__":
    main()
