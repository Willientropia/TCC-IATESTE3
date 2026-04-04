"""
export_sim_data.py — Exporta as simulações do DQN para uso no frontend.

Lê o modelo treinado e os arquivos de teste originais e cria as trajetórias
(simBattery, simGrid, simDecisions) salvando em arquivos locais no /simulador/
"""
import sys
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from backend.rl.data_loader import load_data, load_scenario_json
from backend.rl.planner import WeeklyPlanner
from backend.rl.environment import BatteryEnv
from backend.rl.dqn_agent import DQNAgent
from backend.rl.reward import DISCHARGE

def process_scenario(agent, env, planner, scenario_path, days_for_plan, output_path):
    scenario = load_scenario_json(scenario_path)
    plan = planner.compute_plan(days_for_plan)
    
    # Quedas de energia
    outage_slots = set()
    desc = scenario["descricao"]
    if "Queda de energia das 00:00 às 05:00 no D2" in desc:
        for i in range(96, 117):
            outage_slots.add(i)
    if "queda 3h às 02h no D6" in desc:
        for i in range(488, 501):
            outage_slots.add(i)
            
    state = env.reset(
        scenario["load"],
        scenario["generation"],
        plan,
        initial_soc=scenario["socInicial"] / 100.0,
        force_outage_slots=outage_slots,
    )
    
    sim_battery = []
    sim_grid = []
    sim_decisions = []
    
    prev_battery = env.battery_wh
    
    while True:
        action = agent.select_action(state, training=False)
        result = env.step(action)
        
        info = result.info
        curr_battery = env.battery_wh
        sim_battery.append(curr_battery)
        
        is_outage = info["slot"] in outage_slots
        
        # Define a decisão para o visualizador Javascript
        if is_outage:
            dec_str = "OUTAGE"
        elif info["in_solar"]:
            dec_str = "SOLAR_CHARGE"
        elif action == DISCHARGE and info["energy_saved"] > 0:
            dec_str = "DISCHARGE"
        else:
            dec_str = "STANDBY"
            
        sim_decisions.append(dec_str)
        
        # Calcula o impacto na rede (Grid Power)
        # O visualizador considera energia que entra/sai como rede (+ = importando)
        grid_w = 0
        if is_outage:
            grid_w = 0
        elif info["in_solar"]:
            # Se carregar a bateria, a potência "injetada" muda
            charge_wh = curr_battery - prev_battery
            grid_w = info["load_w"] - info["gen_w"] + (charge_wh / 0.25)
        else:
            grid_w = max(0, info["load_w"] - (info["energy_saved"] / 0.25))
            
        sim_grid.append(grid_w)
        prev_battery = curr_battery
        
        if result.done:
            break
        state = result.state
        
    out_data = {
        "simBattery": [round(float(b), 2) for b in sim_battery],
        "simGrid": [round(float(g), 2) for g in sim_grid],
        "simDecisions": sim_decisions,
        "totalSaved": round(float(env.total_saved), 2)
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False)
    print(f"Salvo: {output_path} (Economia: {env.total_saved:.0f} Wh)")

def main():
    print("⏳ Carregando sistema DQN...")
    
    # Carregar últimos dados para o planner (como feito no treinamento)
    data_path = ROOT / "dados/training_data_real.json"
    days = load_data(data_path)
    planner = WeeklyPlanner()
    
    # Iniciar ambiente e Agente
    env = BatteryEnv()
    agent = DQNAgent()
    
    model_path = ROOT / "backend/rl/model/dqn_battery.pth"
    if model_path.exists():
        agent.load(model_path)
        print("✅ Modelo carregado com sucesso.")
    else:
        print("❌ Modelo de pesos não encontrado.")
        sys.exit(1)
    
    cenario_3d = ROOT / "dados/cenarios/3d_D1_00h_queda.json"
    cenario_7d = ROOT / "dados/cenarios/prova_7dias.json"
    
    out_3d = ROOT / "simulador/3d_dqn.json"
    out_7d = ROOT / "simulador/7d_dqn.json"
    
    # Exportar os 2 cenários base
    process_scenario(agent, env, planner, cenario_3d, days[-7:], out_3d)
    process_scenario(agent, env, planner, cenario_7d, days[-7:], out_7d)

if __name__ == "__main__":
    main()
