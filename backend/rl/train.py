"""
train.py — Script principal de treinamento do DQN.

Carrega dados reais do leituras.json, cria episódios de 1-3 dias,
treina o agente DQN e valida nos cenários de teste.
"""
import sys
import time
import numpy as np
from pathlib import Path

# Adiciona o diretório raiz ao path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from backend.rl.data_loader import load_data, load_scenario_json
from backend.rl.planner import WeeklyPlanner, SLOTS_PER_DAY, SLOT_H
from backend.rl.environment import BatteryEnv
from backend.rl.dqn_agent import DQNAgent


def create_episode(days: list[dict], n_days: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """
    Cria um episódio de treino selecionando n_days consecutivos aleatórios.
    Retorna (load[96*n], gen[96*n]).
    """
    if len(days) < n_days:
        n_days = len(days)
    
    start = np.random.randint(0, len(days) - n_days + 1)
    selected = days[start:start + n_days]
    
    load = np.concatenate([d["load"] for d in selected])
    gen = np.concatenate([d["gen"] for d in selected])
    
    return load, gen


def evaluate_scenario(
    agent: DQNAgent,
    env: BatteryEnv,
    planner: WeeklyPlanner,
    scenario_path: str,
    scenario_name: str,
    days_for_plan: list[dict],
) -> dict:
    """Avalia o agente num cenário de teste."""
    scenario = load_scenario_json(scenario_path)
    plan = planner.compute_plan(days_for_plan)
    
    # Detectar slots de queda a partir da descrição
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
    
    total_reward = 0
    soc_at_sunrise = []
    actions_log = []
    
    while True:
        action = agent.select_action(state, training=False)
        result = env.step(action)
        total_reward += result.reward
        actions_log.append(result.info)
        
        # Registrar SOC ao amanhecer
        if result.info["slot_in_day"] == plan.solar_start_slot:
            soc_at_sunrise.append(result.info["soc"])
        
        if result.done:
            break
        state = result.state
    
    # Métricas
    discharge_during_solar = sum(
        1 for a in actions_log if a["in_solar"] and a["action"] == 1
    )
    
    outage_survived = all(
        a["soc"] > 0.05 for a in actions_log if not a["grid_ok"]
    )
    
    # Contar trocas de estado
    switches = sum(
        1 for i in range(1, len(actions_log))
        if actions_log[i]["action"] != actions_log[i-1]["action"]
    )
    
    results = {
        "scenario": scenario_name,
        "total_reward": total_reward,
        "total_saved_wh": env.total_saved,
        "avg_soc_sunrise": np.mean(soc_at_sunrise) if soc_at_sunrise else -1,
        "discharge_during_solar": discharge_during_solar,
        "outage_survived": outage_survived,
        "action_switches": switches,
    }
    
    return results


def print_eval(results: dict):
    """Imprime resultados de avaliação formatados."""
    r = results
    status = "✅" if r["outage_survived"] else "❌"
    solar_ok = "✅" if r["discharge_during_solar"] == 0 else f"❌ ({r['discharge_during_solar']})"
    soc_ok = "✅" if 0 < r["avg_soc_sunrise"] <= 0.20 else "⚠️"
    
    print(f"\n{'='*50}")
    print(f"  Cenário: {r['scenario']}")
    print(f"  Recompensa total: {r['total_reward']:.1f}")
    print(f"  Economia total: {r['total_saved_wh']:.0f} Wh")
    print(f"  SOC médio amanhecer: {r['avg_soc_sunrise']*100:.1f}% {soc_ok}")
    print(f"  Queda sobrevivida: {status}")
    print(f"  Descarga no solar: {solar_ok}")
    print(f"  Trocas ação: {r['action_switches']}")
    print(f"{'='*50}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Treinar DQN para gestão de bateria solar")
    parser.add_argument("--episodes", type=int, default=800, help="Número de episódios")
    parser.add_argument("--days-per-episode", type=int, default=2, help="Dias por episódio")
    parser.add_argument("--eval-every", type=int, default=100, help="Avaliar a cada N episódios")
    parser.add_argument("--data", type=str, default="dados/training_data_real.json", help="Caminho dos dados")
    parser.add_argument("--save-path", type=str, default="backend/rl/model/dqn_battery.pth")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.97)
    args = parser.parse_args()
    
    print("=" * 60)
    print(" 🔋 Treinamento DQN — Gestão de Bateria Solar")
    print("=" * 60)
    
    # 1. Carregar dados
    print(f"\n📊 Carregando dados de {args.data}...")
    data_path = ROOT / args.data
    days = load_data(data_path)
    print(f"   {len(days)} dias completos carregados ({days[0]['date']} → {days[-1]['date']})")
    
    # 2. Criar planejador e plano
    print("\n📅 Criando plano semanal...")
    planner = WeeklyPlanner()
    plan = planner.compute_plan(days[-7:])
    print(plan.describe())
    
    # 3. Criar ambiente e agente
    env = BatteryEnv(outage_prob=0.15)
    agent = DQNAgent(
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
    )
    
    # 4. Treinar
    print(f"\n🚀 Iniciando treinamento ({args.episodes} episódios)...")
    best_eval_reward = -float("inf")
    episode_rewards = []
    start_time = time.time()
    
    for ep in range(1, args.episodes + 1):
        # Criar episódio aleatório
        load, gen = create_episode(days, n_days=args.days_per_episode)
        
        # Variar SOC inicial
        initial_soc = np.random.uniform(0.70, 1.0)
        
        # Plano para este episódio (usar últimos 7 dias antes do episódio)
        ep_plan = planner.compute_plan(days[-7:], soc_at_solar_end=initial_soc)
        
        state = env.reset(load, gen, ep_plan, initial_soc=initial_soc)
        ep_reward = 0
        ep_losses = []
        
        while True:
            action = agent.select_action(state, training=True)
            result = env.step(action)
            
            agent.store(state, action, result.reward, result.state, result.done)
            loss = agent.train_step()
            if loss is not None:
                ep_losses.append(loss)
            
            ep_reward += result.reward
            
            if result.done:
                break
            state = result.state
        
        agent.decay_epsilon()
        episode_rewards.append(ep_reward)
        
        # Log
        if ep % 50 == 0 or ep == 1:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_loss = np.mean(ep_losses) if ep_losses else 0
            elapsed = time.time() - start_time
            print(
                f"  Ep {ep:4d}/{args.episodes} | "
                f"Reward: {avg_reward:8.1f} | "
                f"Loss: {avg_loss:.4f} | "
                f"ε: {agent.epsilon:.3f} | "
                f"Buffer: {len(agent.buffer):5d} | "
                f"Saved: {env.total_saved:6.0f} Wh | "
                f"Time: {elapsed:.0f}s"
            )
        
        # Avaliar
        if ep % args.eval_every == 0:
            print(f"\n  📊 Avaliação no episódio {ep}:")
            
            scenario_3d = ROOT / "dados" / "cenarios" / "3d_D1_00h_queda.json"
            scenario_7d = ROOT / "dados" / "cenarios" / "prova_7dias.json"
            
            if scenario_3d.exists():
                r3d = evaluate_scenario(
                    agent, env, planner, str(scenario_3d), "3d (Queda D2)", days[-7:]
                )
                print_eval(r3d)
                
                if r3d["total_reward"] > best_eval_reward:
                    best_eval_reward = r3d["total_reward"]
                    save_path = ROOT / args.save_path
                    agent.save(save_path)
                    print(f"  💾 Novo melhor modelo! (reward={best_eval_reward:.1f})")
            
            if scenario_7d.exists():
                r7d = evaluate_scenario(
                    agent, env, planner, str(scenario_7d), "7d (Queda D6)", days[-7:]
                )
                print_eval(r7d)
    
    # 5. Salvar modelo final
    total_time = time.time() - start_time
    save_path = ROOT / args.save_path
    agent.save(save_path)
    
    print(f"\n{'='*60}")
    print(f" ✅ Treinamento concluído!")
    print(f"    Episódios: {args.episodes}")
    print(f"    Tempo: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"    Melhor reward: {best_eval_reward:.1f}")
    print(f"    Modelo salvo: {save_path}")
    print(f"{'='*60}")
    
    # 6. Avaliação final
    print("\n📊 AVALIAÇÃO FINAL:")
    
    scenario_3d = ROOT / "dados" / "cenarios" / "3d_D1_00h_queda.json"
    scenario_7d = ROOT / "dados" / "cenarios" / "prova_7dias.json"
    
    if scenario_3d.exists():
        r3d = evaluate_scenario(
            agent, env, planner, str(scenario_3d), "3d (Queda D2)", days[-7:]
        )
        print_eval(r3d)
    
    if scenario_7d.exists():
        r7d = evaluate_scenario(
            agent, env, planner, str(scenario_7d), "7d (Queda D6)", days[-7:]
        )
        print_eval(r7d)


if __name__ == "__main__":
    main()
