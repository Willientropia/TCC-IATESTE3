"""
ai_bridge.py — Motor em Tempo Real que liga a IA (DQN) ao Inversor via Firebase

Este script deve ficar rodando 24/7 num Computador/MiniPC com internet.
A cada 15 minutos ele:
 1. Baixa métricas do ESP32 (Firebase RTDB).
 2. Transforma em Input para o Agente DQN.
 3. Decide se a bateria fica em STANDBY (Modo BackUp) ou DESCARREGA (Modo Self Use).
 4. Envia o comado via fila Modbus para o ESP32.
 5. Alimente o Dashboard do App React/Native com as logs e métricas.
"""
import sys
import os
import time
import json
import uuid
import requests
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from backend.rl.planner import WeeklyPlanner, WeeklyPlan
from backend.rl.dqn_agent import DQNAgent

# Carregar variáveis de ambiente
load_dotenv()

# ==================== Credenciais Firebase ====================
FB_API_KEY = os.getenv("FB_API_KEY")
FB_URL = os.getenv("FB_URL")
FB_EMAIL = os.getenv("FB_EMAIL")
FB_PASS = os.getenv("FB_PASS")
DEVICE_ID = os.getenv("DEVICE_ID", "esp32_01")

# ==================== Caminhos Firebase ====================
PATH_REALTIME = os.getenv("PATH_REALTIME", f"/devices/{DEVICE_ID}/realtime")
PATH_HISTORY = os.getenv("PATH_HISTORY", f"/devices/{DEVICE_ID}/leituras")
PATH_COMMANDS = os.getenv("PATH_COMMANDS", f"/devices/{DEVICE_ID}/commands")
PATH_DASHBOARD = os.getenv("PATH_DASHBOARD", f"/devices/{DEVICE_ID}/ai_dashboard")

if not FB_API_KEY:
    print("❌ Arquivo .env não configurado ou FB_API_KEY ausente.")
    sys.exit(1)

# Endereçamento
REG_WORKMODE = 49203
VAL_STANDBY = 3     # BackUp (Segura a carga)
VAL_DISCHARGE = 1   # Self Use (Bateria corre solta pela casa)

SLOT_H = 0.25

class FirebaseClient:
    def __init__(self):
        self.token = self._login()

    def _login(self):
        print(f"🔑 Autenticando com {FB_EMAIL}...")
        url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FB_API_KEY}"
        resp = requests.post(url, json={"email": FB_EMAIL, "password": FB_PASS, "returnSecureToken": True})
        if resp.status_code == 200:
            print("✅ Login via REST API concluído.")
            return resp.json().get("idToken")
        else:
            print("❌ Falha de Login:", resp.text)
            sys.exit(1)

    def get_data(self, path):
        url = f"{FB_URL}{path}.json?auth={self.token}"
        resp = requests.get(url)
        return resp.json() if resp.status_code == 200 else None

    def put_data(self, path, data):
        url = f"{FB_URL}{path}.json?auth={self.token}"
        requests.put(url, json=data)
        
    def push_command(self, address, value):
        cmd_id = str(uuid.uuid4())
        cmd_path = f"{PATH_COMMANDS}/{cmd_id}"
        payload = {
            "type": "modbus_write",
            "payload": {
                "addr": address,
                "regs": f"[{value}]"
            },
            "status": "pending",
            "createdAt": int(time.time())
        }
        self.put_data(cmd_path, payload)
        print(f"📡 Comando Enviado! Modbus {address} -> [{value}]. CmdID: {cmd_id[:8]}")

class AIEngine:
    def __init__(self):
        print("🧠 Inicializando Motores da IA...")
        self.fb = FirebaseClient()
        self.planner = WeeklyPlanner()
        self.agent = DQNAgent()
        
        # Load pre-trained weights
        model_path = ROOT / "backend/rl/model/dqn_battery.pth"
        if model_path.exists():
            self.agent.load(model_path)
        else:
            print("❌ Modelo Base DQN não encontrado. Treine primeiro.")
            sys.exit(1)
            
        self.current_plan = None
        self.last_decision = VAL_STANDBY

    def fetch_historical_week(self):
        """O Agente 1 (Planner) baixa os ultimos 7 dias da Firebase para criar as janelas"""
        print("🗓️ Lendo dados da semana passada no banco histórico (Agente 1)...")
        
        now = datetime.now()
        load_list = []
        gen_list = []
        
        # Coletar os últimos 7 dias do Firebase
        for i in range(1, 8):
            dia = (now - timedelta(days=i)).strftime("%Y-%m-%d")
            data_path = f"{PATH_HISTORY}/{dia}"
            dia_data = self.fb.get_data(data_path)
            
            if dia_data and isinstance(dia_data, dict):
                # Calcular médias brutas para o dia caso existam medições
                for hora_str, values in dia_data.items():
                    # values: {"load_w": 500, "solar_w": 2000, ...}
                    # O Firebase pode ter chaves variadas dependendo da versão do código ESP32
                    load = values.get("load_w") or values.get("Load Combined Power") or values.get("Load Total Power/kW")
                    gen = values.get("solar_w") or values.get("PV total Power/kW")
                    
                    if load is not None:
                        # Convert kW to W if needed based on value size
                        load_val = float(load) * 1000 if "kW" in str(load) or float(load) < 50 else float(load)
                        load_list.append(load_val)
                        
                    if gen is not None:
                        gen_val = float(gen) * 1000 if "kW" in str(gen) or float(gen) < 50 else float(gen)
                        gen_list.append(gen_val)

        # Média calculada a partir dos dados do Firebase (com fallback se vazio)
        avg_night_load = np.mean(load_list) if load_list else 500.0
        avg_day_gen = np.mean(gen_list) * 12 if gen_list else 8000.0 # Estimativa de produção total baseada na média diária
        
        self.current_plan = WeeklyPlan(
            solar_start_slot=24,   # 06:00
            solar_end_slot=72,     # 18:00
            avg_night_load_w=float(avg_night_load),
            avg_day_gen_wh=float(avg_day_gen),
            discharge_start_slot=76, # Permite descarga das 19:00 até o amanhecer
            discharge_slots=16,
            usable_wh=3000.0,
            night_deficit_wh=5000.0
        )
        print(f"✅ Planejador Semanal Calculado: Janela livre entre slot {self.current_plan.discharge_start_slot} ao {self.current_plan.solar_start_slot}")

    def loop_tick(self):
        now = datetime.now()
        
        # Identificar Slot do Tempo Real (0 a 95)
        current_minute = now.hour * 60 + now.minute
        slot = current_minute // 15
        
        print(f"\n[{now.strftime('%H:%M:%S')}] ⏰ Acordando DQN no Slot {slot}")
        
        # 1. Puxar as estatística quente (Leitura mais recente do dia atual)
        hoje_str = now.strftime("%Y-%m-%d")
        daily_data = self.fb.get_data(f"{PATH_HISTORY}/{hoje_str}")
        
        soc_atual = 0.85
        load_w = 600.0
        gen_w = 2500.0
        grid_ok = 1.0
        
        if daily_data and isinstance(daily_data, dict):
            # Encontrar a última inserção de horário
            latest_time = sorted(daily_data.keys())[-1]
            rt_data = daily_data[latest_time]
            print(f"📥 Coleta em Tempo Real via banco Histórico: {latest_time} (ESP32 Post)")
            
            # Lógica extraída do seu data_loader.py original
            soc_raw = rt_data.get("soc", 85.0)
            load_raw = rt_data.get("Load Combined Power", 600.0)
            grid_raw = rt_data.get("Coleta do Medidor", 0.0)
            
            soc_atual = float(soc_raw) / 100.0 if float(soc_raw) > 1 else float(soc_raw)
            load_w = float(load_raw)
            gen_w = max(0.0, load_w + float(grid_raw)) # Geração = Carga + Excedente Grid
            
            # Se fosse extrair falta de rede (Apagão) poderiamos ver depois
            grid_ok = 1.0
        else:
            print(f"⚠️ Sem dados para o dia de hoje ({hoje_str}) ainda. Usando fallback.")
        
        # 2. Avaliando premissas como os blocos de treinamento
        in_solar = (self.current_plan.solar_start_slot <= slot <= self.current_plan.solar_end_slot)
        in_discharge = False
        if self.current_plan.discharge_start_slot <= self.current_plan.solar_start_slot:
            in_discharge = (self.current_plan.discharge_start_slot <= slot < self.current_plan.solar_start_slot)
        else:
            in_discharge = (slot >= self.current_plan.discharge_start_slot) or (slot < self.current_plan.solar_start_slot)

        slots_until_sunrise = self.current_plan.solar_start_slot - slot
        if slots_until_sunrise < 0:
            slots_until_sunrise += 96
            
        print(f"📊 Sensores: SOC={soc_atual*100}% | Carga={load_w}W | Sol={gen_w}W | "
              f"InSolar={in_solar} | InDischarge={in_discharge}")
        
        # 3. Montar State de PyTorch (Compatível com 12 Features do Agente)
        hora_sin = np.sin(2 * np.pi * slot / 96.0)
        hora_cos = np.cos(2 * np.pi * slot / 96.0)
        slots_solar = float(slots_until_sunrise) / 96.0
        soc_vs_alvo = soc_atual - 0.10 # SOC_FLOOR
        na_janela = 1.0 if in_discharge else 0.0
        
        expected_per_slot = self.current_plan.avg_day_gen_wh / (self.current_plan.solar_end_slot - self.current_plan.solar_start_slot + 1) / SLOT_H
        solar_ratio = min(2.0, gen_w / max(1, expected_per_slot))
        
        usable = max(0, soc_atual * 5000.0 - 0.10 * 5000.0)
        deficit_remaining = self.current_plan.avg_night_load_w * slots_until_sunrise * SLOT_H
        razao_cobertura = min(2.0, usable / max(1, deficit_remaining)) if deficit_remaining > 0 else 1.0

        state = np.array([
            soc_atual,                                  # 0: soc
            hora_sin,                                   # 1: hora_sin
            hora_cos,                                   # 2: hora_cos
            slots_solar,                                # 3: slots_solar
            soc_vs_alvo,                                # 4: soc_vs_alvo
            na_janela,                                  # 5: na_janela
            solar_ratio,                                # 6: solar_ratio
            grid_ok,                                    # 7: grid_ok
            razao_cobertura,                            # 8: razao_cobertura
            1.0,                                        # 9: soc_fim_solar (mockup para rede Neural)
            load_w / 2000.0,                            # 10: carga_recente
            1.0 if self.last_decision == VAL_DISCHARGE else 0.0 # 11: ultima_acao
        ], dtype=np.float32)

        # 4. Decisão da Maquina
        action = self.agent.select_action(state, training=False)
        acao_nome = "DISCHARGE ⚡" if action == 1 else "STANDBY ⏸"
        print(f"🤖 IA Decidiu: {acao_nome}")

        # 5. Efetuar a mudanca no Inversor Fisico (se houver alteração de estado)
        novo_valor = VAL_DISCHARGE if action == 1 else VAL_STANDBY
        
        if novo_valor != self.last_decision:
            print(f"⚠️ Ação alterada pelo Modelo! Repassando ordem Modbus pro ESP...")
            self.fb.push_command(REG_WORKMODE, novo_valor)
            self.last_decision = novo_valor

        # 6. Atualizar a vitrine do Aplicativo React/Native (/ai_dashboard)
        self.fb.put_data(PATH_DASHBOARD, {
            "current_slot": slot,
            "dqn_action": acao_nome,
            "agent_1_start": self.current_plan.discharge_start_slot,
            "agent_1_end": self.current_plan.solar_start_slot,
            "soc": soc_atual * 100,
            "load_w": load_w,
            "gen_w": gen_w,
            "updatedAt": str(now)
        })

def main():
    engine = AIEngine()
    engine.fetch_historical_week()
    
    print("\n🚀 Bridge Ativa e Operante em Servidor 24/7.")
    print("O script vai dormir até pingar o ESP32 novamente e comandar a placa Modbus...")
    try:
        while True:
            # Roda 1 Vez
            engine.loop_tick()
            
            # Dorme 15 minutos (900 segs)
            minutos = 15
            print(f"\n💤 Dormindo até o próximo Slot de medição... ({minutos} min)")
            time.sleep(minutos * 60)
            
            # Verifica se já é meia-noite de domingo para re-rolar a semana (0 = Segunda, 6 = Domingo)
            agora = datetime.now()
            if agora.weekday() == 6 and agora.hour == 23 and agora.minute >= 45:
                print("🔄 Virada de semana! Atualizando plano do Agente 1...")
                engine.fetch_historical_week()
    except KeyboardInterrupt:
        print("\n⏹️ Ponte da IA Desligada pelo Operador.")

if __name__ == "__main__":
    main()
