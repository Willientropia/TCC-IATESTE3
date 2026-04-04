"""
data_loader.py — Carrega dados de treinamento em slots de 15 minutos.

Suporta dois formatos:
1. leituras.json (legado) — dados da plataforma de monitoramento
2. training_data_real.json (novo) — gerado pelo csv_to_training_json.py

Output: Lista de dicts, cada um representando 1 dia com arrays de 96 slots.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


# Campos do leituras.json (legado)
FIELD_LOAD = "Load Combined Power"
FIELD_GRID = "Coleta do Medidor"
FIELD_SOC = "soc"

SLOTS_PER_DAY = 96
SLOT_DURATION_MIN = 15


def _time_key_to_minutes(key: str) -> float:
    """Converte chave 'HH-MM-SS' para minutos do dia."""
    parts = key.split("-")
    h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
    return h * 60 + m + s / 60


def _slot_for_minute(minute: float) -> int:
    """Retorna índice do slot (0-95) para um minuto do dia."""
    return min(int(minute // SLOT_DURATION_MIN), SLOTS_PER_DAY - 1)


# ─── Formato Novo: training_data_real.json ──────────────────────

def load_training_json(path: str | Path) -> list[dict]:
    """
    Carrega o training_data_real.json (gerado pelo csv_to_training_json.py).
    
    Retorna lista de dicts com:
    - date: str
    - load: ndarray(96,) em W
    - gen: ndarray(96,) em W
    - soc: ndarray(96,) em %
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    days = []
    for day_data in data["days"]:
        load = np.array(day_data["load"], dtype=np.float32)
        gen = np.array(day_data["generation"], dtype=np.float32)
        soc = np.array(day_data["soc_real"], dtype=np.float32)
        
        if len(load) != SLOTS_PER_DAY:
            continue  # Dia incompleto
        
        days.append({
            "date": day_data["date"],
            "load": load,
            "gen": gen,
            "soc": soc,
        })
    
    return days


# ─── Formato Legado: leituras.json ─────────────────────────────

def load_raw_json(path: str | Path) -> dict:
    """Carrega o JSON bruto (leituras.json)."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def process_readings(raw: dict) -> pd.DataFrame:
    """
    Processa o leituras.json em um DataFrame com slots de 15 minutos.
    
    Retorna DataFrame com colunas:
    - date: str (YYYY-MM-DD)
    - slot: int (0-95)
    - load_w: float (potência média de carga em W)
    - generation_w: float (potência média de geração solar em W)
    - soc: float (SOC médio, 0-100)
    """
    rows = []
    
    for date_str, readings in sorted(raw.items()):
        slot_data = {s: {"load": [], "gen": [], "soc": []} for s in range(SLOTS_PER_DAY)}
        
        for time_key, values in readings.items():
            minute = _time_key_to_minutes(time_key)
            slot = _slot_for_minute(minute)
            
            load = values.get(FIELD_LOAD, 0) or 0
            grid = values.get(FIELD_GRID, 0) or 0
            soc = values.get(FIELD_SOC, 0) or 0
            
            load = max(0, float(load))
            grid = float(grid)
            # Geração solar = carga + excedente exportado para rede
            gen = max(0, load + grid)
            
            slot_data[slot]["load"].append(load)
            slot_data[slot]["gen"].append(gen)
            slot_data[slot]["soc"].append(float(soc))
        
        for slot in range(SLOTS_PER_DAY):
            sd = slot_data[slot]
            if sd["load"]:
                rows.append({
                    "date": date_str,
                    "slot": slot,
                    "load_w": np.mean(sd["load"]),
                    "generation_w": np.mean(sd["gen"]),
                    "soc": np.mean(sd["soc"]),
                })
            else:
                rows.append({
                    "date": date_str,
                    "slot": slot,
                    "load_w": np.nan,
                    "generation_w": np.nan,
                    "soc": np.nan,
                })
    
    df = pd.DataFrame(rows)
    df["load_w"] = df["load_w"].interpolate(method="linear").fillna(0)
    df["generation_w"] = df["generation_w"].interpolate(method="linear").fillna(0)
    df["soc"] = df["soc"].interpolate(method="linear").fillna(50)
    
    return df


def load_and_process(path: str | Path) -> pd.DataFrame:
    """Convenience: carrega e processa leituras.json em uma chamada."""
    raw = load_raw_json(path)
    return process_readings(raw)


def get_day_arrays(df: pd.DataFrame, date: str) -> tuple[np.ndarray, np.ndarray]:
    """Retorna (load_w[96], generation_w[96]) para um dia específico."""
    day = df[df["date"] == date].sort_values("slot")
    return day["load_w"].values, day["generation_w"].values


def get_all_days(df: pd.DataFrame) -> list[dict]:
    """
    Retorna lista de dicts, um por dia, com arrays numpy.
    Cada dict: {"date", "load" (96,), "gen" (96,), "soc" (96,)}
    """
    days = []
    for date, group in df.groupby("date", sort=True):
        group = group.sort_values("slot")
        if len(group) < SLOTS_PER_DAY:
            continue
        days.append({
            "date": date,
            "load": group["load_w"].values.copy(),
            "gen": group["generation_w"].values.copy(),
            "soc": group["soc"].values.copy(),
        })
    return days


# ─── Carregador Universal ──────────────────────────────────────

def load_data(path: str | Path) -> list[dict]:
    """
    Carrega dados de qualquer formato suportado.
    Detecta automaticamente pelo conteúdo do arquivo.
    
    Retorna lista de dicts com: {"date", "load" (96,), "gen" (96,), "soc" (96,)}
    """
    path = Path(path)
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Detectar formato: training_data_real.json tem "metadata" e "days"
    if isinstance(data, dict) and "metadata" in data and "days" in data:
        print(f"  📋 Formato detectado: training_data_real.json")
        return load_training_json(path)
    else:
        # Formato legado (leituras.json)
        print(f"  📋 Formato detectado: leituras.json (legado)")
        df = process_readings(data)
        return get_all_days(df)


# ─── Cenários de Teste ──────────────────────────────────────────

def load_scenario_json(path: str | Path) -> dict:
    """Carrega cenário de teste (3d ou 7d) no formato do visualizador."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "nome": data.get("nome", ""),
        "nDias": data.get("nDias", 1),
        "socInicial": data.get("socInicial", 99),
        "load": np.array(data["load"], dtype=np.float32),
        "generation": np.array(data["generation"], dtype=np.float32),
        "slotLabels": data.get("slotLabels", []),
        "descricao": data.get("descricao", ""),
    }


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "dados/training_data_real.json"
    
    print(f"Carregando {path}...")
    days = load_data(path)
    
    print(f"Total: {len(days)} dias completos")
    print(f"Datas: {days[0]['date']} a {days[-1]['date']}")
    print(f"\nExemplo dia {days[0]['date']}:")
    print(f"  Carga média: {days[0]['load'].mean():.0f} W")
    print(f"  Geração média: {days[0]['gen'].mean():.0f} W")
    print(f"  Geração máx: {days[0]['gen'].max():.0f} W")
    print(f"  SOC médio: {days[0]['soc'].mean():.1f}%")
