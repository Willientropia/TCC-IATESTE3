"""
csv_to_training_json.py — Processa CSVs do inversor (REALTIME_REPORT) e gera
um JSON de treinamento com dados reais agrupados em slots de 15 minutos.

Campos usados do CSV:
  - time                    → timestamp
  - PV total Power/kW       → geração solar (kW → W)
  - Load Total Power/kW     → carga total (kW → W)
  - Grid Power/kW           → potência da rede (positivo=import, negativo=export)
  - SoC/%                   → estado de carga da bateria
"""

import csv
import json
import sys
import os
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

# ─── Configurações ───────────────────────────────────────────────
SLOT_DURATION_MIN = 15
SLOTS_PER_DAY = 96

# Colunas do CSV (com possíveis variações de nome)
COL_TIME = "time"
COL_PV_POWER = "PV total Power/kW"
COL_LOAD_POWER = "Load Total Power/kW"
COL_GRID_POWER = "Grid Power/kW"
COL_SOC = "SoC/%"
COL_BAT_POWER = "BAT Power(INV)/kW"


def parse_csv(filepath: str) -> list[dict]:
    """Lê um CSV do inversor e retorna lista de registros limpos."""
    records = []
    
    with open(filepath, "r", encoding="utf-8") as f:
        # Pular a primeira linha (SN, serial, etc.)
        first_line = f.readline()
        if not first_line.strip().startswith("time"):
            # A segunda linha tem os headers reais
            reader = csv.DictReader(f)
        else:
            # Se a primeira linha já é header
            f.seek(0)
            reader = csv.DictReader(f)
        
        for row in reader:
            time_str = row.get(COL_TIME, "").strip()
            if not time_str or len(time_str) < 10:
                continue
            
            try:
                dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M")
            except ValueError:
                continue
            
            def safe_float(val):
                try:
                    v = float(val.strip()) if val and val.strip() else 0.0
                    return v
                except (ValueError, AttributeError):
                    return 0.0
            
            pv_kw = safe_float(row.get(COL_PV_POWER, "0"))
            load_kw = safe_float(row.get(COL_LOAD_POWER, "0"))
            grid_kw = safe_float(row.get(COL_GRID_POWER, "0"))
            soc = safe_float(row.get(COL_SOC, "0"))
            bat_kw = safe_float(row.get(COL_BAT_POWER, "0"))
            
            records.append({
                "datetime": dt,
                "date": dt.strftime("%Y-%m-%d"),
                "minute": dt.hour * 60 + dt.minute,
                "pv_w": max(0, pv_kw * 1000),          # kW → W, nunca negativo
                "load_w": max(0, load_kw * 1000),       # kW → W, nunca negativo
                "grid_w": grid_kw * 1000,                # kW → W (mantém sinal)
                "soc": max(0, min(100, soc)),            # Clamp 0-100
                "bat_w": bat_kw * 1000,                  # kW → W (positivo=descarga)
            })
    
    return records


def group_into_slots(records: list[dict]) -> dict:
    """
    Agrupa registros por dia e slot de 15 min.
    Retorna: {date_str: {slot_idx: {pv_w: [...], load_w: [...], ...}}}
    """
    days = defaultdict(lambda: defaultdict(lambda: {
        "pv_w": [], "load_w": [], "grid_w": [], "soc": [], "bat_w": []
    }))
    
    for rec in records:
        slot = min(int(rec["minute"] // SLOT_DURATION_MIN), SLOTS_PER_DAY - 1)
        day = days[rec["date"]][slot]
        day["pv_w"].append(rec["pv_w"])
        day["load_w"].append(rec["load_w"])
        day["grid_w"].append(rec["grid_w"])
        day["soc"].append(rec["soc"])
        day["bat_w"].append(rec["bat_w"])
    
    return days


def build_training_data(days: dict) -> list[dict]:
    """
    Converte dados agrupados em formato de treinamento.
    Cada dia = um dict com arrays de 96 slots.
    """
    training_days = []
    
    for date_str in sorted(days.keys()):
        day_slots = days[date_str]
        
        pv_arr = np.zeros(SLOTS_PER_DAY)
        load_arr = np.zeros(SLOTS_PER_DAY)
        grid_arr = np.zeros(SLOTS_PER_DAY)
        soc_arr = np.full(SLOTS_PER_DAY, np.nan)
        bat_arr = np.zeros(SLOTS_PER_DAY)
        
        filled_slots = 0
        for slot_idx in range(SLOTS_PER_DAY):
            if slot_idx in day_slots and day_slots[slot_idx]["pv_w"]:
                sd = day_slots[slot_idx]
                pv_arr[slot_idx] = np.mean(sd["pv_w"])
                load_arr[slot_idx] = np.mean(sd["load_w"])
                grid_arr[slot_idx] = np.mean(sd["grid_w"])
                soc_arr[slot_idx] = np.mean(sd["soc"])
                bat_arr[slot_idx] = np.mean(sd["bat_w"])
                filled_slots += 1
        
        # Interpolar slots vazios
        if filled_slots < 10:
            print(f"  ⚠ Dia {date_str}: apenas {filled_slots} slots preenchidos, pulando...")
            continue
        
        # Interpolação linear para slots faltantes
        valid = ~np.isnan(soc_arr)
        if not np.all(valid):
            indices = np.arange(SLOTS_PER_DAY)
            for arr in [pv_arr, load_arr, grid_arr, soc_arr, bat_arr]:
                mask = np.isnan(arr) if np.any(np.isnan(arr)) else (arr == 0) & ~valid
                if np.any(np.isnan(arr)):
                    arr[np.isnan(arr)] = np.interp(
                        indices[np.isnan(arr)],
                        indices[~np.isnan(arr)],
                        arr[~np.isnan(arr)]
                    )
        
        # Gerar labels de tempo
        labels = []
        for s in range(SLOTS_PER_DAY):
            h = (s * SLOT_DURATION_MIN) // 60
            m = (s * SLOT_DURATION_MIN) % 60
            labels.append(f"{h:02d}:{m:02d}")
        
        day_data = {
            "date": date_str,
            "slots_filled": filled_slots,
            "load": [round(float(x), 1) for x in load_arr],
            "generation": [round(float(x), 1) for x in pv_arr],
            "grid": [round(float(x), 1) for x in grid_arr],
            "soc_real": [round(float(x), 1) for x in soc_arr],
            "bat_power": [round(float(x), 1) for x in bat_arr],
            "slotLabels": labels,
        }
        
        training_days.append(day_data)
    
    return training_days


def print_summary(training_days: list[dict]):
    """Imprime um resumo dos dados processados."""
    print(f"\n{'='*60}")
    print(f"  RESUMO DO DATASET DE TREINAMENTO")
    print(f"{'='*60}")
    print(f"  Total de dias: {len(training_days)}")
    
    if training_days:
        print(f"  Período: {training_days[0]['date']} a {training_days[-1]['date']}")
        
        print(f"\n  {'Dia':<12} {'Slots':<8} {'Carga Méd':<12} {'PV Méd':<12} {'PV Máx':<12} {'SOC Ini':<10} {'SOC Fim':<10}")
        print(f"  {'-'*12} {'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*10} {'-'*10}")
        
        for day in training_days:
            load_mean = np.mean(day["load"])
            pv_mean = np.mean(day["generation"])
            pv_max = max(day["generation"])
            soc_start = day["soc_real"][0]
            soc_end = day["soc_real"][-1]
            
            print(f"  {day['date']:<12} {day['slots_filled']:<8} "
                  f"{load_mean:>8.0f} W  {pv_mean:>8.0f} W  {pv_max:>8.0f} W  "
                  f"{soc_start:>6.0f}%   {soc_end:>6.0f}%")
    
    print(f"{'='*60}\n")


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    
    # CSVs de entrada
    csv_files = [
        project_root / "REALTIME_REPORT_ROW_J05H502051HB122_30.csv",
        project_root / "REALTIME_REPORT_ROW_J05H502051HB122_30 (1).csv",
    ]
    
    # Arquivo de saída
    output_path = project_root / "dados" / "training_data_real.json"
    
    print("🔄 Processando CSVs do inversor...")
    
    all_records = []
    for csv_file in csv_files:
        if csv_file.exists():
            print(f"  📂 Lendo {csv_file.name}...")
            records = parse_csv(str(csv_file))
            print(f"     → {len(records)} registros encontrados")
            all_records.extend(records)
        else:
            print(f"  ⚠ Arquivo não encontrado: {csv_file.name}")
    
    if not all_records:
        print("❌ Nenhum registro encontrado!")
        sys.exit(1)
    
    # Ordenar por datetime
    all_records.sort(key=lambda r: r["datetime"])
    print(f"\n  Total de registros: {len(all_records)}")
    print(f"  Período: {all_records[0]['datetime']} a {all_records[-1]['datetime']}")
    
    # Agrupar em slots de 15 minutos
    print("\n🔄 Agrupando em slots de 15 minutos...")
    days = group_into_slots(all_records)
    
    # Construir dados de treinamento
    print("🔄 Construindo dados de treinamento...")
    training_days = build_training_data(days)
    
    # Resumo
    print_summary(training_days)
    
    # Criar JSON final
    output_data = {
        "metadata": {
            "description": "Dados reais do inversor para treinamento DQN",
            "source": "REALTIME_REPORT CSVs",
            "slot_duration_min": SLOT_DURATION_MIN,
            "slots_per_day": SLOTS_PER_DAY,
            "total_days": len(training_days),
            "period": f"{training_days[0]['date']} a {training_days[-1]['date']}" if training_days else "",
            "fields": {
                "load": "Potência de carga em W (consumo total da planta)",
                "generation": "Potência de geração solar PV em W",
                "grid": "Potência da rede em W (positivo=importando, negativo=exportando)",
                "soc_real": "Estado de carga real da bateria (%)",
                "bat_power": "Potência da bateria em W (positivo=descarregando)"
            }
        },
        "days": training_days
    }
    
    os.makedirs(output_path.parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✅ JSON salvo em: {output_path}")
    print(f"   Tamanho: {file_size_mb:.2f} MB")
    print(f"   {len(training_days)} dias prontos para treinamento!")


if __name__ == "__main__":
    main()
