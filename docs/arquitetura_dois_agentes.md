# Arquitetura de Dois Agentes para Gestão de Bateria Solar

## Motivação

O sistema anterior usava um único DQN com 35 features e uma função de recompensa de
9 prioridades tentando resolver dois problemas ao mesmo tempo:

1. **Planejamento semanal:** "Qual é o padrão desta planta? Quando devo descarregar?"
2. **Ajuste fino em tempo real:** "Dado o estado atual, descarrego agora ou espero?"

Essas duas tarefas têm horizontes de tempo e incertezas completamente diferentes.
O resultado era instabilidade de treinamento, catastrophic forgetting e falhas em
cenários básicos.

A nova arquitetura separa as responsabilidades:

```
┌─────────────────────────────────────────────────────────┐
│                     Dados Históricos                    │
│              (semana anterior — 7 × 96 slots)           │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│              AGENTE 1 — Planejador Semanal              │
│           (análise estatística pura, zero ML)           │
│                                                         │
│  • Calcula hora média do nascer/pôr do sol              │
│  • Calcula déficit noturno típico (Wh)                  │
│  • Define janela segura de descarga                     │
│  • Avalia desempenho da semana anterior                 │
│                                                         │
│  Saída: PLANO (dict com parâmetros da semana)           │
└───────────────────────┬─────────────────────────────────┘
                        │  PLANO (atualizado semanalmente)
                        ▼
┌─────────────────────────────────────────────────────────┐
│              AGENTE 2 — Executor (DQN)                  │
│         (12 features, 4 regras de recompensa)           │
│                                                         │
│  • Segue o plano como guia principal                    │
│  • Ajusta se solar hoje < esperado → adia descarga      │
│  • Ajusta se queda detectada → preserva bateria         │
│  • Conservador: só age quando há vantagem clara         │
│                                                         │
│  Saída: STANDBY / DISCHARGE a cada 15 minutos           │
└─────────────────────────────────────────────────────────┘
```

---

## Agente 1 — Planejador Semanal

**Arquivo:** `backend/rl/planner.py`  
**Classe:** `WeeklyPlanner`  
**Treinamento:** nenhum — análise estatística pura

### O que analisa

Para cada dia da semana passada:
- `solar_start`: primeiro slot com geração ≥ 300W (nascer do sol)
- `solar_end`: último slot com geração ≥ 300W (pôr do sol)
- `deficit_noturno`: Wh necessários da rede/bateria antes do sol nascer e após o pôr

### O que produz (formato do plano)

```json
{
  "solar_start_slot_medio": 28,
  "solar_end_slot_medio": 72,
  "desvio_solar_start": 2.5,
  "deficit_noturno_medio_wh": 4200,
  "desvio_deficit_wh": 380,
  "geracao_diaria_media_wh": 14500,
  "carga_diaria_media_wh": 18200,
  "descarga_ideal_inicio_slot": 11,
  "descarga_conservadora_inicio_slot": 17,
  "soc_alvo_fim_noite": 15.0,
  "dias_baixa_geracao": 1,
  "avaliacao_semana": "OK",
  "n_dias_analisados": 7
}
```

### Cálculo da janela de descarga

```
n_slots_necessários = ceil(deficit_noturno_medio / 250 Wh/slot)
descarga_ideal_inicio = solar_start_medio - n_slots_necessários
```

**Exemplo:** déficit = 4200 Wh, solar começa slot 28 (07:00)
- `n_slots = ceil(4200 / 250) = 17 slots (4h15)`
- `descarga_ideal = 28 - 17 = slot 11 = 02:45`
- Interpretação: **não descarregar antes das 02:45**, pois antes disso a bateria não
  tem energia suficiente para cobrir tanto a descarga quanto uma possível queda.

### Isolamento de dados

O Agente 1 **só usa dados da semana anterior** para planejar a próxima.
Nunca tem acesso a dados futuros. Na semana 1, usa um bootstrap com todos
os 30 dias disponíveis.

| Semana executada | Dados usados pelo Agente 1 |
|-----------------|---------------------------|
| S1 (D1–D7)      | Bootstrap (todos os 30 dias) |
| S2 (D8–D14)     | Apenas D1–D7 |
| S3 (D15–D21)    | Apenas D8–D14 |
| S4 (D22–D28)    | Apenas D15–D21 |
| D29–D30         | Plano da S4 (sem atualização) |

---

## Agente 2 — Executor (DQN Simplificado)

**Arquivo:** `backend/rl/environment_v2.py`, `backend/rl/reward_v2.py`  
**Classe:** `BatteryEnvV2`  
**Treinamento:** DQN com 12 features, 4 regras de recompensa

### Estado (12 features)

| # | Feature | Descrição |
|---|---------|-----------|
| 0 | `soc` | SOC atual normalizado (0–1) |
| 1 | `hora_sin` | Hora do dia — componente seno |
| 2 | `hora_cos` | Hora do dia — componente cosseno |
| 3 | `slots_ate_solar` | Slots até o solar (normalizado) |
| 4 | `soc_vs_alvo` | SOC acima/abaixo do alvo do plano |
| 5 | `janela_ideal` | 1 se dentro da janela ideal de descarga |
| 6 | `janela_conservadora` | 1 se dentro da janela conservadora |
| 7 | `solar_ratio` | Geração de hoje / esperado pelo plano |
| 8 | `grid_ok` | 1 = rede normal, 0 = queda |
| 9 | `razao_cobertura` | energia_util / deficit_restante (normalizado) |
| 10 | `soc_fim_solar_ontem` | SOC ao final do período solar do dia anterior |
| 11 | `last_action` | Ação anterior (0 ou 1) |

### Função de recompensa (4 regras)

```
REGRA 1: Sobrevivência à queda
  Durante queda: -200 × (energia_desperdiçada / energia_máxima_utilizável)
  Sobreviver: +3

REGRA 2: Reserva de segurança
  SOC < 10%: -100
  SOC entre 10-15%: -1 por ponto

REGRA 3: SOC alvo ao amanhecer
  |SOC - alvo| ≤ 5%:  +15  (perfeito)
  |SOC - alvo| ≤ 10%: +10  (bom)
  |SOC - alvo| ≤ 20%: +5   (razoável)
  SOC > alvo + 30%:   -10  (bateria ociosa — desperdiçou noite)

REGRA 4: Janela do plano
  DISCHARGE dentro da janela com razão ≥ 1:  +8
  DISCHARGE fora da janela:                  -5
  DISCHARGE com razão < 1 (bateria insuficiente): -10 × (1 - razão)
```

### Comportamento esperado

| Situação | Ação esperada |
|----------|--------------|
| Antes da janela de descarga | STANDBY (preserva para queda) |
| Na janela + razão ≥ 1 | DISCHARGE (seguro e econômico) |
| Na janela + razão < 1 | STANDBY (bateria não cobre déficit) |
| Solar fraco hoje (< 80% esperado) | Adia para janela conservadora |
| Queda de energia (grid = 0) | Não interfere (inversor assume) |
| Sol nunca veio no dia | STANDBY todo dia (preserva bateria) |

---

## Treinamento

**Arquivo:** `backend/rl/train_v2.py`

```bash
# Treinamento padrão (500 episódios)
py backend/rl/train_v2.py

# Treinamento rápido para teste
py backend/rl/train_v2.py --episodes 200 --log-freq 10
```

**Hiperparâmetros:**

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| `lr` | 0.0005 | Maior que o v1 (estado simpler → gradiente mais limpo) |
| `gamma` | 0.97 | Horizonte médio: cada episódio é de 1-3 dias |
| `epsilon_decay` | 0.995 | Mais rápido que v1 (plano reduz espaço de busca) |
| `buffer_size` | 50.000 | Menor que v1 (estado menor, mais eficiente) |
| `target_update` | a cada 25 ep | Frequente (episódios curtos) |
| `replay_every` | a cada 2 steps | Aproveita mais as transições |

**Modelo salvo em:** `backend/rl/model/battery_dqn_v2.pth`

---

## Verificação e validação

### Teste do Agente 1 (planner.py)

```bash
py backend/rl/planner.py
```

Saída esperada: 4 planos (bootstrap + semanas 1-3) com:
- Solar start próximo de slot 28 (~07:00)
- Déficit noturno próximo de 4000-5000 Wh
- Janela de descarga a partir de slot 11-16 (~02:45–04:00)

### Teste do treinamento completo

```bash
py backend/rl/train_v2.py --episodes 300
```

No checkpoint dos 30 dias, verificar:
- Semanas sem violações de reserva
- Energia economizada > 0 (está descarregando)
- SOC final de cada semana próximo de 80-100% (solar recarregou)

### Cenário 3d_D1_00h_queda

O novo sistema deve passar porque:
1. O plano proíbe descarga antes da janela (~02:45)
2. O solar do D1 recarrega a bateria até ~100% (não foi desperdiçada)
3. A bateria chega ao D2 00:00 com SOC ≥ 90% → cobre os 4200 Wh da queda

---

## Comparação com sistema anterior

| Critério | DQN v1 (single agent) | Dois Agentes (v2) |
|----------|----------------------|-------------------|
| State size | 35 features | 12 features |
| Reward rules | 9 prioridades | 4 regras |
| Episódios necessários | 2000+ | ~500 |
| Tempo de treinamento | ~2h | ~15min |
| Interpretabilidade | Baixa | Alta (plano legível) |
| Adaptação a outras plantas | Baixa | Alta (Agente 1 aprende padrões) |
| Falha 3d_D1_00h_queda | Sim | Não (plano previne) |
| Complexidade de debug | Alta | Baixa (dois componentes separáveis) |

---

## Arquivos do sistema

```
backend/rl/
├── planner.py          ← NOVO: Agente 1 (análise semanal)
├── environment_v2.py   ← NOVO: Ambiente para Agente 2 (12 features)
├── reward_v2.py        ← NOVO: Recompensa simplificada (4 regras)
├── train_v2.py         ← NOVO: Loop de treinamento v2
├── dqn_agent.py        ← Reutilizado sem modificação
├── environment.py      ← Mantido para referência/comparação
├── reward.py           ← Mantido para referência/comparação
└── model/
    ├── battery_dqn_v2.pth       ← Modelo de produção v2
    ├── battery_dqn_v2_best.pth  ← Melhor checkpoint v2
    └── model_v2_info.json       ← Metadados do modelo v2
```
