# Plano de Implementação — IA de Gestão de Bateria Solar

## Visão Geral

Implementar um sistema de dois agentes que decide, a cada 15 minutos, se a bateria deve **descarregar** (economizando rede) ou ficar em **standby** (preservando para emergência).

---

## Comportamento Esperado da IA

### Regras Absolutas (invioláveis)

1. **NUNCA** descarregar durante o período solar (bateria deve carregar)
2. **NUNCA** descarregar no início da noite (bateria preservada para queda)
3. **NUNCA** alternar entre standby e descarga sem motivo (sem flickering)
4. **NUNCA** chegar ao amanhecer com mais de 15% de bateria (desperdiçou economia)
5. **NUNCA** deixar a planta sem energia durante uma queda suprível

### Ciclo Diário Ideal

```
  06:45          17:30            ~01:00          06:45
    │              │                │               │
    ├──────────────┤────────────────┤───────────────┤
    │  ☀️ SOLAR    │  ⏸ STANDBY     │  ⚡ DESCARGA  │
    │  Bat carrega │  Bat = ~100%   │  Bat → ~12%   │
    │  até ~100%   │  (preserva)    │  (contínuo)   │
```

### Cálculo da Hora de Início da Descarga

```
hora_inicio = nascer_sol - (energia_utilizavel / carga_media_noturna)

Exemplo:
  Bateria 100% = 5000 Wh
  Alvo amanhecer = 12% = 600 Wh
  Utilizável = 4400 Wh
  Carga média noturna = 750W
  Horas = 4400 / 750 = 5h50
  Sol nasce: 06:45
  Início descarga: 00:55
```

---

## Arquitetura: Dois Agentes

### Agente 1 — Planejador Semanal (sem ML)

**Arquivo:** `backend/rl/planner.py`

Análise estatística pura dos dados da semana anterior. Produz um **plano** que guia o Agente 2.

**Input:** 7 dias de dados históricos (load, generation por slot)

**Output (plano da semana):**
```json
{
  "solar_start_slot": 27,
  "solar_end_slot": 70,
  "deficit_noturno_medio_wh": 4200,
  "carga_noturna_media_w": 750,
  "geracao_diaria_media_wh": 14500,
  "descarga_inicio_slot_ideal": 4,
  "soc_alvo_amanhecer_pct": 12
}
```

**Implementação:**
- Calcula hora média do nascer/pôr do sol (onde geração ≥ 300W)
- Calcula déficit noturno típico em Wh
- Define slot ideal de início de descarga
- Recalcula automaticamente a cada 7 dias

**Adaptação:** Quando novos dados reais chegam do inversor, o plano se atualiza automaticamente (é só refazer as médias).

---

### Agente 2 — Executor DQN

**Arquivos:** `backend/rl/environment.py`, `backend/rl/dqn_agent.py`, `backend/rl/reward.py`

Rede neural pequena (~12.500 parâmetros) que toma decisões em tempo real usando o plano como guia.

**Estado (12 features):**

| # | Feature | Descrição | Range |
|---|---------|-----------|-------|
| 0 | `soc` | SOC normalizado | 0–1 |
| 1 | `hora_sin` | sin(2π × slot/96) | -1 a 1 |
| 2 | `hora_cos` | cos(2π × slot/96) | -1 a 1 |
| 3 | `slots_ate_solar` | Normalizado por 96 | 0–1 |
| 4 | `soc_vs_alvo` | SOC - alvo do plano | -1 a 1 |
| 5 | `na_janela_descarga` | 1 se dentro da janela | 0 ou 1 |
| 6 | `solar_ratio` | Geração hoje / esperado | 0–2+ |
| 7 | `grid_ok` | 1 = rede normal | 0 ou 1 |
| 8 | `razao_cobertura` | energia_util / deficit_restante | 0–2+ |
| 9 | `soc_fim_solar_ontem` | SOC ao fim do solar anterior | 0–1 |
| 10 | `carga_recente` | Carga média últimos 4 slots / 2000 | 0–2+ |
| 11 | `ultima_acao` | Ação do slot anterior | 0 ou 1 |

**Ações:**
- `0` = STANDBY
- `1` = DISCHARGE

**Rede:**
```
Input(12) → Hidden(128, ReLU) → Hidden(64, ReLU) → Output(2)
Total: ~12.500 parâmetros
```

---

## Função de Recompensa

A recompensa **codifica** as 5 regras absolutas como sinais numéricos:

```python
def calcular_recompensa(estado, acao, novo_estado, plano):
    reward = 0.0

    # REGRA 1: Sobreviver à queda (punição máxima)
    if queda_ativa and soc < 0.10:
        reward -= 200  # Catastrófico

    if queda_ativa and soc >= 0.10:
        reward += 5    # Sobreviveu

    # REGRA 2: Reserva mínima
    if soc < 0.10:
        reward -= 100

    # REGRA 3: SOC ao amanhecer (descarregou o suficiente?)
    if fim_da_noite:
        if soc <= 0.15:
            reward += 15   # Perfeito — usou a bateria!
        elif soc <= 0.25:
            reward += 5    # OK
        elif soc > 0.50:
            reward -= 15   # Ruim — desperdiçou economia

    # REGRA 4: Descarga na janela do plano
    if acao == DISCHARGE:
        if na_janela_descarga:
            reward += 3    # Correto
        else:
            reward -= 10   # Fora da janela!

    if acao == STANDBY:
        if na_janela_descarga and soc > 0.15:
            reward -= 3    # Deveria estar descarregando!

    # REGRA 5: Estabilidade (não flickar)
    if acao != ultima_acao and not mudou_periodo:
        reward -= 2        # Penaliza troca sem motivo

    return reward
```

---

## Treinamento

### Dados de Treino

Usando `dados/leituras.json`:
- 44 dias de dados reais
- Processados em slots de 15 min (96/dia)
- Separados em episódios de 1-3 dias

### Processo

```
1. Carregar leituras.json → DataFrame (load_w, generation_w, por slot)
2. Construir perfis EWMA (Agente 1)
3. Para cada episódio (1-3 dias aleatórios):
   a. Agente 1 gera o plano semanal
   b. Agente 2 percorre os slots decidindo STANDBY/DISCHARGE
   c. Bateria simulada aplica física real
   d. Recompensa calculada a cada slot
   e. Experiência armazenada no replay buffer
   f. Rede treinada com batch aleatório
4. Repetir por 500-1000 episódios
5. Salvar melhor modelo
```

### Quedas Simuladas

Durante o treino, com ~15% de probabilidade, um período de 3-5h tem `grid=0` forçado. Isso ensina o agente sobre a importância da reserva.

### Hiperparâmetros

| Parâmetro | Valor |
|-----------|-------|
| Episódios | 500-1000 |
| Batch size | 64 |
| Learning rate | 0.0005 |
| Gamma (desconto) | 0.97 |
| Epsilon início → fim | 1.0 → 0.01 |
| Epsilon decay | 0.995 |
| Replay buffer | 50.000 |
| Target network update | a cada 25 episódios |

---

## Validação

### Cenários de Teste

1. **3d_D1_00h_queda** — Verificar sobrevivência à queda de 5h
2. **prova_7dias** — Verificar economia contínua com dias nublados e queda

### Métricas

| Métrica | Alvo |
|---------|------|
| Sobrevivência a quedas supríveis | 100% |
| SOC médio ao amanhecer | 10-15% |
| Economia de rede vs STANDBY puro | > 3000 Wh/dia |
| Estabilidade (trocas STANDBY↔DISCHARGE por noite) | ≤ 2 |
| Descarga durante período solar | 0 slots |

### Comparação

Comparar lado a lado:
- **Sem IA** (STANDBY puro): zero economia
- **Regras fixas** (MPC/EWMA): economia com regras if/else
- **DQN treinado**: economia com decisões aprendidas

---

## Adaptação em Produção

### Agente 1 (Planejador)
- Recalcula automaticamente com dados da última semana
- Sem retreinamento — é cálculo estatístico

### Agente 2 (DQN)
- **Modo conservador:** modelo fixo, adapta via features normalizadas
- **Modo avançado:** fine-tuning semanal com 50-100 episódios usando dados reais novos
- **Proteção:** se recompensa média cair, reverte para modelo anterior

---

## Ordem de Implementação

1. **`backend/rl/planner.py`** — Agente 1 (planejador semanal)
2. **`backend/rl/environment.py`** — Ambiente de simulação
3. **`backend/rl/reward.py`** — Função de recompensa
4. **`backend/rl/dqn_agent.py`** — Rede neural DQN
5. **`backend/rl/train.py`** — Loop de treinamento
6. **Treinar e validar** com os cenários de teste
7. **Integrar** com o simulador visual
