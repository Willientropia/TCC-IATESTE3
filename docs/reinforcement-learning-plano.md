# Plano de Implementação — Reinforcement Learning para Gestão de Bateria Solar

**Objetivo:** Substituir o sistema de regras fixas (`decide()`) por um agente de RL que aprenda as prioridades a partir do treinamento, sem que elas sejam codificadas como `if/else`.

---

## 1. Qual Tipo de Reinforcement Learning?

### Recomendação: **DQN (Deep Q-Network)**

**Por quê DQN e não outros?**

| Algoritmo | Prós | Contras para este projeto |
|-----------|------|--------------------------|
| **Q-Learning tabular** | Simples, interpretável | Estado é contínuo (SOC é float), a tabela seria infinita |
| **DQN** ✅ | Funciona com estado contínuo, ação discreta, bem documentado, fácil de explicar em TCC | Precisa de replay buffer, target network |
| **PPO** | Mais estável, state-of-the-art | Mais complexo de implementar e explicar, overkill para ação binária |
| **A3C/A2C** | Paralelismo, bom para ambientes rápidos | Desnecessário — o ambiente é trivialmente rápido |
| **SAC** | Ótimo para ações contínuas | Ação é discreta (DISCHARGE/STANDBY), SAC é para ações contínuas |
| **DDPG** | Controle contínuo | Mesmo problema do SAC — ação discreta |

**DQN é a escolha certa porque:**
1. O espaço de ações é **discreto e pequeno** (2-3 ações)
2. O espaço de estados é **contínuo mas de baixa dimensão** (~20-30 features)
3. É **academicamente sólido** — o paper original (Mnih et al., 2015, DeepMind/Atari) é um dos mais citados em RL
4. É **simples de implementar** com PyTorch (~200 linhas)
5. É **fácil de explicar** numa apresentação de TCC

### O que é DQN em uma frase?

Uma rede neural que aprende a estimar "quanto vale cada ação em cada estado" — o **Q-value**. A cada momento, o agente escolhe a ação com maior Q-value. A rede é treinada comparando a recompensa real com a estimativa, e ajustando os pesos para reduzir o erro.

---

## 2. Arquitetura da Rede Neural

### 2.1 Vetor de Estado (Input) — O que a IA "vê"

```
Estado = [
    # Bateria
    soc_normalizado,           # SOC / 100 → [0, 1]

    # Tempo (codificação cíclica para o modelo entender que 23:45 é perto de 00:00)
    sin(2π × slot / 96),       # componente seno do horário
    cos(2π × slot / 96),       # componente cosseno do horário

    # Dia da semana (codificação cíclica)
    sin(2π × dia_semana / 7),
    cos(2π × dia_semana / 7),

    # Geração solar recente (últimos 4 slots = 1 hora de contexto)
    gen_t0 / 3000,             # normalizado pelo pico típico
    gen_t1 / 3000,
    gen_t2 / 3000,
    gen_t3 / 3000,

    # Carga recente (últimos 4 slots)
    load_t0 / 2000,
    load_t1 / 2000,
    load_t2 / 2000,
    load_t3 / 2000,

    # Previsão EWMA comprimida (próximas 24h em 8 blocos de 3h)
    ewma_gen_bloco_0,          # média dos slots 0-11 (próximas 3h)
    ewma_gen_bloco_1,          # média dos slots 12-23 (3-6h)
    ewma_gen_bloco_2,          # 6-9h
    ewma_gen_bloco_3,          # 9-12h
    ewma_gen_bloco_4,          # 12-15h
    ewma_gen_bloco_5,          # 15-18h
    ewma_gen_bloco_6,          # 18-21h
    ewma_gen_bloco_7,          # 21-24h

    ewma_load_bloco_0..7,      # mesma coisa para carga (8 valores)

    # Tendência de geração (feature que resolve a fragilidade #1)
    tendencia_geracao_7d,      # slope da geração média dos últimos 7 dias
                               # positivo = melhorando, negativo = piorando

    # SOC ao final do período solar do dia anterior
    soc_fim_solar_ontem,       # 0-1, indica se ontem carregou bem
]

Total: 1 + 2 + 2 + 4 + 4 + 8 + 8 + 1 + 1 = 31 features
```

### 2.2 Espaço de Ações (Output)

```
Ação 0: STANDBY          — bateria não age
Ação 1: FORCE_DISCHARGE  — bateria cobre o déficit
```

**Nota:** Poderíamos adicionar `Ação 2: PARTIAL_DISCHARGE (50%)` no futuro, mas para o TCC a decisão binária é suficiente e mais fácil de avaliar.

### 2.3 Topologia da Rede

```
Input (31 neurônios)
    │
    ▼
Hidden Layer 1 (128 neurônios, ReLU)
    │
    ▼
Hidden Layer 2 (64 neurônios, ReLU)
    │
    ▼
Output (2 neurônios — Q-values para STANDBY e DISCHARGE)
```

### 2.4 Contagem de Parâmetros

```
Camada 1:  31 × 128 + 128 (bias) =  4.096 parâmetros
Camada 2: 128 ×  64 +  64 (bias) =  8.256 parâmetros
Saída:     64 ×   2 +   2 (bias) =    130 parâmetros
                                   ─────────
Total:                             12.482 parâmetros
```

**Perspectiva:** Isso é uma rede *minúscula*. O GPT-4 tem ~1.8 trilhão de parâmetros. Um modelo de classificação de imagens típico tem milhões. Com ~12.500 parâmetros, o treinamento leva segundos ou poucos minutos em CPU — não precisa de GPU.

---

## 3. A Função de Recompensa — Como Codificar as Prioridades

**Esta é a parte mais importante.** No RL, as prioridades não são `if/else` — elas são codificadas como **recompensas e punições numéricas**. O agente aprende a maximizar a recompensa acumulada.

### 3.1 Escala de Prioridades

```python
def calcular_recompensa(estado_anterior, acao, estado_novo, contexto):
    """
    A recompensa é calculada APÓS a ação ser executada.
    O agente aprende a maximizar a soma de recompensas ao longo do episódio.
    """
    reward = 0.0

    soc_antes = estado_anterior['soc']
    soc_depois = estado_novo['soc']
    gen_atual = contexto['generation']
    load_atual = contexto['load']
    deficit = max(0, load_atual - gen_atual)
    hora = contexto['hora']  # 0-23.75

    # ═══════════════════════════════════════════════════════════
    # PRIORIDADE 1 (MÁXIMA): Ter reserva para queda de energia
    # ═══════════════════════════════════════════════════════════
    # Se SOC cair abaixo da reserva: punição CATASTRÓFICA
    # O agente aprende que isso NUNCA deve acontecer
    if soc_depois < RESERVA_MINIMA:
        reward -= 100.0

    # Simulação de queda de energia aleatória durante treinamento
    # O agente aprende que precisa de reserva mesmo sem queda real
    if contexto.get('outage_simulado'):
        if soc_depois < RESERVA_MINIMA:
            reward -= 200.0  # punição ainda maior durante queda real
        else:
            reward += 5.0    # sobreviveu à queda — recompensa

    # ═══════════════════════════════════════════════════════════
    # PRIORIDADE 2: Carregar durante período solar
    # ═══════════════════════════════════════════════════════════
    # Se está gerando solar E a bateria não está cheia, o ideal
    # é STANDBY (deixar carregar). Descarregar durante solar é ruim.
    if gen_atual >= 300:  # período solar
        if acao == DISCHARGE:
            reward -= 15.0   # "por que você está descarregando com sol?"
        elif soc_depois > soc_antes:
            reward += 2.0    # "bom, está carregando"

    # ═══════════════════════════════════════════════════════════
    # PRIORIDADE 3: Economizar energia da rede (objetivo econômico)
    # ═══════════════════════════════════════════════════════════
    # Cada Wh que a bateria cobre em vez da rede = recompensa positiva
    if acao == DISCHARGE and deficit > 0 and gen_atual < 300:
        energia_salva_wh = min(
            deficit * SLOT_H,            # déficit do slot
            (soc_antes - RESERVA) * CAP,  # energia disponível
            MAX_DISCHARGE_WH              # limite físico
        )
        # Recompensa proporcional à energia economizada
        reward += energia_salva_wh * 0.01  # ~3.24 max por slot

    # ═══════════════════════════════════════════════════════════
    # PRIORIDADE 4: Penalizar STANDBY quando era seguro descarregar
    # ═══════════════════════════════════════════════════════════
    # Se a bateria está cheia, é noite, e o agente escolheu STANDBY
    # quando poderia ter descarregado com segurança → oportunidade perdida
    if acao == STANDBY and deficit > 0 and gen_atual < 300:
        if soc_antes > 80 and not contexto.get('incerteza_alta'):
            reward -= 0.5  # penalidade leve por ser muito conservador

    # ═══════════════════════════════════════════════════════════
    # PRIORIDADE 5: Bônus por SOC alto ao final do dia
    # ═══════════════════════════════════════════════════════════
    if contexto.get('fim_periodo_solar'):
        if soc_depois >= 95:
            reward += 10.0  # "bateria totalmente carregada ao pôr do sol, ótimo"
        elif soc_depois >= 80:
            reward += 5.0
        elif soc_depois < 60:
            reward -= 5.0   # "bateria não carregou bem — dia ruim"

    # ═══════════════════════════════════════════════════════════
    # PRIORIDADE 6: Bônus por adaptação a dias ruins
    # ═══════════════════════════════════════════════════════════
    # Se ontem a bateria não carregou 100%, e hoje o agente está
    # sendo conservador (STANDBY à noite) → bom comportamento
    if contexto.get('soc_fim_solar_ontem', 100) < 80:
        if acao == STANDBY and gen_atual < 300:
            reward += 1.0  # "dia anterior foi ruim, bom conservar"

    return reward
```

### 3.2 Por que essas escalas?

As magnitudes das recompensas definem implicitamente as prioridades:

| Prioridade | Recompensa/Punição | Magnitude |
|---|---|---|
| 1. Reserva para emergência | -100 a -200 | **Catastrófica** — agente aprende a evitar a todo custo |
| 2. Carregar durante solar | -15 (descarregar com sol) | **Alta** — regra quase inviolável |
| 3. Economia de energia | +0 a +3.24 por slot | **Moderada** — incentivo constante mas não dominante |
| 4. Não ser medroso demais | -0.5 por oportunidade perdida | **Leve** — empurrão suave para descarregar |
| 5. SOC alto ao pôr do sol | +5 a +10 | **Bônus pontual** — ensina o valor de um dia bem carregado |
| 6. Conservar após dia ruim | +1 | **Sutil** — favorece cautela após geração fraca |

O agente **não recebe essas regras como `if/else`**. Ele recebe números e, ao longo de milhares de episódios, descobre sozinho que:
- "Descarregar com sol = punição grande → melhor não fazer"
- "Ficar sem reserva = punição catastrófica → nunca deixar SOC cair demais"
- "Descarregar à noite quando tenho bastante energia = recompensa → é bom fazer quando seguro"

**Isso é o que torna o RL fundamentalmente diferente do sistema de regras.** As regras são _emergentes_ do treinamento, não _programadas_.

---

## 4. Como Funciona o Treinamento

### 4.1 O Ambiente (Gym)

O simulador que já existe no projeto É o ambiente de treinamento. Basta convertê-lo para Python:

```python
class BatteryEnv:
    """Ambiente OpenAI Gym para treinamento do agente RL."""

    def __init__(self, dados_historicos, ewma_profile):
        self.dados = dados_historicos       # DataFrame com load_w, generation_w por dia
        self.ewma = ewma_profile            # perfil EWMA de 96 slots
        self.capacidade = 5000              # Wh
        self.max_discharge = 1485           # Wh/slot (27A × 220V × 0.25h)
        self.max_charge = 742.5              # Wh/slot (13.5A × 220V × 0.25h)

    def reset(self):
        """Inicia um novo episódio (1 a 3 dias aleatórios)."""
        # Escolhe 1-3 dias consecutivos aleatórios do dataset
        # Sorteia SOC inicial entre 50-100%
        # Define se haverá eventos de queda de energia (probabilidade 20%)
        return estado_inicial

    def step(self, action):
        """Executa uma ação e retorna (novo_estado, recompensa, done, info)."""
        # 1. Aplica a física da bateria (idêntica ao aplicarBateria do JS)
        # 2. Calcula a recompensa
        # 3. Avança o slot
        # 4. Retorna o novo estado
        return novo_estado, recompensa, done, info
```

### 4.2 Loop de Treinamento

```
Para cada episódio (1 a 5000):
    estado = ambiente.reset()   # sorteia dias, SOC, condições

    Para cada slot (0 a 96×n_dias):
        # Escolhe ação (com exploração ε-greedy)
        Se random() < epsilon:
            ação = aleatória (STANDBY ou DISCHARGE)
        Senão:
            ação = argmax(Q_network(estado))

        # Executa e observa resultado
        novo_estado, recompensa, done = ambiente.step(ação)

        # Armazena experiência no replay buffer
        buffer.add(estado, ação, recompensa, novo_estado, done)

        # Treina a rede com batch aleatório do buffer
        Se len(buffer) >= BATCH_SIZE:
            batch = buffer.sample(BATCH_SIZE)
            loss = treinar_dqn(batch)

        estado = novo_estado

    # Decai epsilon (menos exploração ao longo do tempo)
    epsilon = max(0.01, epsilon * 0.995)
```

### 4.3 Parâmetros de Treinamento

| Parâmetro | Valor | Justificativa |
|---|---|---|
| Episódios | 3.000 – 5.000 | Suficiente para convergência com estado pequeno |
| Batch size | 64 | Padrão para DQN |
| Learning rate | 0.001 → 0.0001 | Começa rápido, desacelera |
| Gamma (desconto) | 0.99 | Valoriza recompensas futuras (importante!) |
| Epsilon inicial | 1.0 | 100% exploração no início |
| Epsilon final | 0.01 | 1% exploração após convergência |
| Replay buffer | 50.000 experiências | Memória de experiências passadas |
| Target network update | a cada 100 episódios | Estabiliza o treinamento |
| Duração do episódio | 1-3 dias (96-288 slots) | Variável para generalização |

### 4.4 Técnicas Essenciais

1. **Experience Replay:** Armazena transições (s, a, r, s') em um buffer e treina com amostras aleatórias. Quebra correlações temporais e melhora a estabilidade.

2. **Target Network:** Uma cópia "congelada" da rede Q usada para calcular os targets. Atualizada periodicamente. Evita instabilidade do "cachorro perseguindo o próprio rabo".

3. **ε-Greedy Decay:** Começa explorando aleatoriamente (ε=1.0) e gradualmente muda para exploração pura da política aprendida (ε→0.01). Garante que o agente descubra estratégias antes de exploitá-las.

4. **Queda de energia simulada:** Durante o treinamento, com ~20% de probabilidade, um período aleatório do dia tem `grid=0` forçado. Isso ensina o agente sobre a importância da reserva sem precisar codificar regras.

---

## 5. O Que o Agente Aprende (Resultados Esperados)

### 5.1 Comportamentos Emergentes Esperados

Após treinamento bem-sucedido, o agente deve exibir estes comportamentos **sem que nenhum deles tenha sido programado como regra**:

| # | Comportamento | Como emerge |
|---|---|---|
| 1 | STANDBY durante geração solar | Punição de -15 por descarregar com sol — aprende rápido a evitar |
| 2 | Não descarregar abaixo da reserva | Punição de -100/-200 — aprende nos primeiros ~500 episódios |
| 3 | Descarregar à noite quando tem energia suficiente | Recompensa de +0 a +3.24/slot por economizar rede |
| 4 | Ser conservador após dia de baixa geração | Bônus de +1 por STANDBY quando ontem foi ruim + punição se ficar sem reserva |
| 5 | Descarregar mais cedo na madrugada do que no início da noite | Gamma=0.99 faz o agente "pensar" em recompensa futura — se descarregar demais cedo, não terá para depois |
| 6 | Diferença de comportamento entre dia útil e fim de semana | Features de dia da semana no estado — se consumo é diferente, Q-values refletem |
| 7 | Adaptação sazonal | Feature de tendência de geração — se geração caiu nos últimos 7 dias, rede aprende a ser mais conservadora |

### 5.2 O Cenário que Você Descreveu

> "Se num período de 3 dias, o primeiro dia a geração não foi suficiente para carregar a bateria toda, ele tem que se preservar até o dia seguinte."

**Como o agente aprende isso:**

1. Durante o treinamento, o agente encontra muitos episódios onde o dia 1 teve geração ruim
2. Se ele descarrega à noite do dia 1, chega no dia 2 com SOC baixo
3. Se o dia 2 também é ruim, ele fica sem reserva → **punição -100**
4. Após centenas de episódios assim, a rede neural aprende: "quando `soc_fim_solar_ontem` é baixo e `tendencia_geracao_7d` é negativa → Q-value de STANDBY é maior que DISCHARGE"
5. O agente **emerge** com o comportamento conservador, sem que isso tenha sido uma regra

**Isso é fundamentalmente diferente do sistema atual**, onde teríamos que programar: `if soc_fim_solar < 80%: return STANDBY`. No RL, o agente **descobre o limiar sozinho** — e pode ser um limiar mais nuançado (ex: "se SOC é 75% E é terça-feira E a tendência de geração é negativa → STANDBY, mas se é sexta e a tendência é estável → vale arriscar um pouco").

### 5.3 Métricas de Avaliação

Para a apresentação do TCC, comparar:

| Métrica | Sistema de Regras | Agente RL |
|---|---|---|
| Energia da rede evitada (Wh/dia) | Medir | Medir |
| Vezes que SOC caiu abaixo da reserva | Contar | Contar |
| SOC médio ao pôr do sol | Medir | Medir |
| Comportamento após dia ruim | Fixo | Adaptativo |
| Comportamento dia útil vs fim de semana | Igual | Diferenciado |
| Economia projetada (R$/mês) | Calcular | Calcular |

**Resultado esperado:** O agente RL deve economizar **10-30% mais energia** que o sistema de regras, enquanto mantém segurança igual ou superior (nunca ficando sem reserva). A diferença vem principalmente de:
- Planejamento multi-slot (não é greedy)
- Adaptação a padrões de dia da semana
- Sensibilidade a tendências de geração

---

## 6. Aprendizado Contínuo

### 6.1 Como Funciona

Após o treinamento inicial, a cada dia real que passa:

```
1. Novos dados do inversor chegam (24h de leituras reais)
2. O novo dia é adicionado ao replay buffer
3. A rede é retreinada com 100-200 batches (fine-tuning, não do zero)
4. O perfil EWMA é atualizado com o novo dia
5. A tendência de geração é recalculada
```

Isso significa que:
- A rede **não esquece** o que aprendeu, mas **refina** com dados novos
- Se o consumo da casa mudar (ex: comprou um ar condicionado), o agente se adapta em ~1 semana
- Se o inverno chegar e a geração cair, a tendência reflete isso e a rede ajusta

### 6.2 Proteção contra Degradação

Para evitar que o agente "desaprenda" coisas boas:
- Manter 70% do replay buffer com dados do treinamento original
- Adicionar 30% com dados novos
- Monitorar métricas: se a recompensa média cair, reverter para o modelo anterior

---

## 7. Implementação Prática — O Que Precisa Ser Feito

### 7.1 Arquivos a Criar

```
backend/
├── rl/
│   ├── environment.py      # Ambiente Gym (simula bateria + dias reais)
│   ├── dqn_agent.py        # Agente DQN (rede neural + replay buffer)
│   ├── reward.py            # Função de recompensa (prioridades)
│   ├── train.py             # Script de treinamento
│   ├── evaluate.py          # Comparação RL vs regras
│   └── model/
│       └── battery_dqn.pth  # Modelo treinado (salvo com torch.save)
```

### 7.2 Dependências Adicionais

```
pip install torch numpy  # PyTorch para a rede neural (CPU é suficiente)
```

**Não precisa de GPU.** Com 12.482 parâmetros, o treinamento de 5.000 episódios leva ~2-5 minutos em CPU.

### 7.3 Integração com o Sistema Atual

Duas opções:

**Opção A: Substituir `decide()` pelo modelo treinado**
```python
# decision_engine.py
def decide_rl(state_vector):
    q_values = model(torch.tensor(state_vector))
    action = q_values.argmax().item()  # 0=STANDBY, 1=DISCHARGE
    return "STANDBY" if action == 0 else "FORCE_DISCHARGE"
```

**Opção B (recomendada para TCC): Manter ambos e comparar**
```python
# app.py
@app.route("/api/decide", methods=["POST"])
def api_decide():
    # Decisão por regras (modelo atual)
    resultado_regras = decide(...)

    # Decisão por RL
    state = build_state_vector(...)
    resultado_rl = decide_rl(state)

    return jsonify({
        "decision": resultado_rl["decision"],
        "decision_regras": resultado_regras["decision"],  # para comparação
        "reasoning": resultado_rl["reasoning"],
    })
```

---

## 8. Cronograma de Implementação

| Etapa | Tempo estimado | Descrição |
|---|---|---|
| 1. Ambiente Python | 1-2 dias | Criar `environment.py` replicando a física da bateria |
| 2. Agente DQN | 1 dia | Implementar rede neural + replay buffer |
| 3. Função de recompensa | 1 dia | Calibrar pesos das prioridades |
| 4. Treinamento | 1 dia | Treinar, ajustar hiperparâmetros, validar |
| 5. Avaliação | 1 dia | Comparar RL vs regras em cenários diversos |
| 6. Integração | 1 dia | Conectar modelo treinado ao Flask |
| 7. Frontend (toggle) | 0.5 dia | Botão para alternar entre "IA Regras" e "IA RL" |

**Total: ~7 dias de trabalho**

---

## 9. O Que Mostrar na Apresentação do TCC

### Slide 1: O Problema
"Como decidir quando usar a bateria sem saber o futuro?"

### Slide 2: Abordagem 1 — Sistema de Regras
- EWMA para previsão
- 4 regras em cascata
- Funciona, mas é rígido

### Slide 3: Limitações das Regras
- Não captura tendências
- Decisão greedy (slot a slot)
- Sem noção de incerteza
- Conservadorismo fixo

### Slide 4: Abordagem 2 — Reinforcement Learning
- DQN com ~12.500 parâmetros
- Aprende as prioridades por recompensa/punição
- Não recebe regras — descobre estratégias sozinho

### Slide 5: O Que a IA Aprendeu
Mostrar lado a lado:
- Cenário com dia ruim → Regras: comportamento fixo, RL: adaptou
- Cenário multi-dia → Regras: greedy, RL: planejou
- Dia útil vs fim de semana → Regras: igual, RL: diferenciou

### Slide 6: Resultados Quantitativos
Tabela comparativa: energia economizada, segurança, adaptabilidade

### Slide 7: Aprendizado Contínuo
"A cada dia novo, a IA fica mais inteligente"

---

## 10. Resposta às Fragilidades Identificadas

| Fragilidade | Como o RL resolve |
|---|---|
| 1. EWMA não captura tendências | Feature `tendencia_geracao_7d` no estado — a rede aprende a reagir |
| 2. Decisão greedy | Gamma=0.99 faz o agente otimizar recompensa ACUMULADA, não instantânea |
| 3. Sem modelagem de incerteza | Treinamento com dias variados ensina o agente a ser cauteloso sob variância |
| 4. Não diferencia dias da semana | Features de dia da semana no estado — se padrão difere, rede captura |
| 7. Conservadorismo fixo | O agente aprende o "nível certo" de conservadorismo dinamicamente |

**Nota:** A fragilidade #4 do documento original (previsão meteorológica) não é resolvida pelo RL — continua dependendo dos dados disponíveis. Se uma API de meteorologia fosse adicionada como feature do estado, o agente a incorporaria automaticamente.
