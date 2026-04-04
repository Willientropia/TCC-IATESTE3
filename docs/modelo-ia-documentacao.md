# Documentação Técnica do Modelo de IA — Sistema de Apoio à Decisão para Bateria Solar

**Autor:** Gerado por Claude (Anthropic) como apoio ao TCC
**Data:** Março 2026
**Versão do sistema:** 3.0 (terceira iteração da lógica de decisão)

---

## 1. Tipo de IA Utilizada

O sistema **não** utiliza redes neurais, aprendizado profundo ou qualquer modelo de machine learning "caixa-preta". A abordagem é composta por dois componentes estatísticos/algorítmicos distintos:

### 1.1 Modelo de Previsão: EWMA por Faixa Horária

**EWMA** = *Exponentially Weighted Moving Average* (Média Móvel Exponencialmente Ponderada).

É um método estatístico clássico, amplamente utilizado em controle de qualidade (cartas de controle de Shewhart/EWMA), finanças (suavização de séries temporais) e engenharia de processos. **Não é um modelo de machine learning** — é uma média ponderada determinística, sem fase de treinamento, sem otimização de hiperparâmetros por gradiente e sem função de perda.

O perfil EWMA é construído assim:

```
Para cada um dos 96 slots de 15 minutos do dia (00:00, 00:15, ..., 23:45):
    Para cada dia histórico d no dataset:
        peso(d) = exp(-λ × dias_atrás)

    valor_slot = Σ(peso(d) × valor_real(d, slot)) / Σ(peso(d))
```

Onde:
- **λ (lambda)** é a taxa de decaimento (padrão: 0.1)
- **dias_atrás** = (data_referência - data_do_dado).days
- O resultado são dois vetores de 96 posições: `load_profile` (carga prevista) e `generation_profile` (geração solar prevista)

### 1.2 Modelo de Decisão: MPC Simplificado com Saída Binária

**MPC** = *Model Predictive Control* (Controle Preditivo por Modelo).

É uma técnica de controle que usa um modelo do processo para prever o comportamento futuro e tomar decisões ótimas no presente. A versão implementada é um caso **extremamente simplificado** de MPC:

- **Horizonte:** cíclico de 24h (o perfil EWMA de 96 slots "enrola" ao redor da meia-noite)
- **Saída:** binária (`FORCE_DISCHARGE` ou `STANDBY`) — diferente de MPC clássico que otimiza um sinal contínuo
- **Sem otimização iterativa:** o MPC clássico resolve um problema de otimização a cada passo; aqui, a decisão é tomada por regras determinísticas baseadas na comparação de duas grandezas (energia utilizável vs. déficit previsto)

A classificação mais precisa seria: **sistema de regras baseado em previsão estatística** (*rule-based system with statistical forecasting*). Chama-se "MPC simplificado" no código por analogia acadêmica com a estrutura de prever-e-decidir.

---

## 2. Parâmetros do Sistema

### 2.1 Parâmetros da Bateria (constantes físicas)

| Parâmetro | Valor | Origem |
|-----------|-------|--------|
| Capacidade total | 5000 Wh (5 kWh) | Especificação do fabricante |
| Tensão nominal | 220V | Especificação do inversor |
| Corrente máx. descarga | 27A → 5940W | Limite do inversor |
| Corrente máx. carga | 13.5A → 2970W | Limite recomendado |
| Energia máx. descarga/slot | 1485 Wh (5940W × 0.25h) | Derivado |
| Energia máx. carga/slot | 742.5 Wh (2970W × 0.25h) | Derivado |
| Duração do slot | 15 min (0.25h) | Definição do sistema |

### 2.2 Parâmetros Configuráveis pelo Usuário

| Parâmetro | Faixa | Padrão | Efeito |
|-----------|-------|--------|--------|
| **λ (lambda)** | 0.01 – 1.0 | 0.10 | Controla o peso dos dias recentes vs. antigos na previsão |
| **SOC Inicial** | 0% – 100% | 99% | Estado de carga no início da simulação |
| **Reserva de Segurança** | 0% – 50% | 20% | Percentual mínimo preservado para emergência |
| **Velocidade (ms)** | 50 – 2000 | — | Velocidade de avanço da simulação (visual) |

### 2.3 Constantes da Lógica de Decisão

| Constante | Valor | Justificativa |
|-----------|-------|---------------|
| `SOLAR_RETORNO_MIN_W` | 300W | Limiar mínimo para considerar que "a geração solar está ativa". Abaixo disso, a geração é residual e não justifica priorizar carregamento. |

### 2.4 Efeito do Lambda (λ)

O lambda controla a "memória" do sistema:

| λ | Peso de ontem | Peso de 7 dias atrás | Peso de 30 dias atrás | Comportamento |
|---|---|---|---|---|
| 0.01 | 0.990 | 0.932 | 0.741 | Memória longa — todos os dias pesam quase igual |
| **0.10** | **0.905** | **0.497** | **0.050** | **Padrão — equilíbrio entre recente e histórico** |
| 0.30 | 0.741 | 0.122 | 0.000 | Memória curta — praticamente só última semana |
| 1.00 | 0.368 | 0.001 | 0.000 | Memória ultracurta — quase só ontem |

---

## 3. Lógica de Decisão — Passo a Passo

A função `decide()` é chamada a cada slot de 15 minutos da simulação. Ela recebe o estado atual e o forecast EWMA, e retorna uma decisão binária.

### 3.1 Fluxo Completo

```
ENTRADA:
  - slot_atual (0-95)
  - soc_percent (estado real da bateria)
  - load_forecast[96] (previsão EWMA de carga, cíclica)
  - generation_forecast[96] (previsão EWMA de geração, cíclica)
  - safety_reserve_percent (reserva mínima)

PASSO 1: Calcular energia disponível
  energia_disponivel = soc_percent × capacidade / 100
  reserva = safety_reserve × capacidade / 100
  utilizavel = max(0, disponivel - reserva)

PASSO 2: Encontrar retorno solar
  Percorre generation_forecast[0..95]:
    Se generation_forecast[i] >= 300W:
      solar_retorna = True
      horizonte = i slots
      PARAR
  Se não encontrou: horizonte = 96 slots (24h)

PASSO 3: Calcular déficit acumulado até o horizonte
  Para i de 0 até horizonte:
    demanda_liquida = load_forecast[i] - generation_forecast[i]
    Se demanda_liquida > 0:
      deficit += demanda_liquida × 0.25h

PASSO 4: Decidir (4 regras em cascata)
  REGRA 1: Se generation_forecast[0] >= 300W → STANDBY
           (período solar, prioridade é carregar)

  REGRA 2: Se utilizavel <= 0 → STANDBY
           (bateria na reserva de segurança)

  REGRA 3: Se utilizavel >= deficit → FORCE_DISCHARGE
           (bateria cobre o déficit até o sol voltar)

  REGRA 4: Senão → STANDBY
           (déficit maior que energia disponível)
```

### 3.2 A Pergunta Central

A lógica inteira pode ser resumida em uma única pergunta:

> **"Se acabasse a luz agora, eu teria bateria suficiente para durar até que a energia solar assuma?"**
> - Se **sim** → FORCE_DISCHARGE (usar a bateria para evitar consumo da rede)
> - Se **não** → STANDBY (preservar a bateria para emergências)

### 3.3 Prioridades (em ordem)

1. **Período solar → STANDBY sempre.** Durante o dia, a prioridade absoluta é carregar a bateria com energia solar. Mesmo que haja energia sobrando, a bateria não descarrega durante a geração.

2. **Reserva de segurança → STANDBY sempre.** Se a bateria atingiu o nível mínimo de reserva, ela para de descarregar independentemente de qualquer outro fator.

3. **Energia suficiente → FORCE_DISCHARGE.** Se a bateria tem energia suficiente para cobrir todo o déficit previsto até o próximo período solar, a decisão é descarregar.

4. **Caso contrário → STANDBY.** Na dúvida, conservar.

### 3.4 Forecast Cíclico

O forecast é **cíclico**: ele "enrola" de volta ao início do perfil EWMA quando ultrapassa o slot 95. Isso é fundamental para que o sistema funcione à noite:

```
Exemplo: slot atual = 88 (22:00)
  forecast[0] = ewma[88]  → 22:00 (carga noturna)
  forecast[1] = ewma[89]  → 22:15
  ...
  forecast[7] = ewma[95]  → 23:45
  forecast[8] = ewma[0]   → 00:00 (cruzou meia-noite!)
  forecast[9] = ewma[1]   → 00:15
  ...
  forecast[34] = ewma[26] → 06:30 (sol começa a voltar)
```

Sem o forecast cíclico, o sistema "acharia" que após 23:45 não existe mais nada, e tomaria decisões incorretas (foi um bug da versão 1).

### 3.5 Separação Previsão vs. Realidade

Um princípio fundamental do sistema:

- **A IA prevê** usando o perfil EWMA armazenado (perfil histórico médio ponderado)
- **A simulação executa** usando os dados reais do gráfico (que o usuário pode modificar arrastando pontos)

Isso significa que a IA **nunca vê o futuro**. Ela toma decisões baseada no que "acha" que vai acontecer (EWMA), mas o resultado real pode ser diferente. Essa separação é essencial para que a simulação seja honesta e represente o que aconteceria em produção.

---

## 4. Física da Bateria — Simulação

A cada slot, após a decisão da IA, a função `aplicarBateria()` calcula o efeito real:

### 4.1 Cenário 1: Superávit Solar (geração > carga)

```
excedente = geração - carga
energia_para_bateria = min(excedente × 0.25h, espaço_disponível, 742.5Wh)
bateria += energia_para_bateria
rede = (carga - geração + taxa_bateria_W)  → geralmente negativa (exportando)
```

### 4.2 Cenário 2: Déficit + FORCE_DISCHARGE

```
deficit = carga - geração
energia_da_bateria = min(deficit × 0.25h, energia_acima_da_reserva, 1485Wh)
bateria -= energia_da_bateria
rede = (carga - geração - taxa_descarga_W)  → reduzida ou zerada
```

### 4.3 Cenário 3: Déficit + STANDBY

```
bateria não age (delta = 0)
rede = carga - geração  → importação total da rede
```

### 4.4 Equação de Balanço Energético

Em todos os cenários:
```
rede = carga - geração + taxa_bateria
```
Onde `taxa_bateria` é:
- Positiva quando carrega (a bateria "consome" da rede/solar)
- Negativa quando descarrega (a bateria "fornece" para a carga)

---

## 5. Dados de Entrada

### 5.1 Origem dos Dados

- **Fonte:** Inversor solar real, dados coletados via protocolo Modbus e armazenados no Firebase
- **Formato:** JSON aninhado `{ "YYYY-MM-DD": { "HH-MM-SS": { campos } } }`
- **Período:** ~44 dias (02/2026 – 03/2026), com ~35 dias válidos após filtragem
- **Resolução original:** leituras a cada ~30 segundos
- **Resolução de trabalho:** agregado em slots de 15 minutos (média)

### 5.2 Campos Utilizados

| Campo no JSON | Unidade original | Conversão | Variável no sistema |
|---|---|---|---|
| `Active power` | kW | × 1000 | `generation_w` (W) |
| `Load Combined Power` | W | — | `load_w` (W) |
| `CT1 R Phase Active Power` | W | — | `grid_w` (W) |
| `Energy Storage Module 1` | W | — | `battery_w` (W) |
| `soc` | % | — | `soc` (%) |

### 5.3 Filtragem de Qualidade

- Dias com menos de 48 slots preenchidos (menos de ~50% do dia) são descartados do perfil EWMA
- Valores ausentes são preenchidos com interpolação linear dentro do dia
- Outliers não são tratados (o EWMA naturalmente suaviza variações)

---

## 6. Limites e Restrições Impostas

### 6.1 Limites Físicos (hardware)

- A bateria **nunca** descarrega acima de 27A (5940W) → 1485 Wh por slot
- A bateria **nunca** carrega acima de 13.5A (2970W) → 742.5 Wh por slot
- A bateria **nunca** ultrapassa 100% (5000 Wh) nem fica abaixo de 0%

### 6.2 Limites Lógicos (software)

- A bateria **nunca** descarrega abaixo da reserva de segurança (padrão 20%)
- A bateria **nunca** descarrega durante o período solar (geração prevista ≥ 300W)
- O sistema **nunca** usa dados reais futuros para tomar decisões — apenas o perfil EWMA

### 6.3 Limitação de Horizonte

- O horizonte de previsão é fixo em 24h (96 slots cíclicos)
- Isso significa que o sistema não consegue "planejar" além de um dia
- Se o sol não retornar dentro de 24h (evento extremamente raro), o sistema assume horizonte = 24h de déficit

---

## 7. Como o Modelo Foi "Treinado"

**O modelo não foi treinado no sentido de machine learning.** Não há:
- Fase de treinamento vs. teste
- Otimização de parâmetros por gradiente descendente
- Função de perda minimizada iterativamente
- Validação cruzada

O que existe é:
1. **Carga de dados históricos:** O JSON é lido e transformado em DataFrame
2. **Cálculo do perfil EWMA:** Média ponderada determinística dos dados históricos, agrupada por slot de 15 minutos
3. **Regras de decisão codificadas manualmente:** As 4 regras da cascata foram escritas com base no raciocínio lógico do problema (não aprendidas dos dados)

O lambda (λ) foi definido como 0.10 por escolha de projeto (não por tuning). O limiar solar de 300W foi escolhido empiricamente como um valor que separa "geração residual" de "geração útil".

---

## 8. Análise Crítica do Modelo

### 8.1 Pontos Fortes

1. **Totalmente interpretável.** Cada decisão pode ser rastreada numericamente: "a bateria tem 3950Wh utilizáveis, o déficit previsto é 2932Wh, portanto sobram 1018Wh, decisão: descarregar." Não há "caixa-preta".

2. **Computacionalmente trivial.** O cálculo é feito em microssegundos — somas e comparações. Pode rodar em microcontroladores.

3. **Robusto a dados ruidosos.** O EWMA suaviza naturalmente variações diárias. Dias atípicos (chuva, manutenção) perdem influência conforme envelhecem.

4. **Conservador por padrão.** Na dúvida, o sistema preserva a bateria. Isso é correto para o caso de uso (proteção contra queda de energia).

5. **Sem risco de overfitting.** Como não há treinamento, não há risco de memorizar padrões específicos dos dados de treino.

### 8.2 Fragilidades e Questionamentos

1. **O EWMA não captura tendências.** Se a geração solar está caindo progressivamente ao longo das semanas (inverno se aproximando), o EWMA ainda carrega informação dos dias de alta geração. O sistema pode ser otimista demais sobre a geração solar futura. Um modelo que detecte tendências (regressão linear no tempo, ou Holt-Winters com componente de tendência) seria mais preciso.

2. **A decisão é binária e greedy.** A cada slot, o sistema decide DESCARGA ou STANDBY independentemente do que decidiu antes. Não há planejamento multi-slot — por exemplo, "descarregar um pouco agora e guardar o resto para mais tarde" não é possível. Um MPC real otimizaria a trajetória inteira.

3. **Não modela incerteza.** O EWMA gera um único número por slot (a média ponderada). Não há noção de variância ou intervalo de confiança. Em um dia nublado, a geração real pode ser 30% do previsto, e o sistema não tem como saber. Um modelo probabilístico (como um perfil com bandas de confiança) permitiria decisões mais conservadoras automaticamente.

4. **Não incorpora previsão meteorológica.** O sistema é puramente baseado em dados históricos. Se uma API de previsão do tempo fosse integrada, o modelo poderia antecipar dias nublados/chuvosos e ajustar o comportamento.

5. **O limiar solar de 300W é fixo.** Um valor que funciona bem no verão pode ser inadequado no inverno, quando a geração de pico pode ser menor. Idealmente, esse limiar seria relativo ao pico médio de geração.

6. **Não modela degradação da bateria.** O sistema assume capacidade constante de 5000Wh. Na prática, baterias perdem capacidade com o tempo e com ciclos de carga/descarga.

7. **Não diferencia dias da semana.** O consumo residencial pode variar entre dias úteis e fins de semana (pessoas em casa = mais consumo). O EWMA trata todos os dias igualmente.

8. **Conservadorismo fixo.** A reserva de segurança é uma porcentagem fixa. Se a frequência de quedas de energia for conhecida (dados da concessionária), a reserva poderia ser ajustada dinamicamente.

---

## 9. Alternativas de IA que Poderiam Ser Usadas

### 9.1 Holt-Winters (Suavização Exponencial Tripla)

**O que é:** Extensão do EWMA que captura nível, tendência e sazonalidade.

**Vantagem:** Detectaria que a geração solar está diminuindo ao longo das semanas (tendência) e que o padrão se repete a cada 24h (sazonalidade). Seria mais preciso em períodos de transição sazonal.

**Complexidade:** Baixa. Não é ML, é estatística clássica. Disponível no `statsmodels` (Python).

**Adequação para este projeto:** Alta. Seria a melhoria mais natural e de menor custo sobre o EWMA.

### 9.2 ARIMA / SARIMA

**O que é:** Modelos auto-regressivos integrados de médias móveis, com componente sazonal.

**Vantagem:** Captura correlações temporais complexas. SARIMA com sazonalidade de 96 (um dia) poderia prever o próximo dia com alta precisão.

**Desvantagem:** Requer seleção de parâmetros (p, d, q, P, D, Q, s), é sensível a estacionariedade, e é mais difícil de interpretar.

**Adequação:** Média. Mais potente que EWMA, mas o ganho pode não justificar a complexidade para 44 dias de dados.

### 9.3 Redes Neurais Recorrentes (LSTM / GRU)

**O que é:** Redes neurais projetadas para séries temporais, com "memória" de longo prazo.

**Vantagem:** Podem aprender padrões arbitrariamente complexos nos dados. Com dados suficientes, uma LSTM poderia prever com precisão a geração solar de amanhã baseado nos últimos 7 dias.

**Desvantagem:** Requer significativamente mais dados (~anos, não semanas), treinamento com GPU, e perde interpretabilidade. Com 44 dias, há alto risco de overfitting.

**Adequação:** Baixa para este dataset. Em um sistema de produção com anos de dados, seria excelente.

### 9.4 Reinforcement Learning (Aprendizado por Reforço)

**O que é:** Um agente aprende a política ótima de carga/descarga maximizando uma recompensa acumulada (ex: minimizar custo de rede + manter reserva para emergência).

**Vantagem:** Otimiza diretamente o objetivo real (economia + segurança), não precisa de regras manuais, e naturalmente resolve o trade-off entre usar a bateria agora vs. guardar para depois. Resolveria a fragilidade #2 (decisão greedy) automaticamente.

**Desvantagem:** Requer ambiente de simulação (que já existe!), bastante tempo de treinamento, e é uma "caixa-preta" difícil de auditar. Se o agente RL tomar uma decisão ruim, é difícil explicar por quê.

**Adequação:** Média-alta. O simulador já existe, o que facilita muito. Mas a explicabilidade seria comprometida.

### 9.5 Programação Linear / Otimização Convexa (MPC Real)

**O que é:** Resolver matematicamente o problema de otimização a cada passo: "dada a previsão de geração e carga para as próximas 24h, qual a sequência ótima de carga/descarga que minimiza o consumo de rede respeitando as restrições da bateria?"

**Vantagem:** Garante otimalidade matemática (não apenas "bom o suficiente"). Resolveria a fragilidade #2 completamente.

**Desvantagem:** Requer formulação cuidadosa do problema (função objetivo, restrições), solver instalado (ex: `cvxpy`, `PuLP`), e a previsão ainda precisa vir de algum modelo (EWMA, etc.).

**Adequação:** Alta. É a abordagem mais rigorosa e academicamente sólida para MPC. Porém, a formulação do problema é mais complexa.

### 9.6 Árvores de Decisão / Random Forest

**O que é:** Modelos de classificação que poderiam aprender a decisão (DISCHARGE vs STANDBY) a partir de features como hora, SOC, geração recente, temperatura, etc.

**Vantagem:** Interpretável (pode-se visualizar a árvore), rápido, e funcionaria como uma versão aprendida das regras manuais atuais.

**Desvantagem:** Precisa de dados rotulados (qual era a decisão "correta" em cada situação?), o que não existe. Seria necessário criar um rótulo de "boa decisão" vs "má decisão" retroativamente.

**Adequação:** Baixa. Sem rótulos de qualidade, é difícil treinar. Mas poderia ser usado como alternativa ao EWMA para a previsão (Random Forest para regressão de geração solar).

### 9.7 Modelos Híbridos (Forecast + Otimização)

**O que é:** Combinar um modelo de previsão probabilístico (EWMA com bandas de confiança, ou um modelo bayesiano) com um otimizador que minimize custo esperado sob incerteza (*stochastic MPC*).

**Vantagem:** O melhor dos dois mundos — previsão honesta sobre a incerteza + decisão ótima sob essa incerteza.

**Desvantagem:** Muito mais complexo. Geralmente usado em sistemas industriais de grande porte.

**Adequação:** Baixa para TCC, alta para sistemas reais de gestão de baterias.

---

## 10. Conclusão e Avaliação Pessoal

### O que o modelo faz bem

O modelo cumpre seu propósito como **sistema demonstrativo para TCC**: é simples o suficiente para ser compreendido e explicado integralmente, mas sofisticado o suficiente para produzir decisões que fazem sentido em simulação. A separação entre previsão (EWMA) e decisão (regras baseadas em déficit) é limpa e didática. A simulação interativa com gráficos arrastáveis permite visualizar o impacto das decisões de forma tangível.

A escolha do EWMA é acertada para um dataset de 44 dias: modelos mais complexos não teriam dados suficientes para generalizar, e o EWMA captura o padrão diário típico com pesos que fazem sentido intuitivo.

### O que melhoraria

Se o projeto evoluísse além do TCC, as melhorias de maior impacto seriam:

1. **Substituir EWMA por Holt-Winters** — ganho imediato na previsão, quase zero de complexidade adicional.

2. **Incorporar previsão meteorológica** — APIs como OpenWeatherMap fornecem previsão de irradiância solar. Isso eliminaria o maior ponto cego do sistema: não saber que amanhã será chuvoso.

3. **Substituir as regras manuais por otimização linear** — transformar a decisão em um problema `min Σ(grid_import)` sujeito às restrições da bateria. A previsão continuaria vindo do EWMA, mas a decisão seria matematicamente ótima.

4. **Adicionar bandas de confiança ao EWMA** — calcular não apenas a média ponderada, mas também o desvio ponderado por slot. Usar o percentil 25 (cenário pessimista) para decisões de descarga tornaria o sistema mais conservador em horários de alta variabilidade.

### Honestidade sobre o que o modelo **não** é

Este modelo **não é IA no sentido popular do termo**. Não há redes neurais, não há "aprendizado" no sentido de que o sistema muda seu comportamento com o tempo (a menos que novos dados sejam adicionados e o EWMA seja recalculado). É um sistema de regras alimentado por estatística descritiva. Isso não é uma crítica — é uma constatação. Para o problema em questão, com o volume de dados disponível, esta abordagem é provavelmente **mais confiável** do que uma rede neural que correria alto risco de overfitting.

O termo mais preciso seria: **sistema de apoio à decisão baseado em previsão estatística ponderada** (*weighted statistical forecasting-based decision support system*).

---

## Apêndice A: Diagrama do Fluxo de Dados

```
  ┌─────────────────┐
  │  Dados Históricos│
  │  (Firebase JSON) │
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │  data_loader.py  │  Lê JSON, agrega em slots de 15 min
  └────────┬────────┘
           │
           ▼
  ┌─────────────────────┐
  │  profile_builder.py  │  Calcula perfil EWMA (96 slots × 2 séries)
  │  peso = e^(-λ×dias)  │
  └────────┬────────────┘
           │
           ▼
  ┌─────────────────────────────┐
  │  Frontend (simulation.js)    │
  │  Armazena ewmaLoad96[],      │
  │  ewmaGen96[] separados       │
  │  dos dados do gráfico        │
  └──────────┬──────────────────┘
             │
    A cada slot da simulação:
             │
             ▼
  ┌───────────────────────────────┐
  │  buildCyclicForecast(slot)     │  Gera forecast cíclico de 24h
  │  idx = (slotInDay + i) % 96   │  usando perfil EWMA original
  └──────────┬────────────────────┘
             │
             ▼
  ┌──────────────────────────────────┐
  │  decision_engine.py → decide()    │
  │  Compara energia_utilizavel       │
  │  vs. deficit_previsto             │
  │  → FORCE_DISCHARGE ou STANDBY    │
  └──────────┬───────────────────────┘
             │
             ▼
  ┌──────────────────────────────────┐
  │  aplicarBateria()                 │
  │  Usa dados REAIS do gráfico       │
  │  (não o forecast)                 │
  │  → novo SOC, novo grid_real       │
  └──────────┬───────────────────────┘
             │
             ▼
  ┌─────────────────┐
  │  Gráfico Chart.js│  Atualiza bateria (Wh) e rede (W)
  │  + Relatório     │
  └─────────────────┘
```

## Apêndice B: Exemplo Numérico Completo

**Cenário:** 03:00 da manhã, SOC = 99%, λ = 0.10, reserva = 20%

```
PASSO 1: Energia
  disponível = 99% × 5000 = 4950 Wh
  reserva    = 20% × 5000 = 1000 Wh
  utilizável = 4950 - 1000 = 3950 Wh

PASSO 2: Retorno solar
  Perfil EWMA prevê geração ≥ 300W a partir do slot 26 (06:30)
  Slot atual = 12 (03:00), horizonte = 26 - 12 = 14 slots (3.5h)

PASSO 3: Déficit
  14 slots × ~800W carga × 0.25h = ~2800 Wh de déficit
  (geração = 0W em todos os 14 slots → demanda líquida = carga)

PASSO 4: Decisão
  Regra 1: geração_forecast[0] = 0W < 300W → NÃO se aplica
  Regra 2: utilizável = 3950 > 0 → NÃO se aplica
  Regra 3: 3950 ≥ 2800 → FORCE_DISCHARGE ✓

  Justificativa: "Energia utilizável (3950Wh) cobre o déficit previsto
  (2800Wh, 3.5h até solar). Seguro descarregar."
```
