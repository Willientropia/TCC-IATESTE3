# Avaliação de Quedas de Energia — Critérios e Comportamento Esperado do Agente

Este documento explica como avaliar o comportamento do agente DQN em cenários com queda de energia,
quais são os critérios de sucesso, e por que certos padrões de decisão são incorretos mesmo que
pareçam razoáveis à primeira vista.

---

## As duas regras fundamentais

O sistema tem **dois objetivos**, em ordem de prioridade:

1. **Nunca deixar a planta sem energia durante uma queda de luz**, desde que a bateria, estando
   carregada ao máximo no início da noite, seja fisicamente capaz de suprir essa queda.

2. **Descarregar a bateria de forma inteligente** ao longo da noite, economizando energia da rede
   e reduzindo a conta de luz — mas apenas quando isso não compromete a regra 1.

A regra 1 é absoluta. A regra 2 só existe dentro do espaço que a regra 1 permite.

---

## O que é uma queda "suprível"

Uma queda é suprível se:

```
energia_utilizável_no_início_da_noite ≥ energia_necessária_durante_a_queda
```

Onde:
- `energia_utilizável = (SOC_atual - reserva_segurança) × capacidade_bateria`
- `energia_necessária = carga_média_da_casa × duração_da_queda`

**Exemplo concreto (cenário 3d_D1_00h_queda):**
- Bateria cheia ao fim do período solar: SOC = 100% → 4500 Wh utilizáveis (5000 Wh - 10% reserva)
- Queda no D2 00:00, duração 5,25h, carga média ~800W → energia necessária ~4200 Wh
- Conclusão: **4500 Wh ≥ 4200 Wh → a queda É suprível**

Se o agente chega no início da queda com SOC abaixo do necessário, isso é uma **falha do agente**,
pois era fisicamente possível sobreviver à queda com decisões corretas.

---

## Por que descarregar cedo demais é errado

### O cenário 3d_D1_00h_queda como exemplo didático

Este cenário é ideal para avaliar o agente porque a queda ocorre no **D2 00:00**, e tudo que
acontece **depois do período solar do D1** é o que determina se o agente sobreviverá ou não.
O que aconteceu antes (madrugada do D1) não importa para a avaliação da queda do D2.

**O que o agente deveria fazer:**

| Período              | Comportamento correto                                    |
|----------------------|----------------------------------------------------------|
| D1 solar (07h–17h)   | STANDBY — deixar o sol recarregar a bateria até ~100%   |
| D1 tarde (17h–23h)   | STANDBY ou descarga leve e cautelosa                    |
| D1 madrugada (23h→)  | Descarga progressiva **próxima ao amanhecer**, não antes |
| D2 00:00 (queda)     | Bateria próxima de 100% → sobrevive à queda de 5h       |

**O que o agente fez de errado na simulação:**

Em D1 00:00–04:15 (madrugada, antes do sol), o `decision_engine.py` forçou descarga porque o
cálculo MPC avaliou que "energia utilizável > déficit até o solar". O SOC caiu de 75% para 21.8%.
O sol do D1 recarregou parcialmente a bateria, mas só até 72.3% (carga do dia consumiu parte
da geração). A bateria chegou ao D2 00:00 com apenas 72.3% (3115 Wh utilizáveis) — insuficiente
para a queda de ~4200 Wh.

**O erro não foi na madrugada do D1** (antes da geração, o agente pode descarregar para economizar
rede). **O erro foi que a bateria não chegou ao D2 suficientemente carregada.**

---

## Por que "descarregar um pouco + voltar ao standby" não resolve

Um padrão frequente e incorreto é o agente fazer:

```
DISCHARGE → STANDBY → DISCHARGE → STANDBY → DISCHARGE → STANDBY...
```

ao longo da noite, alternando entre descarregar e conservar. Isso é problemático por três razões:

### 1. Redução imprevisível do SOC
Cada pulso de descarga reduz o SOC. Se uma queda de energia ocorre no meio da noite, o SOC
disponível é menor do que o necessário — mesmo que cada decisão individual "parecesse segura"
no momento em que foi tomada.

### 2. O agente não sabe quando a queda vai ocorrer
O agente não tem acesso ao momento exato da próxima queda. Portanto, a única estratégia
segura é tratar **toda a noite como potencialmente a noite de uma queda** e só descarregar
quando o SOC restante ainda cobre o pior caso (queda de duração máxima até o amanhecer).

### 3. O benefício econômico da descarga fragmentada é mínimo
Descarregar 5% da bateria às 22h economiza ~250 Wh de rede. Isso é insignificante comparado
à penalidade de deixar a casa sem energia durante 2–5h de queda. O agente não deve sacrificar
resiliência por economia marginal.

**A estratégia correta de descarga noturna:**

```
Déficit restante até o amanhecer = soma(carga_prevista_de_agora_até_07h) - soma(geração_prevista)
Se SOC_utilizável > Déficit_restante + margem_de_segurança:
    → pode descarregar (o excesso nunca será usado)
Caso contrário:
    → STANDBY (preservar para a queda)
```

O agente deve concentrar a descarga **nas últimas horas antes do amanhecer**, quando o déficit
restante até o sol é pequeno o suficiente para que a bateria cubra tanto o déficit quanto uma
eventual queda.

---

## Por que o modelo precisa de aprendizado contínuo

### O problema com regras fixas (MPC estático)

O `decision_engine.py` usa uma regra determinística:
> "Se energia utilizável ≥ déficit previsto até o solar → FORCE_DISCHARGE"

Esta regra usa o perfil EWMA (média histórica dos últimos dias) para prever carga e geração.
Em dias típicos, funciona razoavelmente. Mas falha em:

- **Dias com carga anormalmente alta** (festas, máquinas ligadas, D4/D5 com >30 kWh de carga):
  o EWMA subestima o déficit → a regra força descarga quando não devia.
- **Sequências de dias nublados** (D4+D5 nublados seguidos): o EWMA ainda usa histórico
  ensolarado → superestima a geração futura → força descarga prematura.
- **Múltiplas quedas em noites consecutivas**: a regra não "sabe" que haverá outra queda
  amanhã — trata cada noite de forma independente.

### O que o aprendizado contínuo resolve

Um agente DQN treinado em dados reais aprende padrões que regras fixas não conseguem codificar:

| Padrão | Regra fixa | DQN treinado |
|--------|------------|--------------|
| Dia nublado seguido de noite com queda | Descarrega (EWMA otimista) | Conserva (viu padrão no treino) |
| Carga alta no jantar seguida de madrugada | Descarrega às 19h | Aguarda até 03h |
| Segunda queda em noites consecutivas | Ignora (stateless) | Preserva (viu episódios multi-queda) |
| Perfil de carga diferente no fim de semana | Usa EWMA geral | Detecta via weekday feature |

Além disso, um modelo com aprendizado contínuo pode ser **atualizado com dados reais da planta**
após o deploy, refinando seu comportamento para o perfil específico daquela residência — algo
impossível com um sistema de regras estático.

### Por que isso é necessário para generalização

O cenário desta planta (Curitiba, carga ~900W média, geração pico ~3kW) é específico. Uma
regra calibrada para esta planta falhará em:
- Uma planta com ar-condicionado (pico de carga 5×maior)
- Uma planta em região com mais nuvens (geração 60% menor)
- Uma planta com bateria de 10 kWh (capacidade 2×)

O DQN, ao receber as features corretas (`razao_cobertura`, `slots_to_solar`, perfil EWMA como
blocos normalizados), aprende a **política ótima em termos do estado** — não em termos dos
valores absolutos. Isso permite transferência para plantas diferentes com retreinamento mínimo.

---

## Critérios de avaliação de um cenário com queda

Para o cenário `3d_D1_00h_queda` (e qualquer cenário com queda suprível), o agente passa na
avaliação se e somente se:

1. **A planta nunca ficou sem energia durante a queda** (SOC nunca caiu abaixo da reserva
   de segurança enquanto `grid_real = 0`)

2. **O agente não desperdiçou energia potencial**: ao fim do período solar do D1, o SOC deve
   ter chegado próximo de 100% (o sol estava disponível para carregar a bateria)

3. **O agente não desconcatenou a bateria antes da queda**: o SOC ao início da queda (D2 00:00)
   deve ser ≥ `energia_necessária_queda / capacidade_utilizável` × 100%

4. **Se a bateria ficou abaixo do necessário por causa de descarga prematura**, isso é reprovação
   independentemente de qualquer outro critério

### O que NÃO é critério de falha

- A bateria ter descarregado na madrugada do D1 (antes de qualquer queda) — isso é economia legítima
- A bateria ter ficado parcialmente vazia ao fim do D2 após a queda — desde que não tenha atingido a reserva durante a queda
- O SOC não ter chegado a 100% ao fim do D1 solar em dias de baixa geração

---

## Resumo para implementação

O comportamento correto do agente em qualquer noite com possibilidade de queda é:

```
janela_segura_de_descarga = quando SOC_utilizável > max(
    déficit_restante_até_amanhecer,
    energia_necessária_para_queda_máxima_prevista
)

se dentro_da_janela_segura:
    DISCHARGE  ← economiza rede sem comprometer resiliência
senão:
    STANDBY    ← preserva para queda
```

Este princípio deve guiar tanto a função de recompensa do DQN quanto os cenários de teste:
um cenário é válido para avaliação apenas se o comportamento ótimo for fisicamente alcançável
com a bateria cheia no início da noite.
