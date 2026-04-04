# Objetivo da IA — Sistema de Gestão de Bateria Solar

---

## O Problema

Sistemas solares híbridos com bateria enfrentam uma decisão recorrente: **quando usar a energia armazenada na bateria, e quando preservá-la**.

Essa decisão parece simples, mas envolve variáveis que mudam constantemente e que se afetam mutuamente:

- Qual é o perfil de consumo dessa residência? Quando há picos de carga?
- Quanto tempo falta até o sol começar a gerar energia novamente?
- Se a rede elétrica cair agora, por quanto tempo a bateria consegue sustentar a carga?
- Vale a pena descarregar agora, ou o risco de ficar sem reserva é alto demais?

Um técnico experiente consegue responder a essas perguntas para uma instalação que ele conhece bem. Mas fazer isso manualmente para cada cliente, com perfis de consumo diferentes, capacidades de sistema diferentes e padrões solares diferentes, **não escala**.

---

## Por que uma IA?

A alternativa mais óbvia a uma IA seria um sistema de regras fixas: "descarre entre 04h e 06h", "mantenha 30% de reserva", etc. Esse tipo de configuração funciona em cenários estáveis e conhecidos, mas falha quando:

- O cliente tem um perfil de consumo diferente do padrão
- A geração solar varia por clima, estação ou sombreamento
- O técnico não tem histórico suficiente para calibrar as regras com precisão
- O comportamento da residência muda ao longo do tempo (novos equipamentos, moradores, rotinas)

Uma IA resolve isso porque ela **aprende o padrão específico de cada instalação** a partir dos dados reais, sem precisar que um humano interprete e configure cada caso individualmente.

---

## O que a IA deve fazer

A IA toma uma decisão binária a cada 15 minutos:

> **DESCARREGAR** — usar a bateria para cobrir a carga atual, economizando energia da rede
> **STANDBY** — preservar a bateria, deixando a rede cobrir a carga

Essa decisão é guiada por três prioridades, nesta ordem:

### Prioridade 1 — Nunca deixar faltar energia durante uma queda
A bateria deve sempre ter reserva suficiente para sustentar a carga durante uma eventual queda de energia. Isso significa que, **antes de decidir descarregar**, a IA precisa calcular: *se a rede cair agora e a bateria precisar sustentar a carga até o sol nascer, ela consegue?* Se a resposta for não, ou se houver risco real, a decisão correta é STANDBY.

### Prioridade 2 — Minimizar o uso da rede elétrica
Dentro da margem de segurança da Prioridade 1, a IA deve maximizar o uso da bateria no lugar da rede. Isso significa escolher os melhores momentos para descarregar — idealmente quando o sol está longe de nascer, o consumo previsto é baixo e a bateria tem energia suficiente para cobrir o período.

### Prioridade 3 — Ser estável nas decisões
Ficar alternando entre DESCARREGAR e STANDBY repetidamente, sem que o cenário tenha mudado de forma significativa, é prejudicial ao inversor e indica uma política indecisa. A IA deve se comprometer com uma decisão e só rever se houver motivo concreto — especificamente, se a Prioridade 1 estiver em risco.

---

## Por que Reinforcement Learning?

A IA usa **Aprendizado por Reforço** (RL) porque o problema tem características que tornam esse paradigma mais adequado do que modelos supervisionados:

**Não existe uma "resposta certa" rotulada para cada momento.** Não há um dataset que diga "às 23h do dia X, a decisão correta era STANDBY". A decisão certa depende do que aconteceu antes e do que vai acontecer depois — é uma cadeia de decisões interdependentes.

**A consequência de uma decisão ruim pode aparecer horas depois.** Se a IA descarrega a bateria às 23h de forma irresponsável, o problema (falta de energia durante uma queda) só aparece às 2h. O RL é projetado justamente para aprender esse tipo de relação temporal, atribuindo responsabilidade a decisões passadas por consequências futuras.

**O objetivo é aprender um comportamento geral, não memorizar casos.** Com RL, o agente aprende a política — a lógica de "dado esse estado, qual a melhor ação" — não casos específicos. Isso permite que ele se adapte a novas instalações, novos perfis de carga e novos padrões solares.

---

## Como ela aprende

O agente é treinado em simulações baseadas em dados históricos reais da instalação. A cada simulação:

1. O agente observa o estado atual: nível da bateria, hora do dia, consumo recente, geração recente
2. Decide DESCARREGAR ou STANDBY
3. Recebe um sinal de recompensa que reflete a qualidade da decisão

As recompensas reforçam as três prioridades:
- Ficar sem energia durante uma queda → punição severa
- Economizar da rede quando é seguro → recompensa
- Decisões instáveis → punição leve

Com milhares de simulações, o agente aprende quais combinações de estado levam a bons resultados e ajusta sua política automaticamente.

---

## Resultado esperado

Uma IA treinada com algumas semanas de histórico de uma instalação deve ser capaz de:

- Tomar decisões corretas na vasta maioria dos cenários do cotidiano daquela residência
- Ser conservadora nos momentos de maior risco (bateria baixa, noite longa, queda iminente)
- Ser agressiva nos momentos seguros (bateria cheia, sol próximo, consumo baixo previsto)
- Adaptar-se a novas instalações com um período curto de aprendizado (~1 semana de dados)
- Generalizar para sistemas com capacidades diferentes (3kW/10kWh, 8kW/15kWh, etc.) quando retreinada com os parâmetros do novo sistema
