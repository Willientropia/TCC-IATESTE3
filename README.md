# TCC — Sistema de Gestão Inteligente de Bateria Solar com IA

Sistema de IA para otimização de descarga de bateria em sistemas solares híbridos, utilizando **Reinforcement Learning (DQN)** com arquitetura de dois agentes.

## Objetivo

Decidir automaticamente **quando descarregar a bateria** para economizar energia da rede elétrica, sem comprometer a segurança em caso de queda de energia.

A IA toma uma decisão binária a cada 15 minutos:
- **DESCARGA** — usar a bateria para cobrir a carga, economizando rede
- **STANDBY** — preservar a bateria para emergências

## Estrutura do Projeto

```
TCCIAteste3/
├── dados/                      # Dados do inversor solar
│   ├── leituras.json           # 44 dias de dados reais (~18MB)
│   └── cenarios/               # Cenários de teste
│       ├── 3d_D1_00h_queda.json   # 3 dias com queda no D2
│       └── prova_7dias.json       # 7 dias com queda no D6
│
├── docs/                       # Documentação técnica
│   ├── objetivo-ia.md          # O problema e a solução proposta
│   ├── modelo-ia-documentacao.md  # Documentação do sistema v1 (EWMA+MPC)
│   ├── reinforcement-learning-plano.md  # Plano do DQN
│   ├── arquitetura_dois_agentes.md      # Arquitetura v2 (dois agentes)
│   ├── avaliacao_quedas.md     # Critérios de avaliação de quedas
│   └── plano-implementacao.md  # Plano de implementação da IA
│
├── simulador/                  # Simulação visual interativa
│   └── visualizacao.html       # Visualização com Chart.js
│
├── backend/                    # (será criado) Código Python da IA
│   └── rl/
│       ├── planner.py          # Agente 1 — Planejador Semanal
│       ├── environment.py      # Ambiente de simulação (Gym)
│       ├── dqn_agent.py        # Agente 2 — DQN
│       ├── reward.py           # Função de recompensa
│       ├── train.py            # Script de treinamento
│       └── model/              # Modelos treinados (.pth)
│
├── .gitignore
└── README.md
```

## Dados

- **Fonte:** Inversor solar real, coletado via Modbus/Firebase
- **Período:** ~44 dias (fevereiro-março 2026)
- **Resolução:** Leituras a cada ~30s, agregadas em slots de 15 minutos (96 por dia)
- **Bateria:** 5 kWh, 220V

## Especificações Técnicas

| Parâmetro | Valor |
|---|---|
| Capacidade da bateria | 5000 Wh (5 kWh) |
| Tensão nominal | 220V |
| Corrente máx. descarga | 27A → 5940W |
| Corrente máx. carga | 13.5A → 2970W |
| Energia máx. descarga/slot | 1485 Wh |
| Energia máx. carga/slot | 742.5 Wh |
| Reserva mínima | 10% (500 Wh) |

## Status

- [x] Dados coletados e processados
- [x] Documentação da arquitetura
- [x] Cenários de teste criados  
- [x] Simulação visual (regra ideal)
- [ ] Agente 1 — Planejador Semanal (Python)
- [ ] Agente 2 — DQN (PyTorch)
- [ ] Treinamento e validação
- [ ] Integração com simulador
