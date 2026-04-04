"""
reward.py — Função de recompensa para o DQN.

Codifica as 5 regras absolutas como sinais numéricos:
1. NUNCA deixar planta sem energia durante queda
2. NUNCA descarregar durante período solar
3. NUNCA descarregar fora da janela (início da noite)
4. NUNCA chegar ao amanhecer com > 15% de bateria
5. NUNCA ficar alternando STANDBY/DISCHARGE sem motivo
"""

# Ações
STANDBY = 0
DISCHARGE = 1


def compute_reward(
    soc: float,
    action: int,
    prev_action: int,
    in_solar: bool,
    in_discharge_window: bool,
    grid_ok: bool,
    is_sunrise: bool,
    energy_saved_wh: float,
) -> float:
    """
    Calcula recompensa para um slot.
    
    Args:
        soc: SOC atual normalizado (0-1)
        action: Ação tomada (0=STANDBY, 1=DISCHARGE)
        prev_action: Ação do slot anterior
        in_solar: Se está no período solar
        in_discharge_window: Se está na janela de descarga do plano
        grid_ok: Se a rede está funcionando
        is_sunrise: Se é o slot de transição para o período solar
        energy_saved_wh: Energia economizada neste slot
    
    Returns:
        Recompensa (float). Positiva = bom, negativa = ruim.
    """
    reward = 0.0
    
    # ========================================
    # REGRA 1: Sobreviver à queda (MÁXIMA prioridade)
    # ========================================
    if not grid_ok:
        if soc < 0.10:
            reward -= 200.0  # Catastrófico: planta sem energia
        elif soc < 0.20:
            reward += 2.0    # Sobrevivendo mas apertado
        else:
            reward += 5.0    # Sobrevivendo confortavelmente
    
    # ========================================
    # REGRA 2: Nunca descarregar no período solar
    # ========================================
    if in_solar and action == DISCHARGE:
        reward -= 20.0  # Forte punição
    
    if in_solar and action == STANDBY:
        reward += 0.5  # Pequeno incentivo
    
    # ========================================
    # REGRA 3: Respeitar a janela de descarga
    # ========================================
    if not in_solar and grid_ok:
        if action == DISCHARGE:
            if in_discharge_window:
                reward += 3.0   # Correto: descarregando na janela
                # Bonus pela energia efetivamente economizada
                reward += energy_saved_wh * 0.01
            else:
                reward -= 15.0  # Fora da janela! Muito ruim
        
        if action == STANDBY:
            if in_discharge_window and soc > 0.15:
                reward -= 5.0   # Deveria estar descarregando!
            elif not in_discharge_window:
                reward += 0.5   # Correto: standby fora da janela
    
    # ========================================
    # REGRA 4: SOC ao amanhecer
    # ========================================
    if is_sunrise:
        if soc <= 0.15:
            reward += 20.0   # Perfeito! Usou a bateria
        elif soc <= 0.25:
            reward += 8.0    # OK
        elif soc <= 0.50:
            reward -= 5.0    # Desperdiçou economia
        else:
            reward -= 20.0   # Péssimo: quase não descarregou
    
    # ========================================
    # REGRA 5: Estabilidade (sem flickering)
    # ========================================
    if action != prev_action:
        # Mudança é OK se:
        # - Entrando na janela de descarga
        # - Saindo da janela (sol nasceu)
        # - Queda/retorno de energia
        natural_transition = (
            (action == DISCHARGE and in_discharge_window) or
            (action == STANDBY and in_solar) or
            not grid_ok
        )
        if not natural_transition:
            reward -= 3.0  # Troca sem motivo
    
    # ========================================
    # REGRA auxiliar: Reserva mínima
    # ========================================
    if soc < 0.10:
        reward -= 50.0  # Nunca ficar abaixo de 10%
    
    return reward
