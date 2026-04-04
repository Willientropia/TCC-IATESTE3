"""
dqn_agent.py — Agente 2: Deep Q-Network para decisão de descarga.

Rede neural pequena (~12.500 parâmetros) que aprende quando
descarregar a bateria baseando-se em 12 features normalizadas.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from pathlib import Path
from typing import Optional


class DQNetwork(nn.Module):
    """Rede Q-Value: 12 features → 2 Q-values (STANDBY, DISCHARGE)."""
    
    def __init__(self, n_features: int = 12, n_actions: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ReplayBuffer:
    """Experience replay buffer com amostragem aleatória."""
    
    def __init__(self, capacity: int = 50_000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> tuple:
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.BoolTensor(dones),
        )
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Agente DQN completo com:
    - Epsilon-greedy exploration
    - Experience replay
    - Target network
    - Soft update
    """
    
    def __init__(
        self,
        n_features: int = 12,
        n_actions: int = 2,
        lr: float = 5e-4,
        gamma: float = 0.97,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 50_000,
        batch_size: int = 64,
        target_update_freq: int = 25,
        tau: float = 0.005,
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.n_features = n_features
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.tau = tau
        
        # Redes
        self.q_network = DQNetwork(n_features, n_actions).to(self.device)
        self.target_network = DQNetwork(n_features, n_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)
        
        self.train_steps = 0
        self.episodes_done = 0
        
        print(f"DQN Agent: {self.q_network.count_parameters()} parâmetros, device={self.device}")
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Seleciona ação usando epsilon-greedy (treino) ou greedy (inferência).
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_t)
            return int(q_values.argmax(dim=1).item())
    
    def store(self, state, action, reward, next_state, done):
        """Armazena experiência no replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)
    
    def train_step(self) -> Optional[float]:
        """
        Um passo de treino. Retorna loss ou None se buffer insuficiente.
        """
        if len(self.buffer) < self.batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Q(s, a) atual
        q_values = self.q_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Q-target: r + γ * max Q'(s', a')
        with torch.no_grad():
            next_q = self.target_network(next_states)
            next_q_max = next_q.max(dim=1).values
            next_q_max[dones] = 0.0
            target = rewards + self.gamma * next_q_max
        
        # Loss
        loss = nn.functional.mse_loss(q_values, target)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        self.train_steps += 1
        
        # Soft update target network
        if self.train_steps % self.target_update_freq == 0:
            self._soft_update()
        
        return loss.item()
    
    def decay_epsilon(self):
        """Decaimento do epsilon no fim do episódio."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.episodes_done += 1
    
    def _soft_update(self):
        """Atualiza target network com soft update (Polyak)."""
        for target_param, param in zip(
            self.target_network.parameters(), self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
    
    def save(self, path: str | Path):
        """Salva modelo treinado."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "train_steps": self.train_steps,
            "episodes_done": self.episodes_done,
        }, path)
        print(f"Modelo salvo em {path}")
    
    def load(self, path: str | Path):
        """Carrega modelo treinado."""
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint.get("epsilon", 0.01)
        self.train_steps = checkpoint.get("train_steps", 0)
        self.episodes_done = checkpoint.get("episodes_done", 0)
        print(f"Modelo carregado de {path} (ep={self.episodes_done}, ε={self.epsilon:.3f})")
