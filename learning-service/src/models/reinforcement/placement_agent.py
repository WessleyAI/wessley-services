"""
Reinforcement Learning agent for optimal component placement in 3D electrical systems.
Uses Deep Q-Network (DQN) and Actor-Critic methods for learning optimal placement strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque, namedtuple
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

@dataclass
class ComponentPlacementAction:
    """Action for component placement."""
    component_id: str
    position: Tuple[float, float, float]  # x, y, z coordinates
    orientation: Tuple[float, float, float]  # rotation angles
    action_type: str  # "place", "move", "rotate"

@dataclass
class PlacementState:
    """State representation for component placement."""
    workspace_occupancy: np.ndarray  # 3D grid of occupied spaces
    placed_components: List[Dict[str, Any]]  # Currently placed components
    remaining_components: List[Dict[str, Any]]  # Components to be placed
    current_component: Optional[Dict[str, Any]]  # Component being placed
    constraint_violations: List[str]  # Current constraint violations
    performance_metrics: Dict[str, float]  # Current layout performance

class PlacementEnvironment:
    """RL environment for component placement learning."""
    
    def __init__(self, workspace_bounds: Tuple[float, float, float],
                 grid_resolution: float = 10.0):
        self.workspace_bounds = workspace_bounds
        self.grid_resolution = grid_resolution
        
        # Discretize workspace
        self.grid_dims = (
            int(workspace_bounds[0] / grid_resolution) + 1,
            int(workspace_bounds[1] / grid_resolution) + 1,
            int(workspace_bounds[2] / grid_resolution) + 1
        )
        
        # Environment state
        self.reset()
        
        # Reward function weights
        self.reward_weights = {
            'placement_success': 100.0,
            'collision_penalty': -50.0,
            'clearance_penalty': -25.0,
            'efficiency_bonus': 20.0,
            'accessibility_bonus': 15.0,
            'completion_bonus': 200.0
        }
    
    def reset(self) -> PlacementState:
        """Reset environment to initial state."""
        self.occupancy_grid = np.zeros(self.grid_dims, dtype=bool)
        self.placed_components = []
        self.remaining_components = []
        self.current_component_idx = 0
        self.step_count = 0
        self.max_steps = 1000
        
        return self._get_current_state()
    
    def step(self, action: ComponentPlacementAction) -> Tuple[PlacementState, float, bool, Dict]:
        """Execute action and return new state, reward, done flag, and info."""
        self.step_count += 1
        
        # Execute action
        placement_successful = self._execute_placement_action(action)
        
        # Calculate reward
        reward = self._calculate_reward(action, placement_successful)
        
        # Check if episode is done
        done = (self.current_component_idx >= len(self.remaining_components) or 
                self.step_count >= self.max_steps)
        
        # Prepare info dict
        info = {
            'placement_successful': placement_successful,
            'violations': self._check_violations(),
            'step_count': self.step_count
        }
        
        # Move to next component if current placement successful
        if placement_successful and self.current_component_idx < len(self.remaining_components) - 1:
            self.current_component_idx += 1
        
        return self._get_current_state(), reward, done, info
    
    def _get_current_state(self) -> PlacementState:
        """Get current environment state."""
        current_component = None
        if self.current_component_idx < len(self.remaining_components):
            current_component = self.remaining_components[self.current_component_idx]
        
        return PlacementState(
            workspace_occupancy=self.occupancy_grid.copy(),
            placed_components=self.placed_components.copy(),
            remaining_components=self.remaining_components[self.current_component_idx:],
            current_component=current_component,
            constraint_violations=self._check_violations(),
            performance_metrics=self._calculate_performance_metrics()
        )
    
    def _execute_placement_action(self, action: ComponentPlacementAction) -> bool:
        """Execute placement action and update environment."""
        # Convert world coordinates to grid coordinates
        grid_pos = self._world_to_grid(action.position)
        
        # Check if placement is valid
        if not self._is_valid_placement(grid_pos, action.component_id):
            return False
        
        # Get component dimensions
        component = self.remaining_components[self.current_component_idx]
        component_dims = component.get('dimensions', (1, 1, 1))
        
        # Update occupancy grid
        self._update_occupancy_grid(grid_pos, component_dims, True)
        
        # Add to placed components
        placed_component = {
            'id': action.component_id,
            'position': action.position,
            'orientation': action.orientation,
            'dimensions': component_dims,
            'type': component.get('type', 'unknown')
        }
        self.placed_components.append(placed_component)
        
        return True
    
    def _is_valid_placement(self, grid_pos: Tuple[int, int, int], 
                          component_id: str) -> bool:
        """Check if placement position is valid."""
        x, y, z = grid_pos
        
        # Check bounds
        if (x < 0 or x >= self.grid_dims[0] or
            y < 0 or y >= self.grid_dims[1] or
            z < 0 or z >= self.grid_dims[2]):
            return False
        
        # Get component dimensions
        component = self.remaining_components[self.current_component_idx]
        dims = component.get('dimensions', (1, 1, 1))
        grid_dims = [max(1, int(d / self.grid_resolution)) for d in dims]
        
        # Check for collisions
        for dx in range(grid_dims[0]):
            for dy in range(grid_dims[1]):
                for dz in range(grid_dims[2]):
                    check_x, check_y, check_z = x + dx, y + dy, z + dz
                    if (check_x >= self.grid_dims[0] or 
                        check_y >= self.grid_dims[1] or 
                        check_z >= self.grid_dims[2]):
                        return False
                    if self.occupancy_grid[check_x, check_y, check_z]:
                        return False
        
        return True
    
    def _update_occupancy_grid(self, grid_pos: Tuple[int, int, int],
                             component_dims: Tuple[float, float, float],
                             occupy: bool):
        """Update occupancy grid for component placement/removal."""
        x, y, z = grid_pos
        grid_dims = [max(1, int(d / self.grid_resolution)) for d in component_dims]
        
        for dx in range(grid_dims[0]):
            for dy in range(grid_dims[1]):
                for dz in range(grid_dims[2]):
                    grid_x, grid_y, grid_z = x + dx, y + dy, z + dz
                    if (0 <= grid_x < self.grid_dims[0] and
                        0 <= grid_y < self.grid_dims[1] and
                        0 <= grid_z < self.grid_dims[2]):
                        self.occupancy_grid[grid_x, grid_y, grid_z] = occupy
    
    def _world_to_grid(self, world_pos: Tuple[float, float, float]) -> Tuple[int, int, int]:
        """Convert world coordinates to grid coordinates."""
        return (
            int(world_pos[0] / self.grid_resolution),
            int(world_pos[1] / self.grid_resolution),
            int(world_pos[2] / self.grid_resolution)
        )
    
    def _calculate_reward(self, action: ComponentPlacementAction, 
                         placement_successful: bool) -> float:
        """Calculate reward for placement action."""
        reward = 0.0
        
        if placement_successful:
            reward += self.reward_weights['placement_success']
            
            # Efficiency bonus based on wire length reduction
            efficiency_score = self._calculate_efficiency_score()
            reward += efficiency_score * self.reward_weights['efficiency_bonus']
            
            # Accessibility bonus
            accessibility_score = self._calculate_accessibility_score(action.position)
            reward += accessibility_score * self.reward_weights['accessibility_bonus']
            
            # Completion bonus if all components placed
            if self.current_component_idx >= len(self.remaining_components) - 1:
                reward += self.reward_weights['completion_bonus']
        
        else:
            # Penalties for invalid placements
            reward += self.reward_weights['collision_penalty']
            
            # Additional penalties based on violation type
            violations = self._check_violations()
            if 'clearance_violation' in violations:
                reward += self.reward_weights['clearance_penalty']
        
        return reward
    
    def _calculate_efficiency_score(self) -> float:
        """Calculate layout efficiency score."""
        if len(self.placed_components) < 2:
            return 0.5
        
        # Calculate total wire length (simplified)
        total_distance = 0.0
        component_count = len(self.placed_components)
        
        for i in range(component_count):
            for j in range(i + 1, component_count):
                pos1 = np.array(self.placed_components[i]['position'])
                pos2 = np.array(self.placed_components[j]['position'])
                distance = np.linalg.norm(pos1 - pos2)
                total_distance += distance
        
        # Normalize and invert (shorter distances = higher score)
        max_possible_distance = component_count * np.linalg.norm(self.workspace_bounds)
        if max_possible_distance > 0:
            efficiency = 1.0 - min(total_distance / max_possible_distance, 1.0)
        else:
            efficiency = 1.0
        
        return efficiency
    
    def _calculate_accessibility_score(self, position: Tuple[float, float, float]) -> float:
        """Calculate accessibility score for position."""
        # Components near edges are more accessible
        x, y, z = position
        edge_distances = [
            x, self.workspace_bounds[0] - x,
            y, self.workspace_bounds[1] - y,
            z  # Height accessibility
        ]
        
        min_edge_distance = min(edge_distances)
        max_distance = max(self.workspace_bounds) / 2
        
        return min(min_edge_distance / max_distance, 1.0)
    
    def _check_violations(self) -> List[str]:
        """Check for constraint violations in current layout."""
        violations = []
        
        # Check clearance violations
        for i, comp1 in enumerate(self.placed_components):
            for j, comp2 in enumerate(self.placed_components[i+1:], i+1):
                distance = np.linalg.norm(
                    np.array(comp1['position']) - np.array(comp2['position']))
                min_clearance = 20.0  # mm
                
                if distance < min_clearance:
                    violations.append('clearance_violation')
                    break
        
        return violations
    
    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate current layout performance metrics."""
        return {
            'efficiency': self._calculate_efficiency_score(),
            'space_utilization': len(self.placed_components) / len(self.remaining_components) if self.remaining_components else 1.0,
            'violation_count': len(self._check_violations())
        }
    
    def add_components(self, components: List[Dict[str, Any]]):
        """Add components to be placed."""
        self.remaining_components = components.copy()
        self.current_component_idx = 0

class DQNNetwork(nn.Module):
    """Deep Q-Network for component placement."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 512):
        super().__init__()
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, action_dim)
        )
    
    def forward(self, state):
        features = self.feature_extractor(state)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Dueling DQN: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values

class ActorCriticNetwork(nn.Module):
    """Actor-Critic network for continuous placement actions."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 512):
        super().__init__()
        
        # Shared feature extractor
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh()  # Actions normalized to [-1, 1]
        )
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, state):
        shared_features = self.shared_layers(state)
        
        action_probs = self.actor(shared_features)
        state_value = self.critic(shared_features)
        
        return action_probs, state_value

class ReplayBuffer:
    """Experience replay buffer for DQN training."""
    
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, experience: Experience):
        """Add experience to buffer."""
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample batch of experiences."""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class PlacementAgent:
    """RL agent for component placement optimization."""
    
    def __init__(self, state_dim: int, action_dim: int, 
                 learning_rate: float = 1e-4, device: str = 'cuda'):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Networks
        self.q_network = DQNNetwork(state_dim, action_dim).to(device)
        self.target_network = DQNNetwork(state_dim, action_dim).to(device)
        self.actor_critic = ActorCriticNetwork(state_dim, action_dim).to(device)
        
        # Optimizers
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.ac_optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        
        # Training parameters
        self.replay_buffer = ReplayBuffer()
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.99  # Discount factor
        self.target_update_freq = 1000
        self.batch_size = 32
        
        # Training metrics
        self.training_step = 0
        self.episode_rewards = []
        
        # Copy weights to target network
        self.update_target_network()
    
    def select_action(self, state: PlacementState, training: bool = True) -> ComponentPlacementAction:
        """Select action using epsilon-greedy policy or actor network."""
        # Convert state to tensor
        state_tensor = self._state_to_tensor(state).to(self.device)
        
        if training and random.random() < self.epsilon:
            # Random exploration
            return self._random_action(state)
        else:
            # Use policy network
            with torch.no_grad():
                if hasattr(self, 'use_continuous_actions') and self.use_continuous_actions:
                    action_probs, _ = self.actor_critic(state_tensor)
                    action = action_probs.cpu().numpy().flatten()
                else:
                    q_values = self.q_network(state_tensor)
                    action_idx = torch.argmax(q_values, dim=1).item()
                    action = self._action_idx_to_action(action_idx, state)
                    
            return self._tensor_to_action(action, state)
    
    def train(self, environment: PlacementEnvironment, num_episodes: int = 1000):
        """Train the agent using DQN and Actor-Critic methods."""
        logger.info(f"Starting training for {num_episodes} episodes")
        
        for episode in range(num_episodes):
            state = environment.reset()
            episode_reward = 0.0
            episode_steps = 0
            
            while True:
                # Select action
                action = self.select_action(state, training=True)
                
                # Execute action
                next_state, reward, done, info = environment.step(action)
                
                # Store experience
                experience = Experience(
                    state=self._state_to_tensor(state),
                    action=self._action_to_tensor(action),
                    reward=reward,
                    next_state=self._state_to_tensor(next_state),
                    done=done
                )
                self.replay_buffer.push(experience)
                
                episode_reward += reward
                episode_steps += 1
                
                # Train networks
                if len(self.replay_buffer) > self.batch_size:
                    self._train_dqn()
                    if episode % 10 == 0:  # Train AC less frequently
                        self._train_actor_critic()
                
                if done:
                    break
                
                state = next_state
            
            # Update target network periodically
            if episode % self.target_update_freq == 0:
                self.update_target_network()
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Record metrics
            self.episode_rewards.append(episode_reward)
            
            if episode % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                logger.info(f"Episode {episode}: Avg Reward = {avg_reward:.2f}, "
                           f"Epsilon = {self.epsilon:.3f}, Steps = {episode_steps}")
    
    def _train_dqn(self):
        """Train DQN using experience replay."""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch
        experiences = self.replay_buffer.sample(self.batch_size)
        
        # Prepare batch tensors
        states = torch.stack([exp.state for exp in experiences]).to(self.device)
        actions = torch.stack([exp.action for exp in experiences]).to(self.device)
        rewards = torch.tensor([exp.reward for exp in experiences], dtype=torch.float32).to(self.device)
        next_states = torch.stack([exp.next_state for exp in experiences]).to(self.device)
        dones = torch.tensor([exp.done for exp in experiences], dtype=torch.bool).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.long())
        
        # Target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.q_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.q_optimizer.step()
        
        self.training_step += 1
    
    def _train_actor_critic(self):
        """Train Actor-Critic network."""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch
        experiences = self.replay_buffer.sample(self.batch_size)
        
        states = torch.stack([exp.state for exp in experiences]).to(self.device)
        actions = torch.stack([exp.action for exp in experiences]).to(self.device)
        rewards = torch.tensor([exp.reward for exp in experiences], dtype=torch.float32).to(self.device)
        next_states = torch.stack([exp.next_state for exp in experiences]).to(self.device)
        dones = torch.tensor([exp.done for exp in experiences], dtype=torch.bool).to(self.device)
        
        # Get current predictions
        action_probs, state_values = self.actor_critic(states)
        _, next_state_values = self.actor_critic(next_states)
        
        # Calculate target values (TD target)
        with torch.no_grad():
            target_values = rewards + (self.gamma * next_state_values.squeeze() * ~dones)
        
        # Calculate advantages
        advantages = target_values - state_values.squeeze()
        
        # Actor loss (policy gradient)
        log_probs = F.log_softmax(action_probs, dim=1)
        action_log_probs = log_probs.gather(1, actions.long()).squeeze()
        actor_loss = -(action_log_probs * advantages.detach()).mean()
        
        # Critic loss (value function)
        critic_loss = F.mse_loss(state_values.squeeze(), target_values)
        
        # Total loss
        total_loss = actor_loss + 0.5 * critic_loss
        
        # Optimize
        self.ac_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 1.0)
        self.ac_optimizer.step()
    
    def update_target_network(self):
        """Copy weights from main network to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def _state_to_tensor(self, state: PlacementState) -> torch.Tensor:
        """Convert state to tensor representation."""
        # Flatten occupancy grid
        occupancy_flat = state.workspace_occupancy.flatten()
        
        # Component features
        placed_count = len(state.placed_components)
        remaining_count = len(state.remaining_components)
        
        # Current component features
        if state.current_component:
            comp_dims = state.current_component.get('dimensions', (0, 0, 0))
            comp_type_encoding = hash(state.current_component.get('type', '')) % 100
        else:
            comp_dims = (0, 0, 0)
            comp_type_encoding = 0
        
        # Performance metrics
        perf_metrics = list(state.performance_metrics.values())
        
        # Combine all features
        features = np.concatenate([
            occupancy_flat,
            [placed_count, remaining_count, comp_type_encoding],
            comp_dims,
            perf_metrics,
            [len(state.constraint_violations)]
        ])
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _action_to_tensor(self, action: ComponentPlacementAction) -> torch.Tensor:
        """Convert action to tensor representation."""
        # Discrete action space: position grid index
        # For simplicity, using flattened grid index
        grid_pos = (
            int(action.position[0] / 10),  # Assuming 10mm grid
            int(action.position[1] / 10),
            int(action.position[2] / 10)
        )
        
        # Convert 3D position to 1D index
        action_idx = (grid_pos[0] * 100 * 100 + 
                     grid_pos[1] * 100 + 
                     grid_pos[2])
        
        return torch.tensor([action_idx], dtype=torch.long)
    
    def _tensor_to_action(self, action_tensor: np.ndarray, 
                         state: PlacementState) -> ComponentPlacementAction:
        """Convert tensor to action."""
        if state.current_component is None:
            # Default action if no component to place
            return ComponentPlacementAction(
                component_id="none",
                position=(0, 0, 0),
                orientation=(0, 0, 0),
                action_type="place"
            )
        
        # Convert normalized action [-1, 1] to workspace coordinates
        position = (
            (action_tensor[0] + 1) * 0.5 * 500,  # Scale to workspace
            (action_tensor[1] + 1) * 0.5 * 500,
            (action_tensor[2] + 1) * 0.5 * 300
        )
        
        # Random orientation for now
        orientation = (
            random.uniform(0, 2 * np.pi),
            random.uniform(0, 2 * np.pi),
            random.uniform(0, 2 * np.pi)
        )
        
        return ComponentPlacementAction(
            component_id=state.current_component['id'],
            position=position,
            orientation=orientation,
            action_type="place"
        )
    
    def _random_action(self, state: PlacementState) -> ComponentPlacementAction:
        """Generate random action for exploration."""
        if state.current_component is None:
            return ComponentPlacementAction(
                component_id="none",
                position=(0, 0, 0),
                orientation=(0, 0, 0),
                action_type="place"
            )
        
        # Random position within workspace
        position = (
            random.uniform(0, 500),
            random.uniform(0, 500),
            random.uniform(0, 300)
        )
        
        # Random orientation
        orientation = (
            random.uniform(0, 2 * np.pi),
            random.uniform(0, 2 * np.pi),
            random.uniform(0, 2 * np.pi)
        )
        
        return ComponentPlacementAction(
            component_id=state.current_component['id'],
            position=position,
            orientation=orientation,
            action_type="place"
        )
    
    def _action_idx_to_action(self, action_idx: int, 
                            state: PlacementState) -> ComponentPlacementAction:
        """Convert discrete action index to action."""
        # Convert 1D index back to 3D grid position
        grid_z = action_idx % 100
        grid_y = (action_idx // 100) % 100
        grid_x = action_idx // (100 * 100)
        
        position = (grid_x * 10, grid_y * 10, grid_z * 10)
        
        if state.current_component is None:
            component_id = "none"
        else:
            component_id = state.current_component['id']
        
        return ComponentPlacementAction(
            component_id=component_id,
            position=position,
            orientation=(0, 0, 0),
            action_type="place"
        )
    
    def save_model(self, path: str):
        """Save trained model."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'q_optimizer_state_dict': self.q_optimizer.state_dict(),
            'ac_optimizer_state_dict': self.ac_optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'episode_rewards': self.episode_rewards
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.q_optimizer.load_state_dict(checkpoint['q_optimizer_state_dict'])
        self.ac_optimizer.load_state_dict(checkpoint['ac_optimizer_state_dict'])
        
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']
        self.episode_rewards = checkpoint['episode_rewards']
        
        logger.info(f"Model loaded from {path}")
    
    def evaluate(self, environment: PlacementEnvironment, 
                num_episodes: int = 100) -> Dict[str, float]:
        """Evaluate agent performance."""
        logger.info(f"Evaluating agent for {num_episodes} episodes")
        
        total_rewards = []
        success_rates = []
        
        for episode in range(num_episodes):
            state = environment.reset()
            episode_reward = 0.0
            placements_successful = 0
            total_placements = 0
            
            while True:
                action = self.select_action(state, training=False)
                next_state, reward, done, info = environment.step(action)
                
                episode_reward += reward
                total_placements += 1
                
                if info['placement_successful']:
                    placements_successful += 1
                
                if done:
                    break
                
                state = next_state
            
            total_rewards.append(episode_reward)
            success_rate = placements_successful / total_placements if total_placements > 0 else 0
            success_rates.append(success_rate)
        
        return {
            'avg_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'avg_success_rate': np.mean(success_rates),
            'std_success_rate': np.std(success_rates)
        }