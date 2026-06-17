import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


# ============================================================
# 1. ENVIRONMENT — diurnal outdoor temp, randomized per episode
# ============================================================
class DiurnalHVACSimulation:
    def __init__(self):
        self.target_temp = 21.0
        self.alpha = 0.1
        self.beta = 1.5
        self.hour = 0

    def _outdoor_temp(self, hour):
        """Sine wave: min at 4 AM, max at 4 PM, ±6°C around a base that varies each episode."""
        return self.base_temp + 6.0 * math.sin(2 * math.pi * (hour - 4) / 24)

    def reset(self):
        self.current_temp = 20.0
        self.hour = 0
        # Randomize the daily average outdoor temp each episode
        self.base_temp = random.uniform(22.0, 32.0)
        outdoor = self._outdoor_temp(self.hour)
        # State: [indoor_temp, outdoor_temp, hour_normalized]
        return [self.current_temp, outdoor, self.hour / 24.0]

    def step(self, action):
        outdoor = self._outdoor_temp(self.hour)

        # Physics
        if action == 1:  # AC ON
            self.current_temp += self.alpha * (outdoor - self.current_temp) - self.beta
        else:  # AC OFF
            self.current_temp += self.alpha * (outdoor - self.current_temp)

        self.current_temp = max(15.0, min(self.current_temp, 35.0))

        # Reward
        comfort_penalty = -abs(self.current_temp - self.target_temp)
        energy_cost = -1.5 if action == 1 else 0.0
        reward = comfort_penalty + energy_cost

        self.hour += 1
        next_outdoor = self._outdoor_temp(self.hour)

        return [self.current_temp, next_outdoor, self.hour / 24.0], reward


# ============================================================
# 2. DQN — now with 3 inputs
# ============================================================
class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 16),  # [indoor, outdoor, hour]
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 3. HYPERPARAMETERS
# ============================================================
LEARNING_RATE = 0.001
DISCOUNT_GAMMA = 0.9
EPSILON = 0.3
BATCH_SIZE = 64
REPLAY_BUFFER_SIZE = 20_000
TARGET_UPDATE_INTERVAL = 10
NUM_EPISODES = 8000
STEPS_PER_EPISODE = 24

# ============================================================
# 4. SETUP
# ============================================================
env = DiurnalHVACSimulation()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

online_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(online_net.state_dict())

optimizer = optim.Adam(online_net.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)


# ============================================================
# 5. TRAINING LOOP
# ============================================================
print("Training DQN agent with diurnal outdoor temps...")
print(f"Device: {device}\n")

step_count = 0

for episode in range(NUM_EPISODES):
    state = env.reset()
    episode_loss = 0.0

    for step in range(STEPS_PER_EPISODE):
        state_tensor = torch.tensor([state], dtype=torch.float32, device=device)

        # ε-greedy
        if random.random() < EPSILON:
            action = random.choice([0, 1])
        else:
            with torch.no_grad():
                q_values = online_net(state_tensor)
                action = torch.argmax(q_values).item()

        next_state, reward = env.step(action)

        # Store experience (state includes indoor, outdoor, hour)
        replay_buffer.append((state, action, reward, next_state))
        state = next_state

        # Training step
        if len(replay_buffer) >= BATCH_SIZE:
            batch = random.sample(replay_buffer, BATCH_SIZE)
            states, actions, rewards, next_states = zip(*batch)

            states_t = torch.tensor(states, dtype=torch.float32, device=device)
            actions_t = torch.tensor(
                actions, dtype=torch.long, device=device
            ).unsqueeze(1)
            rewards_t = torch.tensor(
                rewards, dtype=torch.float32, device=device
            ).unsqueeze(1)
            next_states_t = torch.tensor(
                next_states, dtype=torch.float32, device=device
            )

            # Current Q(s, a)
            current_q = online_net(states_t).gather(1, actions_t)

            # Target: r + γ * max Q(s', a')
            with torch.no_grad():
                next_q = target_net(next_states_t).max(dim=1, keepdim=True)[0]
                target_q = rewards_t + DISCOUNT_GAMMA * next_q

            loss = loss_fn(current_q, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            episode_loss += loss.item()
            step_count += 1

    if episode % TARGET_UPDATE_INTERVAL == 0:
        target_net.load_state_dict(online_net.state_dict())

    if (episode + 1) % 1000 == 0:
        avg_loss = episode_loss / STEPS_PER_EPISODE if episode_loss > 0 else 0
        print(f"  Episode {episode + 1}/{NUM_EPISODES}  |  Avg loss: {avg_loss:.4f}")

print("Training complete!\n")


# ============================================================
# 6. TEST — evaluate on 3 different outdoor base temps
# ============================================================
for label, base in [
    ("Cool day (base=22°C)", 22.0),
    ("Warm day (base=27°C)", 27.0),
    ("Hot day  (base=32°C)", 32.0),
]:
    env.base_temp = base
    env.hour = 0
    env.current_temp = 20.0
    state = [20.0, env._outdoor_temp(0), 0.0]

    print(f"\n{label}")
    print(
        f"{'Hour':<6} | {'Indoor':<8} | {'Outdoor':<8} | {'Action':<12} | {'Reward':<7}"
    )
    print("-" * 55)

    total_reward = 0.0
    for hour in range(24):
        state_tensor = torch.tensor([state], dtype=torch.float32, device=device)
        with torch.no_grad():
            action = torch.argmax(online_net(state_tensor)).item()
        action_text = "AC ON" if action == 1 else "OFF"

        display_indoor = state[0]
        display_outdoor = state[1]

        next_state, reward = env.step(action)
        total_reward += reward

        print(
            f"{hour:<6} | {display_indoor:<8.2f} | {display_outdoor:<8.2f} | {action_text:<12} | {reward:<7.2f}"
        )
        state = next_state

    print(f"{'Total reward:':>30} {total_reward:.2f}")
