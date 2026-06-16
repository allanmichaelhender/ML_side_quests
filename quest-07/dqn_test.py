import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


# ============================================================
# 1. ENVIRONMENT (same physics, but returns raw float temps)
# ============================================================
class SimpleHVACSimulation:
    def __init__(self):
        self.target_temp = 21.0
        self.outdoor_temp = 32.0
        self.alpha = 0.1  # Heat leakage rate
        self.beta = 1.5  # AC cooling power
        self.reset()

    def reset(self):
        self.current_temp = 20.0
        return self.current_temp  # raw float, no rounding

    def step(self, action):
        # Physics update
        if action == 1:  # AC is ON
            self.current_temp += (
                self.alpha * (self.outdoor_temp - self.current_temp) - self.beta
            )
        else:  # AC is OFF
            self.current_temp += self.alpha * (self.outdoor_temp - self.current_temp)

        self.current_temp = max(15.0, min(self.current_temp, 35.0))

        # Reward
        comfort_penalty = -abs(self.current_temp - self.target_temp)
        energy_cost = -1.5 if action == 1 else 0.0
        reward = comfort_penalty + energy_cost

        return self.current_temp, reward


# ============================================================
# 2. DQN NETWORK
# ============================================================
class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 2),  # Q(OFF), Q(ON)
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
REPLAY_BUFFER_SIZE = 10_000
TARGET_UPDATE_INTERVAL = 10  # episodes between target network syncs
NUM_EPISODES = 5000
STEPS_PER_EPISODE = 24

# ============================================================
# 4. SETUP
# ============================================================
env = SimpleHVACSimulation()

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
print("Training the DQN agent...")
print(f"Device: {device}\n")

step_count = 0

for episode in range(NUM_EPISODES):
    temp = env.reset()
    episode_loss = 0.0

    for step in range(STEPS_PER_EPISODE):
        # --- ε-greedy action ---
        state_tensor = torch.tensor([[temp]], dtype=torch.float32, device=device)

        if random.random() < EPSILON:
            action = random.choice([0, 1])
        else:
            with torch.no_grad():
                q_values = online_net(state_tensor)
                action = torch.argmax(q_values).item()

        next_temp, reward = env.step(action)

        # Store experience
        replay_buffer.append((temp, action, reward, next_temp))
        temp = next_temp

        # --- Training step (once we have enough experiences) ---
        if len(replay_buffer) >= BATCH_SIZE:
            # Sample a random mini-batch
            batch = random.sample(replay_buffer, BATCH_SIZE)
            temps, actions, rewards, next_temps = zip(*batch)

            temps_t = torch.tensor(temps, dtype=torch.float32, device=device).unsqueeze(
                1
            )
            actions_t = torch.tensor(
                actions, dtype=torch.long, device=device
            ).unsqueeze(1)
            rewards_t = torch.tensor(
                rewards, dtype=torch.float32, device=device
            ).unsqueeze(1)
            next_temps_t = torch.tensor(
                next_temps, dtype=torch.float32, device=device
            ).unsqueeze(1)

            # Current Q(s, a)
            current_q = online_net(temps_t).gather(1, actions_t)

            # Target: r + γ * max Q(s', a')  (using target network for stability)
            with torch.no_grad():
                next_q = target_net(next_temps_t).max(dim=1, keepdim=True)[0]
                target_q = rewards_t + DISCOUNT_GAMMA * next_q

            loss = loss_fn(current_q, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            episode_loss += loss.item()
            step_count += 1

    # Sync target network periodically
    if episode % TARGET_UPDATE_INTERVAL == 0:
        target_net.load_state_dict(online_net.state_dict())

    # Progress indicator
    if (episode + 1) % 1000 == 0:
        avg_loss = episode_loss / STEPS_PER_EPISODE if episode_loss > 0 else 0
        print(f"  Episode {episode + 1}/{NUM_EPISODES}  |  Avg loss: {avg_loss:.4f}")

print("Training complete!\n")


# ============================================================
# 6. TESTING (greedy, no exploration)
# ============================================================
print("Running a trained 10-hour simulation run:")
print(
    f"{'Hour':<6} | {'Current Temp':<14} | {'AI Action Chosen':<18} | {'Immediate Reward':<16}"
)
print("-" * 65)

temp = env.reset()

for hour in range(1, 11):
    state_tensor = torch.tensor([[temp]], dtype=torch.float32, device=device)

    with torch.no_grad():
        q_values = online_net(state_tensor)
        action = torch.argmax(q_values).item()

    action_text = "TURN AC ON" if action == 1 else "LEAVE AC OFF"

    # Save temp before step for clean printing
    display_temp = temp

    next_temp, reward = env.step(action)

    print(f"{hour:<6} | {display_temp:<14.2f}°C | {action_text:<18} | {reward:<16.2f}")
    temp = next_temp


# ============================================================
# 7. BONUS: print learned Q-values across the temperature range
# ============================================================
print("\nLearned Q-values:")
print(f"{'Temp':<6} | {'Q(OFF)':<10} | {'Q(ON)':<10} | {'Best Action':<12}")
print("-" * 42)

with torch.no_grad():
    for t in range(15, 36):
        x = torch.tensor([[float(t)]], dtype=torch.float32, device=device)
        q = online_net(x).cpu().numpy()[0]
        best = "OFF" if q[0] >= q[1] else "ON"
        print(f"{t:>4}°C  | {q[0]:<+10.4f} | {q[1]:<+10.4f} | {best:<12}")
