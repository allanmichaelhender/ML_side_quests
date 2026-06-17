import random
import numpy as np

class SimpleHVACSimulation:
    def __init__(self):
        self.target_temp = 21.0
        self.outdoor_temp = 32.0
        self.alpha = 0.1   # Heat leakage rate
        self.beta = 1.5    # AC cooling power
        self.reset()

    def reset(self):
        # Start the room at a hot temperature
        self.current_temp = 20.0
        return self._get_discrete_state()

    def _get_state_key(self, temp):
        # Round to nearest integer so the AI can look it up in a simple table
        return int(round(temp))

    def _get_discrete_state(self):
        return self._get_state_key(self.current_temp)

    def step(self, action):
        # 1. Physics update
        if action == 1:  # AC is ON
            self.current_temp += self.alpha * (self.outdoor_temp - self.current_temp) - self.beta
        else:            # AC is OFF
            self.current_temp += self.alpha * (self.outdoor_temp - self.current_temp)

        # Keep temperature within a reasonable simulated bounds
        self.current_temp = max(15.0, min(self.current_temp, 35.0))
        
        # 2. Reward calculation (Per user specification)
        comfort_penalty = -abs(self.current_temp - self.target_temp)
        energy_cost = -1.5 if action == 1 else 0.0
        reward = comfort_penalty + energy_cost

        return self._get_discrete_state(), reward

# --- Q-LEARNING AGENT CONFIGURATION ---
env = SimpleHVACSimulation()

# Create a Q-Table: Keys are states (integer temps 15 to 35), values are [Reward for OFF, Reward for ON]
q_table = {temp: [0.0, 0.0] for temp in range(15, 36)}

# Hyperparameters
alpha_lr = 0.1       # Learning rate (how fast it updates its memory)
discount_g = 0.9     # Discount factor (importance of future rewards)
epsilon = 0.3        # Exploration rate (30% chance to try random moves)

# --- TRAINING LOOP (5,000 Iterations) ---
print("Training the AI agent...")
for _ in range(5000):
    state = env.reset()
    
    # Run a 24-hour simulation cycle
    for hour in range(24):
        # Choose action: Explore vs Exploit
        if random.random() < epsilon:
            action = random.choice([0, 1]) # Random choice
        else:
            action = np.argmax(q_table[state]) # Best known choice

        # Take action, get environment feedback
        next_state, reward = env.step(action)

        # Update Q-table using the Temporal Difference equation
        old_value = q_table[state][action]
        next_max = np.max(q_table[next_state])
        
        # Core RL math formula updating the table
        q_table[state][action] = old_value + alpha_lr * (reward + discount_g * next_max - old_value)
        
        state = next_state

print("Training complete!\n")

# --- SIMULATION AND TESTING ---
print("Running a trained 10-hour simulation run:")
print(f"{'Hour':<6} | {'Current Temp':<14} | {'AI Action Chosen':<18} | {'Immediate Reward':<16}")
print("-" * 65)

state = env.reset()
# Turn off exploration for testing
for hour in range(1, 11):
    # Always exploit the perfected strategy
    action = np.argmax(q_table[state])
    action_text = "TURN AC ON" if action == 1 else "LEAVE AC OFF"
    
    # Save temp before step for clean printing
    display_temp = env.current_temp
    
    next_state, reward = env.step(action)
    
    print(f"{hour:<6} | {display_temp:<14.2f}°C | {action_text:<18} | {reward:<16.2f}")
    state = next_state
