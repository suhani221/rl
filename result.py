import numpy as np
import matplotlib.pyplot as plt
from maternal_environment import MaternalHealthEnvironment

n_arms = 5  # Number of arms
start_seed = 42  # Random seed
horizon = 10 # Simulation horizon
model_data_file = "maternal_A.csv"  # Model data file path (replace with your actual file path)

env = MaternalHealthEnvironment(n_arms, start_seed, horizon, model_data_file, stream_map=None)
total_rewards = np.zeros(horizon)  # Store total rewards obtained at each time step
state_counts = np.zeros((horizon, env.n_states, env.n_states))  # Store counts of transitions between states

# Run the simulation and collect data
for t in range(horizon):
    # Select actions for each arm randomly
    actions = np.random.randint(0, env.n_actions, size=n_arms)
    # Receive rewards from the environment for the chosen actions
    rewards = np.zeros(n_arms)
    for arm in range(n_arms):
        rewards[arm] = env.rewards[arm, env.current_states[arm]]
    
    # Update the state of the environment based on chosen actions and received rewards
    next_states = np.random.choice(env.n_states, size=n_arms, replace=True)  # Reset states randomly
    
    # Collect total rewards obtained at each time step
    total_rewards[t] = rewards.sum()
    
    # Collect counts of transitions between states
    for i in range(n_arms):
        state_counts[t, env.current_states[i], next_states[i]] += 1
    
    # Update the current states
    env.current_states = next_states
    
    # Print state and reward information
    print(f"Time step {t}:")
    print("States:", env.current_states)
    print("Rewards:", rewards)
    print()

# Analyze results
# Calculate the average total reward obtained over the entire simulation horizon
average_total_reward = total_rewards.mean()
print("Average total reward:", average_total_reward)

# Calculate transition probabilities
transition_probs = state_counts.sum(axis=0) / state_counts.sum(axis=0).sum(axis=1, keepdims=True)

# Create State Transition Matrix plot
plt.figure(figsize=(8, 6))
plt.imshow(transition_probs, cmap='viridis', interpolation='nearest')

# Add colorbar
plt.colorbar(label='Transition Probability')

# Add labels and title
plt.xlabel('To State')
plt.ylabel('From State')
plt.title('State Transition Matrix')

# Add state labels
state_labels = [f'State {i}' for i in range(env.n_states)]
plt.xticks(np.arange(len(state_labels)), state_labels)
plt.yticks(np.arange(len(state_labels)), state_labels)

for i in range(env.n_states):
    for j in range(env.n_states):
        plt.text(j, i, f'{transition_probs[i, j]:.2f}', ha='center', va='center', color='white')

# Plot total rewards over time
plt.figure(figsize=(10, 5))
plt.plot(np.arange(horizon), total_rewards, color='blue')
plt.title('Total Rewards Over Time')
plt.xlabel('Time Step')
plt.ylabel('Total Rewards')
plt.grid(True)
plt.show()