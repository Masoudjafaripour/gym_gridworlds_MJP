import gymnasium as gym
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

# Set up the environment
env = gym.make("Gym-Gridworlds/Penalty-3x3-v0")
n_states = env.observation_space.n
n_actions = env.action_space.n

# Set the discount factor
gamma = 0.99

# Initialize the Q-function and policy arbitrarily
Q = defaultdict(lambda: np.zeros(n_actions))
policy = np.zeros(n_states, dtype=int)

# Set the number of episodes to run
num_episodes = 30000

# Returns for keeping track of rewards and updating the Q-function
Returns = defaultdict(list)

def generate_episode(env, policy, exploring_start=False):
    """
    Generates an episode using the current policy. If exploring_start is True,
    then we randomly choose the start state and action.
    
    Returns:
    episode: List of tuples (state, action, reward)
    """
    episode = []
    
    # Always reset the environment first
    state, _ = env.reset()  # Reset before any steps
    
    if exploring_start:
        # Randomly select a start state and action for exploring starts
        state = np.random.choice(range(n_states))  # Random start state
        action = np.random.choice(range(n_actions))  # Random start action
        env.unwrapped.set_state(state)  # Set environment state directly (if allowed by environment)
        next_state, reward, terminated, truncated, _ = env.step(action)
        episode.append((state, action, reward))
        state = next_state
    else:
        while True:
            action = policy[state]
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            if terminated or truncated:
                break
    
    return episode

def mc_es_control(env, num_episodes, gamma=0.99):
    """
    Monte Carlo Exploring Starts for policy control.
    
    Parameters:
    env: The environment to run the algorithm on.
    num_episodes: Number of episodes to train.
    gamma: Discount factor.
    
    Returns:
    Q: The optimal action-value function.
    policy: The optimal policy.
    """
    for i_episode in range(num_episodes):
        # Generate an episode using exploring starts
        episode = generate_episode(env, policy, exploring_start=True)
        
        G = 0  # Return
        
        # Get the states, actions, and rewards from the episode
        states, actions, rewards = zip(*episode)
        T = len(episode)
        
        # Loop backward through the episode (First-Visit)
        visited = set()
        for t in range(T - 1, -1, -1):
            state, action, reward = states[t], actions[t], rewards[t]
            G = gamma * G + reward
            
            if (state, action) not in visited:
                visited.add((state, action))
                Returns[(state, action)].append(G)
                Q[state][action] = np.mean(Returns[(state, action)])
                
                # Update the policy to be greedy with respect to Q
                policy[state] = np.argmax(Q[state])
    
    return Q, policy

# Run the MC-ES Control algorithm
Q, optimal_policy = mc_es_control(env, num_episodes, gamma)

# Print the optimal policy in a readable format
print("Optimal Policy (State -> Best Action):")
for state in range(n_states):
    print(f"State {state}: Best Action -> {optimal_policy[state]}")

print("Optimal Policy:")
print(optimal_policy)

print("vs pi_opt") 
print("[1 2 4 1 2 3 2 2 3]")

# print("\nOptimal Q-Values (State, Action):")
# for state in Q:
#     for action in range(n_actions):
#         print(f"Q({state}, {action}) = {Q[state][action]:.2f}")

# Optional: Plot the value function as a heatmap
def plot_value_function_heatmap(Q, policy, grid_shape=(3, 3)):
    """
    Plot the state-value function based on the optimal policy as a heatmap.
    
    Parameters:
    Q: The state-action value function (Q).
    policy: The optimal policy.
    grid_shape: The shape of the gridworld (e.g., (3, 3)).
    """
    V = np.zeros(n_states)
    for state in range(n_states):
        V[state] = np.max(Q[state])  # Use max action-value for value
    
    plt.figure(figsize=(6, 6))
    V_grid = V.reshape(grid_shape)
    plt.imshow(V_grid, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label='Value')
    
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            plt.text(j, i, f'{V_grid[i, j]:.2f}', ha='center', va='center', color='black')
    
    plt.title('State-Value Function Heatmap')
    plt.xlabel('Grid Column')
    plt.ylabel('Grid Row')
    plt.show()

# Plot the value function heatmap
plot_value_function_heatmap(Q, optimal_policy)
