import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Set up the environment
env = gym.make("Gym-Gridworlds/Penalty-3x3-v0")
n_states = env.observation_space.n
n_actions = env.action_space.n

def random_policy(state):
    """
    A random policy that selects a random action from the action space.
    """
    return np.random.choice(n_actions)

def first_visit_mc_prediction(env, policy, gamma, num_episodes):
    """
    First-visit Monte Carlo prediction algorithm for estimating V ≈ v_π.

    Parameters:
    - env: the environment to interact with
    - policy: the policy to evaluate
    - gamma: the discount factor
    - num_episodes: the number of episodes to run

    Returns:
    - V: the estimated value function for each state
    """
    
    # Initialize V(s) arbitrarily for all states and Returns(s) as empty lists
    V = np.zeros(n_states)
    Returns = defaultdict(list)  # Stores all returns for each state
    
    for episode in range(num_episodes):
        # Generate an episode following policy π
        episode_data = []
        state, _ = env.reset()
        
        done = False
        while not done:
            action = policy(state)  # Select action according to policy
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_data.append((state, reward))  # Store (state, reward)
            state = next_state
        
        # First-visit MC update
        G = 0  # Initialize return
        visited_states = set()  # To track first visits of each state
        
        # Loop through the episode backwards (from T-1 to 0)
        for t in range(len(episode_data) - 1, -1, -1):
            state_t, reward_t = episode_data[t]
            G = gamma * G + reward_t  # Update return G
            
            if state_t not in visited_states:
                visited_states.add(state_t)  # Mark first visit to state
                Returns[state_t].append(G)  # Store return for state_t
                V[state_t] = np.mean(Returns[state_t])  # Update V(state)
    
    return V

# Hyperparameters
gamma = 0.99     # Discount factor
num_episodes = 1000  # Number of episodes to train

# Run First-Visit MC Prediction
V = first_visit_mc_prediction(env, random_policy, gamma, num_episodes)

# Plot the value function as a heatmap
def plot_value_function_heatmap(V, grid_shape=(3, 3)):
    """
    Plot the value function as a heatmap.

    Parameters:
    V: The value function array.
    grid_shape: The shape of the gridworld (e.g., (3, 3) for a 3x3 grid).
    """
    plt.figure(figsize=(6, 6))
    
    # Reshape the value function into the grid shape (3x3)
    V_grid = V.reshape(grid_shape)
    
    # Create the heatmap
    plt.imshow(V_grid, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label='Value')
    
    # Add text annotations for each cell in the heatmap
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            plt.text(j, i, f'{V_grid[i, j]:.2f}', ha='center', va='center', color='black')
    
    plt.title('First-Visit MC Value Function Heatmap')
    plt.xlabel('Grid Column')
    plt.ylabel('Grid Row')
    plt.show()

# Plot the value function as a heatmap
plot_value_function_heatmap(V)

# Print the final value function
print("Final Value Function Estimates:")
print(V)



def extract_optimal_policy(env, V, gamma=1.0):
    """
    Extract the optimal policy from the estimated value function V.
    
    Parameters:
    - env: The environment with defined states and actions
    - V: The value function array (from MC prediction or any policy evaluation algorithm)
    - gamma: The discount factor

    Returns:
    - policy: A list where policy[s] is the optimal action to take in state s
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    
    # Initialize an array to hold the optimal action for each state
    optimal_policy = np.zeros(n_states, dtype=int)
    
    # For each state, calculate the action-value and choose the best action
    for state in range(n_states):
        best_action_value = float('-inf')  # Initialize the best value for comparison
        best_action = None
        
        # Loop through all possible actions to find the best one
        for action in range(n_actions):
            next_state, reward, terminated, truncated, _ = env.unwrapped.step(action)
            done = terminated or truncated
            
            # Compute the action-value Q(s, a) using the reward and the value of the next state
            action_value = reward + gamma * V[next_state] * (not done)
            
            # Select the action that maximizes the action-value
            if action_value > best_action_value:
                best_action_value = action_value
                best_action = action
        
        # Store the best action for the current state
        optimal_policy[state] = best_action
    
    return optimal_policy

# Example usage of extract_optimal_policy
optimal_policy = extract_optimal_policy(env, V, gamma=0.9)

# Print the optimal policy
print("Optimal Policy (Best action for each state):")
print(optimal_policy)
