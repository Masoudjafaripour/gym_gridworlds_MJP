import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

np.set_printoptions(precision=3)

# Set up the environment
env = gym.make("Gym-Gridworlds/Penalty-3x3-v0")
n_states = env.observation_space.n
n_actions = env.action_space.n

# Initialize reward matrix, transition probabilities, and terminal states
R = np.zeros((n_states, n_actions))
P = np.zeros((n_states, n_actions, n_states))
T = np.zeros((n_states, n_actions))

env.reset()
for s in range(n_states):
    for a in range(n_actions):
        env.unwrapped.set_state(s)
        s_next, r, terminated, _, _ = env.step(a)
        R[s, a] = r
        P[s, a, s_next] = 1.0
        T[s, a] = terminated

P = P * (1.0 - T[..., None])  # next state probability for terminal transitions is 0


def random_policy(state):
    """
    A random policy that selects a random action from the action space.
    """
    return np.random.choice(n_actions)


def tabular_td_0(env, policy, alpha, gamma, num_episodes):
    """
    Tabular TD(0) algorithm for policy evaluation.

    Parameters:
    env: The environment, which provides states, actions, rewards, and transitions.
    policy: The policy to evaluate, which is a mapping from states to actions.
    alpha: The step size, learning rate (0 < alpha <= 1).
    gamma: The discount factor (0 <= gamma <= 1).
    num_episodes: The number of episodes to run.

    Returns:
    V: The value function for each state.
    """
    
    # Initialize value function arbitrarily for all states except terminal
    V = np.zeros(env.observation_space.n)  # nS is the number of states
    
    # Store the history of value function updates for plotting
    V_history = []
    
    # Loop for each episode
    for episode in range(num_episodes):
        # Initialize S (start state)
        state, _ = env.reset()  # Reset the environment to start a new episode
        
        # Loop for each step of the episode
        while True:
        # while total_steps < max_steps:

            # Choose action A according to policy π for state S -->> it's easy as policy is given, if Q was given, we should use sth like e-greedy
            action = policy(state) # a random map
            
            # Take action A, observe reward R and next state S'
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # TD(0) update
            V[state] = V[state] + alpha * (reward + gamma * V[next_state] - V[state])
            
            # Update state S <- S'
            state = next_state
            
            # If S' is terminal, exit the loop
            if done:
                break
        
        # Store the value function at the end of each episode for plotting
        V_history.append(np.copy(V))
    
    return V, V_history


# Hyperparameters for TD(0)
alpha = 0.01   # Learning rate
gamma = 0.995  # Discount factor
num_episodes = 40000  # Number of episodes

# Run the TD(0) algorithm with the random policy
V, V_history = tabular_td_0(env, random_policy, alpha, gamma, num_episodes)

# Plot the value function for each state over episodes
def plot_value_function(V_history):
    """
    Plot the value function over episodes.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot value function for specific states over time
    for state in range(len(V_history[0])):
        plt.plot([V[state] for V in V_history], label=f'State {state}')
    
    plt.xlabel('Episode')
    plt.ylabel('Value Function')
    plt.title('TD(0) Value Function Estimates over Episodes')
    plt.legend()
    plt.show()

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
    
    plt.title('Value Function Heatmap')
    plt.xlabel('Grid Column')
    plt.ylabel('Grid Row')
    plt.show()


# Plot the value function over episodes
plot_value_function(V_history)
plot_value_function_heatmap(V)

# Print the final value function
print("Final Value Function Estimates:")
print(V)

# one step policy improvement 

def extract_optimal_policy(env, V, gamma=0.99):
    """
    Extract the optimal policy from the value function V.
    
    Parameters:
    env: The environment which defines the states and actions.
    V: The value function array (from TD(0)).
    gamma: Discount factor.

    Returns:
    policy: A list where policy[s] is the optimal action to take in state s.
    """
    optimal_policy = np.zeros(env.observation_space.n, dtype=int)  # Stores optimal action for each state
    
    for state in range(env.observation_space.n):
        best_action_value = float('-inf')
        best_action = 0
        for action in range(env.action_space.n):
            next_state, reward, terminated, truncated, _ = env.unwrapped.step(action)
            done = terminated or truncated
            
            # Compute the value of taking action `a` in state `s`
            action_value = reward + gamma * V[next_state] * (not done)
            
            # Find the action that maximizes the value
            if action_value > best_action_value:
                best_action_value = action_value
                best_action = action
        
        optimal_policy[state] = best_action
    
    return optimal_policy


# Extract the optimal policy
optimal_policy = extract_optimal_policy(env, V)

# Print the optimal policy for each state
print("Optimal Policy (Best action for each state):")
print(optimal_policy)


# -----------------------------------

# import gymnasium
# import numpy as np
# import matplotlib.pyplot as plt
# from collections import defaultdict


# np.set_printoptions(precision=3)

# # Set up the environment
# env = gymnasium.make("Gym-Gridworlds/Penalty-3x3-v0")
# n_states = env.observation_space.n
# n_actions = env.action_space.n

# # Initialize reward matrix, transition probabilities, and terminal states
# R = np.zeros((n_states, n_actions))
# P = np.zeros((n_states, n_actions, n_states))
# T = np.zeros((n_states, n_actions))

# env.reset()
# for s in range(n_states):
#     for a in range(n_actions):
#         env.unwrapped.set_state(s)
#         s_next, r, terminated, _, _ = env.step(a)
#         R[s, a] = r
#         P[s, a, s_next] = 1.0
#         T[s, a] = terminated

# P = P * (1.0 - T[..., None])  # next state probability for terminal transitions is 0



# def TD_Prediction(policy, episodes, steps, env):
#     alpha = 0.1
#     gamma = 0.99
#     n_states = 9
#     V = np.zeros(n_states)
#     S_terminal = n_states-1
#     V(S_terminal) = 0


#     for e in episodes:
#         S = 0
#         for s in steps:
#             A = policy[S]
#             next_S, reward = env(A)
#             V[S] = V[S] + alpha*(reward + gamma*V[next_S] - V[S])
#             S = next_S
#         S = S + 1
#         if S == S_terminal:
#             break  

#     return V