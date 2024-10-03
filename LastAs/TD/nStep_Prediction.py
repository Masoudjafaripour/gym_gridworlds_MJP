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


def n_step_td_prediction(env, policy, alpha, gamma, num_episodes, n):
    """
    n-step TD algorithm for policy evaluation.

    Parameters:
    env: The environment, which provides states, actions, rewards, and transitions.
    policy: The policy to evaluate, which is a mapping from states to actions.
    alpha: The step size, learning rate (0 < alpha <= 1).
    gamma: The discount factor (0 <= gamma <= 1).
    num_episodes: The number of episodes to run.
    n: The number of steps to consider for TD updates.

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

        # Initialize list to store the states, rewards, and actions
        states = [state]
        rewards = []
        done = False
        t = 0

        # Loop for each step of the episode
        while not done:
            if t < len(states):
                # Take action according to the policy
                action = policy(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # Append the next state and reward
                states.append(next_state)
                rewards.append(reward)
            else:
                next_state = states[t]

            tau = t - n + 1 # is this correct? reverse? 

            if tau >= 0:
                # Compute the n-step return G
                G = sum([gamma ** (i - tau - 1) * rewards[i] for i in range(tau + 1, min(tau + n, len(rewards)))])
                if tau + n < len(rewards):
                    G += gamma ** n * V[states[tau + n]]

                # Update the value function
                V[states[tau]] += alpha * (G - V[states[tau]])

            t += 1

        # Store the value function at the end of each episode for plotting
        V_history.append(np.copy(V))

    return V, V_history


# Hyperparameters for n-step TD
alpha = 0.01   # Learning rate
gamma = 0.995  # Discount factor
n = 20          # n-step prediction
num_episodes = 1000  # Number of episodes

# Run the n-step TD prediction algorithm with the random policy
V, V_history = n_step_td_prediction(env, random_policy, alpha, gamma, num_episodes, n)


# Plot the value function as a heatmap
def plot_value_function_heatmap(V, grid_shape=(3, 3)):
    """
    Plot the final value function as a heatmap over the states.

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


# Plot the final value function as a heatmap over the states
plot_value_function_heatmap(V)


# Print the final value function
print("Final Value Function Estimates:")
print(V)


# Extract the optimal policy from the learned value function
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


def plot_value_function_convergence(V_history, grid_shape=(3, 3)):
    """
    Plot the convergence of the value function over episodes for each state.

    Parameters:
    V_history: A list of value function arrays, where each array is the value function at the end of an episode.
    grid_shape: The shape of the gridworld (e.g., (3, 3) for a 3x3 grid).
    """
    plt.figure(figsize=(10, 6))
    
    # For each state, plot its value over episodes
    num_episodes = len(V_history)
    for state in range(len(V_history[0])):
        plt.plot(range(num_episodes), [V[state] for V in V_history], label=f'State {state}')
    
    plt.xlabel('Episode')
    plt.ylabel('Value Function Estimate')
    plt.title('Convergence of Value Function Estimates over Episodes')
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.grid(True)
    plt.show()


# Plot the convergence of the value function over episodes
plot_value_function_convergence(V_history)
