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


def epsilon_greedy_policy(Q, state, n_actions, epsilon):
    """
    Choose an action using the epsilon-greedy policy.
    
    Parameters:
    Q: The action-value function.
    state: The current state.
    n_actions: Total number of actions.
    epsilon: Exploration rate (probability of choosing a random action).
    
    Returns:
    The chosen action (either greedy or random based on epsilon).
    """
    if np.random.rand() < epsilon:
        return np.random.choice(n_actions)  # Explore: random action
    else:
        return np.argmax(Q[state])  # Exploit: best action based on Q

def sarsa(env, num_episodes, alpha, gamma, epsilon):
    Q = np.zeros((n_states, n_actions))  # Q-table for state-action values
    max_q_values = []  # List to store max Q-values for convergence plotting
    
    for episode in range(num_episodes):
        state, _ = env.reset()  # Initialize state S
        action = epsilon_greedy_policy(Q, state, n_actions, epsilon)  # Choose action A

        while True:
            next_state, reward, terminated, truncated, _ = env.step(action)  # Take action A
            done = terminated or truncated
            
            next_action = epsilon_greedy_policy(Q, next_state, n_actions, epsilon)  # Choose A' from S'
            
            # SARSA update: Q(S, A) ← Q(S, A) + α [R + γ Q(S', A') − Q(S, A)]
            Q[state, action] += alpha * (reward + gamma * Q[next_state, next_action] * (not done) - Q[state, action])
            
            state, action = next_state, next_action  # Update S and A
            
            if done:
                break
        
        # Record the maximum Q-value after each episode
        max_q_values.append(np.max(Q))

    return Q, max_q_values

# Hyperparameters
alpha = 0.01   # Step size (learning rate)
gamma = 0.99  # Discount factor
epsilon = 0.1  # Exploration rate
num_episodes = 20000  # Number of episodes

# Run SARSA algorithm
Q, max_q_values = sarsa(env, num_episodes, alpha, gamma, epsilon)

# Print the final Q-values
print("Final Q-table:")
print(Q)

# Function to extract optimal policy from Q-values
def extract_optimal_policy_from_Q(Q):
    optimal_policy = np.argmax(Q, axis=1)  # Choose the action with the highest Q-value for each state
    return optimal_policy

# Extract the optimal policy
optimal_policy = extract_optimal_policy_from_Q(Q)

# Print the optimal policy for each state
print("Optimal Policy (Best action for each state):")
print(optimal_policy)
      
print("vs pi_opt") 
print("[1 2 4 1 2 3 2 2 3]")

# Function to extract optimal policy from Q-values
def extract_optimal_policy_from_Q(Q):
    """
    Extract the optimal policy from the Q-table.
    
    Parameters:
    Q: The action-value function (Q-table).
    
    Returns:
    policy: The optimal policy, where policy[s] is the best action for state s.
    """
    optimal_policy = np.argmax(Q, axis=1)  # Choose the action with the highest Q-value for each state
    return optimal_policy

# Extract the optimal policy
optimal_policy = extract_optimal_policy_from_Q(Q)


# Plotting convergence
def plot_convergence(max_q_values):
    plt.figure(figsize=(10, 6))
    plt.plot(max_q_values, label='Max Q-value per episode')
    plt.title("Convergence of Q-values")
    plt.xlabel("Episode")
    plt.ylabel("Max Q-value")
    plt.grid()
    plt.legend()
    plt.show()

# Plot the convergence of Q-values
plot_convergence(max_q_values)


# Plotting the heatmap for Q-values of each action separately
def plot_q_heatmaps(Q):
    """
    Plot separate heatmaps for each action in a grid layout.
    
    Parameters:
    Q: The action-value function (Q-table).
    """
    fig, axes = plt.subplots(1, n_actions, figsize=(20, 6))
    fig.suptitle("Q-Values Heatmaps for Each Action", fontsize=16)

    for action in range(n_actions):
        ax = axes[action]
        ax.imshow(Q[:, action].reshape((3, 3)), cmap='hot', interpolation='nearest')  # Assuming a 3x3 grid
        ax.set_title(f'Action {action}')
        ax.set_xlabel("States")
        ax.set_ylabel("Q-value")
        ax.set_xticks(np.arange(3))
        ax.set_yticks(np.arange(3))
        ax.set_xticklabels([f'State {i}' for i in range(3)])
        ax.set_yticklabels([f'State {i}' for i in range(3)])
        for (i, j), val in np.ndenumerate(Q[:, action].reshape((3, 3))):
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='black')

    plt.colorbar(ax.imshow(Q[:, action].reshape((3, 3)), cmap='hot'), ax=axes, label='Q-value', orientation='horizontal')
    plt.show()

# Plot the Q-values heatmaps for each action
plot_q_heatmaps(Q)