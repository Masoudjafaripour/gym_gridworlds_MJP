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

# Double Q-learning algorithm
def double_q_learning(env, num_episodes, alpha, gamma, epsilon, epsilon_decay):
    """
    Double Q-learning TD control algorithm.
    
    Parameters:
    - env: The environment.
    - num_episodes: Number of episodes to run.
    - alpha: Learning rate (step size).
    - gamma: Discount factor.
    - epsilon: Exploration rate (epsilon-greedy).
    - epsilon_decay: Factor by which to decay epsilon after each episode.
    
    Returns:
    - Q1, Q2: The learned Q-value tables.
    - max_q_values: A list of max Q-values to monitor convergence.
    """

    Q1 = np.zeros((n_states, n_actions))  # First Q-table
    Q2 = np.zeros((n_states, n_actions))  # Second Q-table

    max_q_values = []  # For tracking convergence
    
    for episode in range(num_episodes):
        state, _ = env.reset()  # Initialize S

        while True:
            # Choose A from S using epsilon-greedy policy derived from (Q1 + Q2)
            action = epsilon_greedy_policy(Q1 + Q2, state, n_actions, epsilon)

            # Take action A, observe reward R and next state S'
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # With 0.5 probability, update either Q1 or Q2
            if np.random.rand() < 0.5:
                # Update Q1
                best_action_Q1 = np.argmax(Q1[next_state])  # Best action from Q1
                Q1[state, action] += alpha * (reward + gamma * Q2[next_state, best_action_Q1] * (not done) - Q1[state, action])
            else:
                # Update Q2
                best_action_Q2 = np.argmax(Q2[next_state])  # Best action from Q2
                Q2[state, action] += alpha * (reward + gamma * Q1[next_state, best_action_Q2] * (not done) - Q2[state, action])

            # Move to the next state
            state = next_state

            if done:
                break

        # Decay epsilon after each episode (exploration rate decreases)
        # epsilon = max(epsilon * epsilon_decay, 0.01)  # Don't go below a minimum epsilon

        # Append the maximum Q-value from Q1 + Q2 for tracking convergence
        max_q_values.append(np.max(Q1 + Q2))
    
    return Q1, Q2, max_q_values

def epsilon_greedy_policy(Q, state, n_actions, epsilon):
    """
    Epsilon-greedy policy for action selection.
    
    Parameters:
    - Q: The Q-value table.
    - state: Current state.
    - n_actions: Number of possible actions.
    - epsilon: Exploration rate.
    
    Returns:
    - action: The selected action based on epsilon-greedy policy.
    """
    if np.random.rand() < epsilon:
        # Explore: random action
        return np.random.choice(n_actions)
    else:
        # Exploit: greedy action with the highest Q-value
        return np.argmax(Q[state])

# Parameters for Q-learning
alpha = 0.001           # Learning rate
gamma = 0.99          # Discount factor
epsilon = 0.1         # Exploration rate
epsilon_decay = 0.99  # Epsilon decay rate
num_episodes = 100000   # Number of episodes for learning

# Call the Double Q-learning function
Q1, Q2, max_q_values = double_q_learning(env, num_episodes, alpha, gamma, epsilon, epsilon_decay)

# Extract the optimal policy from the learned Q1 + Q2 tables
optimal_policy = np.argmax(Q1 + Q2, axis=1)
print("Optimal Policy:")
print(optimal_policy)

print("vs pi_opt") 
print("[1 2 4 1 2 3 2 2 3]")

# Optionally: Plot max Q-values over episodes to check for convergence
plt.plot(max_q_values)
plt.xlabel('Episodes')
plt.ylabel('Max Q-Value')
plt.title('Convergence of Double Q-learning')
plt.show()
