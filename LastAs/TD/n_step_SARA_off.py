import gymnasium as gym
import numpy as np
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

def epsilon_greedy_policy(Q, state, n_actions, epsilon=0.1):
    """Selects an action using epsilon-greedy strategy."""
    if np.random.random() < epsilon:
        return np.random.choice(n_actions)  # Explore
    else:
        return np.argmax(Q[state])  # Exploit

def behavior_policy(Q, state, n_actions, epsilon=0.2):
    """Behavior policy b(s|a) for off-policy learning (more exploratory)."""
    if np.random.random() < epsilon:
        return np.random.choice(n_actions)  # More exploration
    else:
        return np.argmax(Q[state])  # Exploitation

def n_step_off_policy_td(env, alpha, gamma, epsilon, n, num_episodes):
    """
    Off-policy n-step TD learning algorithm for estimating the action-value function Q.

    Parameters:
    - env: The environment to interact with.
    - alpha: Learning rate (0 < alpha <= 1).
    - gamma: Discount factor (0 <= gamma <= 1).
    - epsilon: Exploration rate for epsilon-greedy target policy.
    - n: Number of steps for n-step TD.
    - num_episodes: Number of episodes to train.

    Returns:
    - Q: The learned action-value function.
    """
    
    # Initialize Q(s, a) arbitrarily for all states and actions
    Q = np.zeros((n_states, n_actions))
    
    for episode in range(num_episodes):
        # Initialize S0 and A0 (start state and action using behavior policy)
        state, _ = env.reset()
        action = behavior_policy(Q, state, n_actions, epsilon)

        # Initialize memory to store n steps of rewards, states, actions, importance ratios
        rewards = []
        states = [state]
        actions = [action]
        importance_ratios = [1]

        T = float('inf')  # Initialize episode length to infinity
        t = 0  # Time step

        while True:
            if t < T:
                # Take action At, observe Rt+1 and St+1
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                rewards.append(reward)
                states.append(next_state)

                if done:
                    T = t + 1  # Episode ends at time T
                else:
                    # Select next action At+1 from behavior policy b
                    next_action = behavior_policy(Q, next_state, n_actions, epsilon)
                    actions.append(next_action)

                    # Compute importance ratio ρ for off-policy correction
                    ratio = (epsilon_greedy_policy(Q, next_state, n_actions, epsilon) == next_action) / (1 / n_actions)
                    importance_ratios.append(ratio)

            # Determine the time τ for which the value needs to be updated
            τ = t - n + 1
            if τ >= 0:
                # Compute G (n-step return)
                G = 0
                # Sum the rewards for the n-step return
                for i in range(τ + 1, min(τ + n, T) + 1):
                    if i < len(rewards):  # Check index validity
                        G += gamma ** (i - τ - 1) * rewards[i]

                # Add the bootstrapped value if we haven't reached the end of the episode
                if τ + n < T and (τ + n) < len(states):
                    G += gamma ** n * Q[states[τ + n], actions[τ + n]]

                # Calculate the importance sampling ratio for correction
                ρ = np.prod(importance_ratios[τ:min(τ + n - 1, T)])

                # Update Q(Sτ, Aτ) using the importance sampling ratio ρ
                state_τ, action_τ = states[τ], actions[τ]
                Q[state_τ, action_τ] += alpha * ρ * (G - Q[state_τ, action_τ])

            # Exit loop when τ reaches T - 1
            if τ == T - 1:
                break

            # Increment time step
            t += 1
            if t < T:
                state, action = states[t], actions[t]

    return Q

# Hyperparameters
alpha = 0.001     # Learning rate
gamma = 0.99      # Discount factor
epsilon = 0.1     # Exploration rate for target policy
n = 5            # Number of steps in n-step TD
num_episodes = 200000  # Number of episodes to train

# Run n-step off-policy TD
Q = n_step_off_policy_td(env, alpha, gamma, epsilon, n, num_episodes)

# Print the learned Q-value function
# print("Learned Q-value function:")
# print(Q)

# Extract optimal policy from Q-values
def extract_policy(Q):
    return np.argmax(Q, axis=1)

# Get optimal policy
optimal_policy = extract_policy(Q)

# Print the optimal policy for each state
optimal_policy = np.argmax(Q, axis=1)
print("Optimal Policy:")
print(optimal_policy)

print("vs pi_opt") 
print("[1 2 4 1 2 3 2 2 3]")
