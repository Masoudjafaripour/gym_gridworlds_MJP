import gymnasium as gym
import numpy as np
from collections import defaultdict

# Set up the environment
env = gym.make("Gym-Gridworlds/Penalty-3x3-v0")
n_states = env.observation_space.n
n_actions = env.action_space.n

# Set the discount factor and small epsilon for epsilon-soft policy
gamma = 0.99
epsilon = 0.1

# Initialize the Q-function and policy arbitrarily
Q = defaultdict(lambda: np.zeros(n_actions))
policy = np.zeros(n_states, dtype=int)

# Set the number of episodes to run
num_episodes = 1000

# Returns for keeping track of rewards and updating the Q-function
Returns = defaultdict(list)

def generate_episode(env, policy):
    episode = []
    
    # Ensure only the state is extracted
    state, _ = env.reset()  # Use the first value returned by env.reset()
    
    while True:
        # Ensure the policy is indexed with an integer state
        action = np.random.choice(n_actions) if np.random.rand() < epsilon else policy[state]
        next_state, reward, terminated, truncated, _ = env.step(action)
        episode.append((state, action, reward))
        if terminated or truncated:
            break
        state = next_state  # Ensure correct state assignment for next step
    
    return episode

def mc_es_control(env, num_episodes, gamma, epsilon):
    for i_episode in range(num_episodes):
        episode = generate_episode(env, policy)
        G = 0  # Return
        states, actions, rewards = zip(*episode)
        T = len(episode)
        visited = set()
        for t in range(T - 1, -1, -1):
            state, action, reward = states[t], actions[t], rewards[t]
            G = gamma * G + reward
            # if (state, action) not in visited:
            visited.add((state, action))
            Returns[(state, action)].append(G)
            Q[state][action] = np.mean(Returns[(state, action)])
            A_star = np.argmax(Q[state])
            for a in range(n_actions):
                policy[state] = ((1 - epsilon + epsilon / n_actions) if a == A_star else epsilon / n_actions)
    return Q, policy

# Run the MC-ES Control algorithm
Q, optimal_policy = mc_es_control(env, num_episodes, gamma, epsilon)

print("Optimal Policy:")
print(optimal_policy)
print("Expected Optimal:")
print("[1 2 4 1 2 3 2 2 3]")
