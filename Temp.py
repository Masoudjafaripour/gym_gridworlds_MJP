import numpy as np
from collections import defaultdict

def epsilon_greedy_policy(Q, state, n_actions, epsilon):
    """
    Create an epsilon-soft policy based on Q-values.
    """
    policy = np.ones(n_actions) * (epsilon / n_actions)  # Initialize with epsilon distribution
    best_action = np.argmax(Q[state])  # Best action according to current Q-values
    policy[best_action] += (1.0 - epsilon)  # Assign the remaining probability to the greedy action
    return policy

def on_policy_every_visit_mc_control(env, gamma, epsilon, epsilon_decay, num_episodes): #(env, Q, gamma, eps_decay, max_steps, episodes_per_iteration, use_is):
    # Initialize Q-values arbitrarily and returns lists
    Q = defaultdict(lambda: np.zeros(env.action_space.n))  # Action-value function
    Returns = defaultdict(list)  # To track returns for state-action pairs
    
    # Loop for each episode
    for episode in range(num_episodes):
        # Generate an episode following the current epsilon-soft policy
        episode_data = []
        state = env.reset()
        done = False
        
        while not done:
            # Choose action using the current epsilon-soft policy
            action_probs = epsilon_greedy_policy(Q, state, env.action_space.n, epsilon)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            
            # Step in the environment
            next_state, reward, done, _ = env.step(action)
            
            # Store the state, action, and reward
            episode_data.append((state, action, reward))
            state = next_state
        
        # Initialize return G
        G = 0
        
        # Loop over the episode backwards to compute returns
        for t in range(len(episode_data) - 1, -1, -1):
            state, action, reward = episode_data[t]
            G = gamma * G + reward  # Incremental return calculation
            
            # Update Q every time this state-action pair appears (every-visit)
            Returns[(state, action)].append(G)
            Q[state][action] = np.mean(Returns[(state, action)])  # Update Q with the mean return
            
            # Find the best action (A*) for this state
            best_action = np.argmax(Q[state])
            
            # Update policy to be epsilon-soft
            for a in range(env.action_space.n):
                if a == best_action:
                    epsilon_greedy_action_prob = 1 - epsilon + (epsilon / env.action_space.n)
                else:
                    epsilon_greedy_action_prob = epsilon / env.action_space.n
        
        # Decay epsilon after each episode to reduce exploration over time
        epsilon = max(epsilon * epsilon_decay, 0.01)  # Ensuring epsilon doesn't go below a threshold (e.g., 0.01)
    
    # Derive final policy from the Q-values
    policy = {}
    for state in Q:
        best_action = np.argmax(Q[state])
        policy[state] = epsilon_greedy_policy(Q, state, env.action_space.n, epsilon)
    
    return Q, policy

# Usage example with a hypothetical OpenAI Gym environment
env = ...  # Initialize your environment here
gamma = 0.9  # Discount factor
epsilon = 1.0  # Initial exploration parameter for epsilon-soft policy
epsilon_decay = 0.99  # Epsilon decay rate per episode
num_episodes = 5000  # Number of episodes to train

Q, policy = on_policy_every_visit_mc_control(env, gamma, epsilon, epsilon_decay, num_episodes)

# Q will contain the learned action-value function
# policy will be the final epsilon-soft policy
