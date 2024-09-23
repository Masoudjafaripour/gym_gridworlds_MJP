import gymnasium
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


np.set_printoptions(precision=3)

# Set up the environment
env = gymnasium.make("Gym-Gridworlds/Penalty-3x3-v0")
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
#------------------------------------------------------------------------------------------------------
# Bellman Equation Function
def bellman_q(pi, gamma):
    I = np.eye(n_states * n_actions)
    P_under_pi = (P[..., None] * pi[None, None]).reshape(n_states * n_actions, n_states * n_actions)
    return (R.ravel() * np.linalg.inv(I - gamma * P_under_pi)).sum(-1).reshape(n_states, n_actions)

# Epsilon-greedy action selection
def eps_greedy_action(Q, s, eps):
    if np.random.rand() < eps:
        return np.random.choice(np.arange(len(Q[s])))  # random action
    else:
        return np.argmax(Q[s])  # greedy action
#------------------------------------------------------------------------------------------------------

# Epsilon-greedy policy
def eps_greedy_probs(Q, eps):
    probs = np.ones((n_states, n_actions)) * (eps / n_actions)
    for s in range(n_states):
        if s in Q and len(Q[s]) == n_actions:  # Ensure Q[s] is initialized
            best_action = np.argmax(Q[s])
            probs[s, best_action] += (1 - eps)
    return probs
#------------------------------------------------------------------------------------------------------

# Collect an episode following the current policy
def episode(env, Q, eps, seed):
    np.random.seed(seed)  # for reproducibility
    data = {"s": [], "a": [], "r": []}
    s, _ = env.reset(seed=seed)
    done = False
    while not done:
        a = eps_greedy_action(Q, s, eps)
        s_next, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        data["s"].append(s)
        data["a"].append(a)
        data["r"].append(r)
        s = s_next
    return data
#------------------------------------------------------------------------------------------------------

# Compute the Bellman error for a given policy
def compute_bellman_error(Q, pi, gamma):
    Q_true = bellman_q(pi, gamma)
    bellman_error = np.abs(Q - Q_true).sum()  # sum -->> Max absolute Bellman error
    return bellman_error
#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------

# Monte Carlo Control with On-policy E-soft Policy
def monte_carlo(env, Q, gamma, eps_decay, max_steps, episodes_per_iteration):
    eps = 1.0
    total_steps = 0
    bellman_errors = []

    pi = np.ones((n_states, n_actions)) * (eps / n_actions)  
    Q_true = bellman_q(pi, gamma)
    bellman_error = np.abs(Q - Q_true).sum()
    bellman_errors.append(bellman_error)

    while total_steps < max_steps:

        visit_count = defaultdict(int)  # This stores how many times (state, action) has been visited
        for _ in range(episodes_per_iteration):
            # Generate episodes
            episode_data = episode(env, Q, eps, seed=np.random.randint(1000))
            G = 0

            # Loop over episode steps in reverse (every-visit MC)
            for t in range(len(episode_data["s"]) - 1, -1, -1):
                state, action, reward = episode_data["s"][t], episode_data["a"][t], episode_data["r"][t]
                G = gamma * G + reward
                    
                # Update the visit count
                visit_count[(state, action)] += 1
                
                # Calculate the running mean for the Q-value
                Q[state][action] += (G - Q[state][action]) / visit_count[(state, action)]  # Incremental mean update

                # Update the epsilon-greedy policy progressively
                pi = eps_greedy_probs(Q, eps)

                Q_true = bellman_q(pi, gamma)
                bellman_error = np.abs(Q - Q_true).sum()
                bellman_errors.append(bellman_error)


            total_steps += len(episode_data["s"])           

            # bellman_error_s = np.abs(bellman_errors)

        # Decay epsilon after each episode
        eps = max(eps - eps_decay * len(episode_data["s"]) / max_steps, 0.01)
    # Ensure bellman_errors has the same length as max_steps
    if len(bellman_errors) > max_steps:
        bellman_errors = bellman_errors[:max_steps]  # Trim to match max_steps
    else:
        # Pad with the last bellman error if the length is shorter
        bellman_errors += [bellman_errors[-1]] * (max_steps - len(bellman_errors))


    return Q, bellman_errors 

#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------

# Error plot with confidence interval
def error_shade_plot(ax, data, stepsize, **kwargs):
    y = np.nanmean(data, 0)
    x = np.arange(len(y))
    x = [stepsize * step for step in range(len(y))]
    (line,) = ax.plot(x, y, **kwargs)
    error = np.nanstd(data, axis=0)
    error = 1.96 * error / np.sqrt(data.shape[0])
    ax.fill_between(x, y - error, y + error, alpha=0.2, linewidth=0.0, color=line.get_color())

# Hyperparameters
init_value = 0.0
gamma = 0.9
max_steps = 200 #2000
horizon = 10

episodes_per_iteration = [1, 10, 50] # for updating Q after episodes_per_iteration
decays = [1, 2, 5]
seeds = np.arange(10) #50

# Initialize the results array
results = np.zeros((
    len(episodes_per_iteration),
    len(decays),
    len(seeds),
    max_steps,
))

# Initialize the plot
fig, axs = plt.subplots(1, 2)
plt.ion()
plt.show()

use_is = False  # Not using Importance Sampling for Part 1
for ax, reward_noise_std in zip(axs, [0.0, 3.0]):  # Two subplots: one with reward noise, one without
    ax.set_prop_cycle(
        color=["red", "green", "blue", "black", "orange", "cyan", "brown", "gray", "pink"]
    )
    ax.set_xlabel("Steps")
    ax.set_ylabel("Absolute Bellman Error")
    
    # Create environment with reward noise if applicable
    env = gymnasium.make(
        "Gym-Gridworlds/Penalty-3x3-v0",
        max_episode_steps=horizon,
        reward_noise_std=reward_noise_std,
    )
    
    # Run the experiment for all combinations of episodes per iteration and decay
    for j, episodes_p in enumerate(episodes_per_iteration):
        for k, decay in enumerate(decays):
            for seed in seeds:
                np.random.seed(seed)
                Q = np.zeros((n_states, n_actions)) + init_value
                Q, be = monte_carlo(env, Q, gamma, decay / 1, max_steps, episodes_p) #/max_steps
                results[j, k, seed] = be
                print(k, j, seed)

            error_shade_plot(
                ax,
                results[j, k],
                stepsize=1,
                label=f"Episodes: {episodes_p}, Decay: {decay}",
            )
            ax.legend()
            plt.draw()
            plt.pause(0.001)


print("done")

plt.ioff()
plt.show()




            # # Convert Q from defaultdict to a regular dict (or numpy array)
            # Q_dict = dict(Q)
            
            # # Get the Q-values and convert them to numpy arrays
            # Q_array = np.array([Q_dict[state] for state in sorted(Q_dict.keys())])
            # bellman_q_array = np.array([bellman_q(pi, gamma)[state] for state in sorted(Q_dict.keys())])

            # Compute Bellman error after each iteration (for plotting)
            # bellman_error = np.abs(Q_array - bellman_q_array).sum()  # Bellman error calculation

            # bellman_errors.append(bellman_error)




        # for _ in range(episodes_per_iteration):

        #     # Generate episodes
        #     episode_data = episode(env, Q, eps, seed=np.random.randint(1000))
        #     G = 0

        #     # Loop over episode steps in reverse (every-visit MC)
        #     for t in range(len(episode_data["s"]) - 1, -1, -1):
        #         state, action, reward = episode_data["s"][t], episode_data["a"][t], episode_data["r"][t]
        #         G = gamma * G + reward
                
        #         Returns[(state, action)].append(G)
        #         Q[state][action] = np.mean(Returns[(state, action)])  # Update Q-value




     # # Convert Q from defaultdict to a regular dict (or numpy array)
        # Q_dict = dict(Q)
        
        # # Get the Q-values and convert them to numpy arrays
        # Q_array = np.array([Q_dict[state] for state in sorted(Q_dict.keys())])
        # bellman_q_array = np.array([bellman_q(pi, gamma)[state] for state in sorted(Q_dict.keys())])

        # # Compute Bellman error after each iteration (for plotting)
        # bellman_error = np.abs(Q_array - bellman_q_array).sum()  # Bellman error calculation
        # # print(bellman_error)
        # bellman_errors.append(bellman_error)

    # # Ensure bellman_errors has length max_steps (pad or truncate if needed)
    # bellman_error_trend = np.zeros(max_steps)
    # bellman_error_trend[:min(len(bellman_errors), max_steps)] = bellman_errors[:max_steps]


# Track the count of visits for each state-action pair
        # visit_count = defaultdict(int)  # This stores how many times (state, action) has been visited

        # for _ in range(episodes_per_iteration):
        #     # Generate episodes
        #     episode_data = episode(env, Q, eps, seed=np.random.randint(1000))
        #     G = 0
        #     visited = set()

        #     # Loop over episode steps in reverse (every-visit MC)
        #     for t in range(len(episode_data["s"]) - 1, -1, -1):
        #         state, action, reward = episode_data["s"][t], episode_data["a"][t], episode_data["r"][t]
        #         G = gamma * G + reward
                
        #         if (state, action) not in visited:
        #             visited.add((state, action))  # Every-visit MC control
                    
        #             # Update the visit count
        #             visit_count[(state, action)] += 1
                    
        #             # Calculate the running mean for the Q-value
        #             Q[state][action] += (G - Q[state][action]) / visit_count[(state, action)]  # Incremental mean update



# # Monte Carlo Control with On-policy E-soft Policy
# def monte_carlo(env, Q, gamma, eps_decay, max_steps, episodes_per_iteration):
#     total_steps = 0
#     eps = 1.0
#     Q = defaultdict(lambda: np.zeros(n_actions))  # Action-value function
#     Returns = defaultdict(list)  # Track returns for state-action pairs
#     bellman_errors = []

#     while total_steps < max_steps:
#         # Generate episodes
#         episode_data = episode(env, Q, eps, seed=np.random.randint(1000))
#         G = 0

#         # Loop over episode steps in reverse (every-visit MC)
#         for t in range(len(episode_data["s"]) - 1, -1, -1):
#             state, action, reward = episode_data["s"][t], episode_data["a"][t], episode_data["r"][t]
#             G = gamma * G + reward

#             Returns[(state, action)].append(G)
#             Q[state][action] = np.mean(Returns[(state, action)])  # Update Q-value

#         # Update the epsilon-greedy policy progressively
#         pi = eps_greedy_probs(Q, eps)
        
#         # Decay epsilon after each episode
#         eps = max(eps - eps_decay * len(episode_data["s"]) / max_steps, 0.01)
#         total_steps += len(episode_data["s"])

#         # Compute Bellman error after each iteration (for plotting)
#         bellman_error = np.abs(Q - bellman_q(pi, gamma)).sum()  # Bellman error calculation
#         bellman_errors.append(bellman_error)

#     return Q, np.array(bellman_errors[:max_steps])


# # Monte Carlo Control with On-policy E-soft Policy
# def monte_carlo(env, Q, gamma, eps_decay, max_steps, episodes_per_iteration):
#     total_steps = 0
#     eps = 1.0
#     Q = defaultdict(lambda: np.zeros(n_actions))  # Action-value function
#     Returns = defaultdict(list)  # Track returns for state-action pairs
#     bellman_errors = []

#     while total_steps < max_steps:
#         # Generate episodes
#         episode_data = episode(env, Q, eps, seed=np.random.randint(1000))
#         G = 0

#         # Loop over episode steps in reverse (every-visit MC)
#         for t in range(len(episode_data["s"]) - 1, -1, -1):
#             state, action, reward = episode_data["s"][t], episode_data["a"][t], episode_data["r"][t]
#             G = gamma * G + reward
            
#             Returns[(state, action)].append(G)
#             Q[state][action] = np.mean(Returns[(state, action)])  # Update Q-value

#         # Update the epsilon-greedy policy progressively
#         pi = eps_greedy_probs(Q, eps)
        
#         # Decay epsilon after each episode
#         eps = max(eps - eps_decay * len(episode_data["s"]) / max_steps, 0.01)
#         total_steps += len(episode_data["s"])

#         # Convert Q from defaultdict to a regular dict (or numpy array)
#         Q_dict = dict(Q)
        
#         # Get the Q-values and convert them to numpy arrays
#         Q_array = np.array([Q_dict[state] for state in sorted(Q_dict.keys())])
#         bellman_q_array = np.array([bellman_q(pi, gamma)[state] for state in sorted(Q_dict.keys())])

#         # Compute Bellman error after each iteration (for plotting)
#         bellman_error = np.abs(Q_array - bellman_q_array).sum()  # Bellman error calculation
#         bellman_errors.append(bellman_error)

#     return Q, np.array(bellman_errors[:max_steps])
