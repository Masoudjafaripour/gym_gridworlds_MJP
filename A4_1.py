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
    probs = np.ones_like(Q) * (eps / n_actions)
    best_actions = np.argmax(Q, axis=1)
    for s in range(n_states):
        probs[s, best_actions[s]] += (1 - eps)
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

# Monte Carlo control with decaying epsilon
def monte_carlo(env, Q, gamma, eps_decay, max_steps, episodes_per_iteration):
    total_steps = 0
    eps = 1.0
    bellman_errors = []

    # Initialize Q-values arbitrarily and returns lists
    Q = defaultdict(lambda: np.zeros(env.action_space.n))  # Action-value function
    Returns = defaultdict(list)  # To track returns for state-action pairs

    # Initialize uniform probability distribution
    # epsilon_greedy_action_prob = np.ones(n_actions, dtype=float) * (epsilon / n_actions)
    
    while total_steps < max_steps:
        episodes_data = []
        for _ in range(episodes_per_iteration):
            # ?? Generate an episode following ⇡: S0, A0, R1, . . . , ST #1, AT #1, RT
            episode_data = episode(env, Q, eps, seed=np.random.randint(1000)) #1000?
            episodes_data.append(episode_data)
            total_steps += len(episode_data["s"])  # count the steps in this episode
            
            # Decay epsilon
            eps = max(eps - eps_decay * len(episode_data["s"]) / max_steps, 0.01)
        

        G = 0

        # Loop over the episode backwards to compute returns
        for t in range(len(episodes_data) - 1, -1, -1):
            # print(t)
            data = episodes_data[t]
            state, action, reward = data["s"][t], data["a"][t], data["r"][t] #episode_data[t]
            G = gamma * G + reward  # Incremental return calculation
            
            # Update Q every time this state-action pair appears (every-visit)
            Returns[(state, action)].append(G)
            Q[state][action] = np.mean(Returns[(state, action)])  # Update Q with the mean return
            
            # Find the best action (A*) for this state
            best_action = np.argmax(Q[state])

        
        # Compute Bellman error after each batch of episodes
        pi = eps_greedy_probs(Q, eps)
        bellman_error = compute_bellman_error(Q, pi, gamma)
        bellman_errors.append(bellman_error)
    
    # Ensure bellman_errors array matches max_steps in length
    if len(bellman_errors) < max_steps:
        # Fill in remaining steps with the last error value
        bellman_errors.extend([bellman_errors[-1]] * (max_steps - len(bellman_errors)))
    
    return Q, np.array(bellman_errors[:max_steps])

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
max_steps = 1000 #2000
horizon = 10

episodes_per_iteration = [1, 10, 50] # for updating Q after episodes_per_iteration
decays = [1, 2, 5]
seeds = np.arange(50) #50

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
                Q, bellman_error_trend = monte_carlo(env, Q, gamma, decay / 1, max_steps, episodes_p) #/max_steps
                results[j, k, seed] = bellman_error_trend
            
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




        # Update Q-function after collecting all episodes 
        # for data in episodes_data:
        #     returns = 0
        #     for t in reversed(range(len(data["s"]))): # Loop for each step of episode, t = T .1, T .2, . . . , 0:
        #         s, a, r = data["s"][t], data["a"][t], data["r"][t]
        #         returns = r + gamma * returns
        #         Returns.append(returns)
        #         Q[s, a] += (returns - Q[s, a]) / (t + 1)  # every-visit MC update
        # Initialize return G



        
            # # Update policy to be epsilon-soft
            # for a in range(env.action_space.n):
            #     if a == best_action:
            #         epsilon_greedy_action_prob = 1 - epsilon + (epsilon / env.action_space.n)
            #     else:
            #         epsilon_greedy_action_prob = epsilon / env.action_space.n