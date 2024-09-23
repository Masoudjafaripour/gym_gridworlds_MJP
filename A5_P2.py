import gymnasium
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=3)

env = gymnasium.make("Gym-Gridworlds/Penalty-3x3-v0")
n_states = env.observation_space.n
n_actions = env.action_space.n

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

def bellman_q(pi, gamma, max_iter=1000): #computing optimal q using bellman iterative method (Q-VI)
    delta = np.inf
    iter = 0
    Q = np.zeros((n_states, n_actions))
    be = np.zeros((max_iter))
    while delta > 1e-5 and iter < max_iter:
        Q_new = R + (np.dot(P, gamma * (Q * pi)).sum(-1))
        delta = np.abs(Q_new - Q).sum()
        be[iter] = delta
        Q = Q_new
        iter += 1
    return Q
#------------------------------------------------------------------------------------------------------

# Epsilon-greedy policy 
def eps_greedy_probs_q(Q_state, eps):
    n_actions = len(Q_state)  # Number of actions for this state
    best_action = np.argmax(Q_state)  # Best action for this state

    # Initialize action probabilities with epsilon-greedy policy
    action_probs = np.ones(n_actions) * (eps / n_actions)
    action_probs[best_action] += (1.0 - eps)
    return action_probs

# Epsilon-greedy policy
def eps_greedy_probs(Q, eps):
    probs = np.ones((n_states, n_actions)) * (eps / n_actions)
    for s in range(n_states):
        if s in Q and len(Q[s]) == n_actions:  # Ensure Q[s] is initialized
            best_action = np.argmax(Q[s])
            probs[s, best_action] += (1 - eps)
    return probs
#------------------------------------------------------------------------------------------------------

# Epsilon-greedy action selection
def eps_greedy_action(Q, s, eps):
    if np.random.rand() < eps:
        return np.random.choice(np.arange(len(Q[s])))  # random action
    else:
        return np.argmax(Q[s])  # greedy action
#------------------------------------------------------------------------------------------------------

def expected_return(env, Q, gamma, episodes=10):
    G = np.zeros(episodes)
    for e in range(episodes):
        s, _ = env.reset(seed=e)
        done = False
        t = 0
        while not done:
            a = eps_greedy_action(Q, s, 0.0)
            s_next, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            G[e] += gamma**t * r
            s = s_next
            t += 1
    return G.mean()
#------------------------------------------------------------------------------------------------------


def td(env, env_eval, Q, gamma, eps, alpha, max_steps, alg):
    be = []  # To store Bellman error over time (logged every 100 steps)
    exp_ret = []  # To store the expected return over time (logged every 100 steps)
    tde = np.zeros(max_steps)  # To store TD error at every step
    eps_decay = 1.0 / max_steps  # Epsilon decay rate
    alpha_decay = 0.1 / max_steps  # Alpha decay rate

    tot_steps = 0
    while tot_steps < max_steps:
        # Reset the environment for each episode
        state, _ = env.reset()
        done = False
        while not done and tot_steps < max_steps: #
            # Choose an action using epsilon-greedy policy
            action = eps_greedy_action(Q, state, eps)

            # Take action and observe next state, reward, and termination
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Q-Learning: TD Target
            if alg == "QL":
                next_action = np.argmax(Q[next_state])  # Greedy action

                td_target = reward + gamma * Q[next_state, next_action]* (not done)
                td_error = (td_target - Q[state, action])  # TD error for Q-learning

            # SARSA: TD Target
            elif alg == "SARSA":
                next_action = eps_greedy_action(Q, state, eps)

                td_target = reward + gamma * Q[next_state, next_action]* (not done)
                td_error = (td_target - Q[state, action])  # TD error for SARSA

            # Expected SARSA: TD Target
            elif alg == "Exp_SARSA":
                expected_value = np.dot(eps_greedy_probs_q(Q[next_state], eps), Q[next_state])

                td_target = reward + gamma * expected_value * (not done)
                td_error = (td_target - Q[state, action])  # TD error for Expected SARSA

            # Log TD error at each step
            tde[tot_steps] = np.abs(td_error)  

            # Update Q-value
            Q[state, action] += alpha * td_error          

            # Log Bellman error and expected return every 100 steps
            if tot_steps % 100 == 0:
                # Compute the Bellman error using the provided bellman_q function
                # Q-Learning: Greedy policy (epsilon = 0, fully greedy)
                if alg == "QL":
                    pi = eps_greedy_probs(Q, 0)

                # SARSA: Epsilon-greedy policy (same as the exploration policy)
                elif alg == "SARSA":
                    pi = eps_greedy_probs(Q, eps)

                # Expected SARSA: Epsilon-greedy policy (same as the exploration policy)
                elif alg == "Exp_SARSA":
                    pi = eps_greedy_probs(Q, eps)

                bellman_Q = bellman_q(pi, gamma = 0.99) 
                bellman_error = np.mean(np.abs(bellman_Q - Q))  # Mean Bellman error
                be.append(bellman_error)

                # Evaluate the expected return using the current Q-function
                G = expected_return(env_eval, Q, gamma)
                exp_ret.append(G)
            
            # Move to the next state
            state = next_state  

            # Decay epsilon and alpha
            eps = max(eps - eps_decay, 0.01)
            alpha = max(alpha - alpha_decay, 0.001)

            tot_steps += 1

    return Q, be, tde, exp_ret



# https://stackoverflow.com/a/63458548/754136
def smooth(arr, span):
    re = np.convolve(arr, np.ones(span * 2 + 1) / (span * 2 + 1), mode="same")
    re[0] = arr[0]
    for i in range(1, span + 1):
        re[i] = np.average(arr[: i + span])
        re[-i] = np.average(arr[-i - span :])
    return re

def error_shade_plot(ax, data, stepsize, smoothing_window=1, **kwargs):
    y = np.nanmean(data, 0)
    x = np.arange(len(y))
    x = [stepsize * step for step in range(len(y))]
    if smoothing_window > 1:
        y = smooth(y, smoothing_window)
    (line,) = ax.plot(x, y, **kwargs)
    error = np.nanstd(data, axis=0)
    if smoothing_window > 1:
        error = smooth(error, smoothing_window)
    error = 1.96 * error / np.sqrt(data.shape[0])
    ax.fill_between(x, y - error, y + error, alpha=0.2, linewidth=0.0, color=line.get_color())

gamma = 0.99
alpha = 0.1
eps = 1.0
max_steps = 10000
horizon = 10

init_values = [-10, 0.0, 10]
algs = ["QL", "SARSA", "Exp_SARSA"]

seeds = np.arange(50)

results_be = np.zeros((
    len(init_values),
    len(algs),
    len(seeds),
    max_steps // 100,
))
results_tde = np.zeros((
    len(init_values),
    len(algs),
    len(seeds),
    max_steps,
))
results_exp_ret = np.zeros((
    len(init_values),
    len(algs),
    len(seeds),
    max_steps // 100,
))

fig, axs = plt.subplots(1, 3)
plt.ion()
plt.show()

reward_noise_std = 3.0  # re-run with 3.0

for ax in axs:
    ax.set_prop_cycle(
        color=["red", "green", "blue", "black", "orange", "cyan", "brown", "gray", "pink"]
    )
    ax.set_xlabel("Steps")

env = gymnasium.make(
    "Gym-Gridworlds/Penalty-3x3-v0",
    max_episode_steps=horizon,
    reward_noise_std=reward_noise_std,
)

env_eval = gymnasium.make(
    "Gym-Gridworlds/Penalty-3x3-v0",
    max_episode_steps=horizon,
)

for i, init_value in enumerate(init_values):
    for j, alg in enumerate(algs):
        for seed in seeds:
            np.random.seed(seed)
            Q = np.zeros((n_states, n_actions)) + init_value
            Q, be, tde, exp_ret = td(env, env_eval, Q, gamma, eps, alpha, max_steps, alg)
            results_be[i, j, seed] = be
            results_tde[i, j, seed] = tde
            results_exp_ret[i, j, seed] = exp_ret
            print(i, j, seed)
        label = f"$Q_0$: {init_value}, Alg: {alg}"
        axs[0].set_title("TD Error")
        error_shade_plot(
            axs[0],
            results_tde[i, j],
            stepsize=1,
            smoothing_window=20,
            label=label,
        )
        axs[0].legend()
        axs[0].set_ylim([0, 5])
        axs[1].set_title("Bellman Error")
        error_shade_plot(
            axs[1],
            results_be[i, j],
            stepsize=100,
            smoothing_window=20,
            label=label,
        )
        axs[1].legend()
        axs[1].set_ylim([0, 50])
        axs[2].set_title("Expected Return")
        error_shade_plot(
            axs[2],
            results_exp_ret[i, j],
            stepsize=100,
            smoothing_window=20,
            label=label,
        )
        axs[2].legend()
        axs[2].set_ylim([-5, 1])
        plt.draw()
        plt.pause(0.001)

plt.ioff()
plt.show()
