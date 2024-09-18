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
        s_next, r, terminated, truncated, _ = env.step(a)
        R[s, a] = r
        P[s, a, s_next] = 1.0
        T[s, a] = terminated

P = P * (1.0 - T[..., None])  # next state probability for terminal transitions is 0

def bellman_q(policy_probs, gamma, max_iter=1000):
    delta = np.inf
    iter = 0
    Q = np.zeros((n_states, n_actions))
    be = np.zeros((max_iter))
    while delta > 1e-5 and iter < max_iter:
        Q_new = R + np.sum(P * (gamma * np.dot(Q, policy_probs).reshape(n_states, n_actions, 1)), axis=2)
        delta = np.abs(Q_new - Q).sum()
        be[iter] = delta
        Q = Q_new
        iter += 1
    return Q

def eval(env, Q, gamma, episodes=10):
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

def eps_greedy_probs(Q, eps):
    def get_probs(s):
        probs = np.ones(n_actions) * eps / n_actions
        best_action = np.argmax(Q[s])
        probs[best_action] += 1.0 - eps
        return probs
    return get_probs

def eps_greedy_action(Q, s, eps):
    probs = eps_greedy_probs(Q, eps)(s)
    return np.random.choice(n_actions, p=probs)

def q_learning_td(Q, gamma, alpha, eps, env, env_eval, max_steps):
    be = []
    G = []
    tde = []
    eps_decay = eps / max_steps
    alpha_decay = alpha / max_steps
    tot_steps = 0
    while tot_steps < max_steps:
        s, _ = env.reset(seed=tot_steps)
        done = False
        while not done:
            a = eps_greedy_action(Q, s, eps)
            s_next, r, terminated, truncated, _ = env.step(a)
            r += np.random.normal(0, 0)  # No noise in Part 1
            done = terminated or truncated
            policy_probs = eps_greedy_probs(Q, eps)(s_next)
            td_target = r + gamma * np.dot(Q[s_next], policy_probs)
            td_error = td_target - Q[s, a]
            Q[s, a] += alpha * td_error
            tde.append(td_error)
            s = s_next
            tot_steps += 1

        if tot_steps % 100 == 0:
            policy_probs = generate_policy_probs(Q, eps)
            Q_true = bellman_q(policy_probs, gamma)
            be.append(np.mean(np.abs(Q - Q_true)))
            G.append(eval(env_eval, Q, gamma))

        eps = max(eps - eps_decay, 0.01)
        alpha = max(alpha - alpha_decay, 0.001)

    return Q, be, tde, G

def sarsa_td(Q, gamma, alpha, eps, env, env_eval, max_steps):
    be = []
    G = []
    tde = []
    eps_decay = eps / max_steps
    alpha_decay = alpha / max_steps
    tot_steps = 0
    while tot_steps < max_steps:
        s, _ = env.reset(seed=tot_steps)
        a = eps_greedy_action(Q, s, eps)
        done = False
        while not done:
            s_next, r, terminated, truncated, _ = env.step(a)
            r += np.random.normal(0, 0)  # No noise in Part 1
            done = terminated or truncated
            a_next = eps_greedy_action(Q, s_next, eps)
            td_target = r + gamma * Q[s_next, a_next]
            td_error = td_target - Q[s, a]
            Q[s, a] += alpha * td_error
            tde.append(td_error)
            s, a = s_next, a_next
            tot_steps += 1

        if tot_steps % 100 == 0:
            policy_probs = generate_policy_probs(Q, eps)
            Q_true = bellman_q(policy_probs, gamma)
            be.append(np.mean(np.abs(Q - Q_true)))
            G.append(eval(env_eval, Q, gamma))

        eps = max(eps - eps_decay, 0.01)
        alpha = max(alpha - alpha_decay, 0.001)

    return Q, be, tde, G

def expected_sarsa_td(Q, gamma, alpha, eps, env, env_eval, max_steps):
    be = []
    G = []
    tde = []
    eps_decay = eps / max_steps
    alpha_decay = alpha / max_steps
    tot_steps = 0
    while tot_steps < max_steps:
        s, _ = env.reset(seed=tot_steps)
        done = False
        while not done:
            a = eps_greedy_action(Q, s, eps)
            s_next, r, terminated, truncated, _ = env.step(a)
            r += np.random.normal(0, 0)  # No noise in Part 1
            done = terminated or truncated
            policy_probs = eps_greedy_probs(Q, eps)(s_next)
            td_target = r + gamma * np.dot(Q[s_next], policy_probs)
            td_error = td_target - Q[s, a]
            Q[s, a] += alpha * td_error
            tde.append(td_error)
            s = s_next
            tot_steps += 1

        if tot_steps % 100 == 0:
            policy_probs = generate_policy_probs(Q, eps)
            Q_true = bellman_q(policy_probs, gamma)
            be.append(np.mean(np.abs(Q - Q_true)))
            G.append(eval(env_eval, Q, gamma))

        eps = max(eps - eps_decay, 0.01)
        alpha = max(alpha - alpha_decay, 0.001)

    return Q, be, tde, G

def generate_policy_probs(Q, eps):
    policy_probs = np.zeros((n_states, n_actions))
    for s in range(n_states):
        policy_probs[s] = eps_greedy_probs(Q, eps)(s)
    return policy_probs

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
    std = np.nanstd(data, axis=0)
    if smoothing_window > 1:
        std = smooth(std, smoothing_window)
    ax.plot(x, y, **kwargs)
    ax.fill_between(x, y - std, y + std, alpha=0.2)

# Parameters
gamma = 0.9
alpha = 0.1
eps = 0.1
max_steps = 20000
init_values = [0.0, 1.0, 10.0]
seeds = [42, 7, 100]
algs = ["QL", "SARSA", "Exp_SARSA"]

# Prepare storage
results_be = np.zeros((len(init_values), len(algs), len(seeds), max_steps // 100))
results_tde = np.zeros((len(init_values), len(algs), len(seeds), max_steps))
results_G = np.zeros((len(init_values), len(algs), len(seeds), max_steps // 100))

fig, axs = plt.subplots(1, 3)
plt.ion()
plt.show()

for i, init_value in enumerate(init_values):
    for j, alg in enumerate(algs):
        for seed in seeds:
            np.random.seed(seed)
            Q = np.zeros((n_states, n_actions)) + init_value
            if alg == "QL":
                results_Q, results_be[i, j, seeds.index(seed)], results_tde[i, j, seeds.index(seed)], results_G[i, j, seeds.index(seed)] = q_learning_td(Q, gamma, alpha, eps, env, env, max_steps)
            elif alg == "SARSA":
                results_Q, results_be[i, j, seeds.index(seed)], results_tde[i, j, seeds.index(seed)], results_G[i, j, seeds.index(seed)] = sarsa_td(Q, gamma, alpha, eps, env, env, max_steps)
            elif alg == "Exp_SARSA":
                results_Q, results_be[i, j, seeds.index(seed)], results_tde[i, j, seeds.index(seed)], results_G[i, j, seeds.index(seed)] = expected_sarsa_td(Q, gamma, alpha, eps, env, env, max_steps)

        for k, alg_name in enumerate(algs):
            error_shade_plot(axs[k], results_be[i, k], 100, 10, label=f"{init_value}", color=["blue", "orange", "green"][k])
            axs[k].set_title(f"{alg_name} - Bellman Error")
            axs[k].set_xlabel("Steps")
            axs[k].set_ylabel("Bellman Error")
            axs[k].legend()

plt.show()
