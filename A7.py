import gymnasium
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

np.set_printoptions(precision=3, suppress=True)

def aggregation_features(state: np.array, centers: np.array) -> np.array:
    distance = ((state[:, None, :] - centers[None, :, :])**2).sum(-1)
    return (distance == distance.min(-1, keepdims=True)) * 1.0  # make it float

def expected_return(env, weights, gamma, episodes=100):
    G = np.zeros(episodes)
    for e in range(episodes):
        s, _ = env.reset(seed=e)
        done = False
        t = 0
        while not done:
            phi = get_phi(s)
            Q = np.dot(phi, weights).ravel()
            a = eps_greedy_action(Q, 0.0)
            s_next, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            G[e] += gamma**t * r
            s = s_next
            t += 1
    return G.mean()

def eps_greedy_action(Q, eps):
    if np.random.rand() < eps:
        return np.random.randint(n_actions)
    else:
        best = np.argwhere(Q == Q.max())
        i = np.random.choice(range(best.shape[0]))
        return best[i][0]

def softmax_action(Q, eps):
    Q_exp = np.exp((Q - np.max(Q, -1, keepdims=True)) / max(eps, 1e-12))
    probs = Q_exp / Q_exp.sum(-1, keepdims=True)
    return np.random.choice(Q.shape[-1], p=probs.ravel())

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

def fqi(seed):
    data = dict()
    ... # init dataset

    idx_data = 0  # use this to keep track of how many samples you stored
    tot_steps = 0
    weights = np.zeros((phi_dummy.shape[1], n_actions))
    exp_return = expected_return(env_eval, weights, gamma, episodes_eval)
    td_error = np.nan
    td_error_history = np.zeros((max_steps))
    exp_return_history = np.zeros((max_steps))
    pbar = tqdm(total=max_steps)
    while True:
        s, _ = env.reset(seed=seed+tot_steps)  # note that this does not make really unique seeds, but let's keep it simple
        done = False
        ep_steps = 0
        while not done and tot_steps < max_steps:
            # collect samples: select act, do env step, store sample
            # after update_frequency, do FQI
            #   for fitting_iterations
            #       fix target
            #           for gradient_steps
            #               gradient descent

            if tot_steps % log_frequency == 0:
                exp_return = expected_return(env_eval, weights, gamma, episodes_eval)
                pbar.set_description(
                    f"TDE: {np.abs(td_error).sum():.3f}, " +
                    f"G: {exp_return:.3f}"
                )
            td_error_history[tot_steps] = np.abs(td_error).sum()
            exp_return_history[tot_steps] = exp_return

            s = s_next
            tot_steps += 1
            ep_steps += 1

        pbar.update(ep_steps)
        if tot_steps >= max_steps:
            break

    pbar.close()
    return td_error_history, exp_return_history


env_id = "Gym-Gridworlds/RiverSwim-6-v0"
env = gymnasium.make(env_id, coordinate_observation=True)
env_eval = gymnasium.make(env_id, coordinate_observation=True, max_episode_steps=100)
# 100 steps horizon will give 15.241 return, but the limit (inf hor) return is actually 20-something
# but 100 steps make evaluation much faster
episodes_eval = 10

state_dim = env.observation_space.shape[0]
n_actions = env.action_space.n

# automatically set centers and sigmas
n_centers = [1, 6]
centers = np.array(
    np.meshgrid(*[
        np.linspace(env.observation_space.low[i], env.observation_space.high[i], n_centers[i])
        for i in range(env.observation_space.shape[0])
    ])
).reshape(env.observation_space.shape[0], -1).T
sigmas = (env.observation_space.high - env.observation_space.low) / n_centers / 4.0 + 1e-8 # 4.0 is arbitrary
get_phi = lambda state : aggregation_features(state.reshape(-1, state_dim), centers)  # reshape because feature functions expect shape (N, S)
phi_dummy = get_phi(env.reset()[0])  # to get the number of features

# hyperparameters
gradient_steps_sweep = [1, 20]
fitting_iterations_sweep = [1, 20]
update_frequency_sweep = [1, 20]
gamma = 0.99
alpha = 0.05
max_steps = 15000
log_frequency = 100
n_seeds = 20

fig, axs = plt.subplots(2, 2)
for ax in axs.flatten():
    ax.set_prop_cycle(
        color=["red", "green", "blue", "black", "orange", "cyan", "brown", "gray", "pink"]
    )

results_tde = np.zeros((
    len(gradient_steps_sweep),
    len(fitting_iterations_sweep),
    len(update_frequency_sweep),
    n_seeds,
    max_steps,
))
results_ret = np.zeros((
    len(gradient_steps_sweep),
    len(fitting_iterations_sweep),
    len(update_frequency_sweep),
    n_seeds,
    max_steps,
))

for i, (gradient_steps, color) in enumerate(zip(gradient_steps_sweep, ["r", "g", "b"])):
    for j, (fitting_iterations, marker) in enumerate(zip(fitting_iterations_sweep, ["o", "+", ""])):
        for k, (update_frequency, linestyle) in enumerate(zip(update_frequency_sweep, ["-", "--", ":"])):
            print(i, j, k)
            for seed in range(n_seeds):
                td_error, exp_return = fqi(seed)
                results_tde[i, j, k, seed] = td_error
                results_ret[i, j, k, seed] = exp_return

            label = f"Grad Steps: {gradient_steps}, " + \
                    f"Fit Iters: {fitting_iterations}, " + \
                    f"Upd Freq: {update_frequency}"
            plot_args = dict(
                smoothing_window=20,
                label=label,
                # marker=marker,
                # color=color,
                # linestyle=linestyle,
                # markevery=100,
            )
            error_shade_plot(
                axs[0][0],
                results_tde[i, j, k],
                stepsize=1,
                **plot_args,
            )
            error_shade_plot(
                axs[1][0],
                results_tde[i, j, k],
                stepsize=fitting_iterations*gradient_steps,
                **plot_args,
            )
            axs[0][0].legend()
            axs[0][0].set_title("TD Error")
            axs[0][0].set_xlabel("Steps")
            axs[1][0].set_xlabel("Updates")

            error_shade_plot(
                axs[0][1],
                results_ret[i, j, k],
                stepsize=1,
                **plot_args,
            )
            error_shade_plot(
                axs[1][1],
                results_ret[i, j, k],
                stepsize=fitting_iterations*gradient_steps,
                **plot_args,
            )
            axs[0][1].set_title("Expected Return")
            axs[0][1].set_xlabel("Steps")
            axs[1][1].set_xlabel("Updates")

plt.show()