import gymnasium
import numpy as np
import matplotlib.pyplot as plt

# Define the environment
env = gymnasium.make("Gym-Gridworlds/Penalty-3x3-v0")
n_states = env.observation_space.n
n_actions = env.action_space.n

# Initialize reward and transition matrices
R = np.zeros((n_states, n_actions))
P = np.zeros((n_states, n_actions, n_states))
T = np.zeros((n_states, n_actions))

env.reset()
for s in range(n_states):
    for a in range(n_actions):
        env.set_state(s)
        s_next, r, terminated, _, _ = env.step(a)
        R[s, a] = r
        P[s, a, s_next] = 1.0
        T[s, a] = terminated

# Policy Iteration
def policy_iteration(env, gamma, theta=1e-6):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    pi = np.ones(n_states, dtype=int) * np.random.choice(n_actions, n_states)
    V = np.zeros(n_states)
    total_iterations = 0
    be_list = []

    while True:
        # Policy Evaluation
        while True:
            delta = 0
            for s in range(n_states):
                v = V[s]
                V[s] = sum([P[s, pi[s], s_next] * (R[s, pi[s]] + gamma * V[s_next]) for s_next in range(n_states)])
                delta = max(delta, abs(v - V[s]))
            if delta < theta:
                break
        be_list.append(delta)
        total_iterations += 1

        # Policy Improvement
        policy_stable = True
        for s in range(n_states):
            old_action = pi[s]
            pi[s] = np.argmax([sum([P[s, a, s_next] * (R[s, a] + gamma * V[s_next]) for s_next in range(n_states)]) for a in range(n_actions)])
            if old_action != pi[s]:
                policy_stable = False
        if policy_stable:
            break
    
    return pi, total_iterations, be_list

# Value Iteration
def value_iteration(env, gamma, theta=1e-6):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    V = np.zeros(n_states)
    pi = np.zeros(n_states, dtype=int)
    total_iterations = 0
    be_list = []

    while True:
        delta = 0
        for s in range(n_states):
            v = V[s]
            V[s] = max([sum([P[s, a, s_next] * (R[s, a] + gamma * V[s_next]) for s_next in range(n_states)]) for a in range(n_actions)])
            delta = max(delta, abs(v - V[s]))
        be_list.append(delta)
        total_iterations += 1
        if delta < theta:
            break

    for s in range(n_states):
        pi[s] = np.argmax([sum([P[s, a, s_next] * (R[s, a] + gamma * V[s_next]) for s_next in range(n_states)]) for a in range(n_actions)])

    return pi, total_iterations, be_list

# Generalized Policy Iteration
def generalized_policy_iteration(env, gamma, theta=1e-6, n_eval_steps=5):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    pi = np.ones(n_states, dtype=int) * np.random.choice(n_actions, n_states)
    V = np.zeros(n_states)
    total_iterations = 0
    be_list = []

    while True:
        for _ in range(n_eval_steps):
            delta = 0
            for s in range(n_states):
                v = V[s]
                V[s] = sum([P[s, pi[s], s_next] * (R[s, pi[s]] + gamma * V[s_next]) for s_next in range(n_states)])
                delta = max(delta, abs(v - V[s]))
            be_list.append(delta)
            total_iterations += 1
            if delta < theta:
                break
        
        policy_stable = True
        for s in range(n_states):
            old_action = pi[s]
            pi[s] = np.argmax([sum([P[s, a, s_next] * (R[s, a] + gamma * V[s_next]) for s_next in range(n_states)]) for a in range(n_actions)])
            if old_action != pi[s]:
                policy_stable = False
        if policy_stable:
            break
    
    return pi, total_iterations, be_list

# Running experiments with different initial values
initial_values = [-100, -10, -5, 0, 5, 10, 100]
results = {'VI': [], 'PI': [], 'GPI': []}

for init_value in initial_values:
    env.reset()
    
    # Initialize the value function
    V_init = np.full(env.observation_space.n, init_value)
    
    # Value Iteration
    pi, total_iter, be = value_iteration(env, gamma=0.99)
    results['VI'].append((total_iter, be))
    
    # Policy Iteration
    pi, total_iter, be = policy_iteration(env, gamma=0.99)
    results['PI'].append((total_iter, be))
    
    # Generalized Policy Iteration
    pi, total_iter, be = generalized_policy_iteration(env, gamma=0.99)
    results['GPI'].append((total_iter, be))

# Analyzing results
mean_iters = {algo: np.mean([r[0] for r in results[algo]]) for algo in results}
std_iters = {algo: np.std([r[0] for r in results[algo]]) for algo in results}

print("Mean and Std Dev of Policy Evaluations")
for algo in results:
    print(f"{algo}: Mean = {mean_iters[algo]}, Std Dev = {std_iters[algo]}")

# Plotting Bellman Error Trends
fig, axs = plt.subplots(3, 7, figsize=(20, 10))
for i, init_value in enumerate(initial_values):
    for j, algo in enumerate(['VI', 'PI', 'GPI']):
        axs[j, i].plot(results[algo][i][1])
        axs[j, i].set_title(f'$V_0$ = {init_value}')
        if i == 0:
            axs[j, i].set_ylabel(algo)
plt.show()
