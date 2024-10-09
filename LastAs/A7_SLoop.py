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

# def fqi(seed):
    N = gradient_steps
    K = fitting_iterations
    D = datasize
    S = 3 

    data = dict()
    eps = 1.0
    idx_data = 0
    tot_steps = 0
    weights = np.zeros((phi_dummy.shape[1], n_actions))  # Initialize weights
    exp_return = expected_return(env_eval, weights, gamma, episodes_eval)
    abs_td_error = np.nan
    exp_return_history = np.zeros((max_steps))
    td_error_history = np.zeros((max_steps))
    pbar = tqdm(total=max_steps)
    
    while tot_steps < max_steps:
        # 1. For S environment steps:
        step_counter = 0
        while step_counter < S and tot_steps < max_steps:
            s, _ = env.reset(seed=seed + tot_steps)  # reset environment
            done = False
            ep_steps = 0
            while not done and step_counter < S:
                phi = get_phi(s)  # Compute features for state
                Q = np.dot(phi, weights).ravel()  # Estimate Q-values
                a = eps_greedy_action(Q, eps)  # Select action using eps-greedy policy
                s_next, r, terminated, truncated, _ = env.step(a)  # Execute action
                done = terminated or truncated  # Check if episode ends
                step_counter += 1
                tot_steps += 1
                ep_steps += 1
                eps = max(eps - 1.0 / max_steps, 0.5)  # Decaying epsilon

                # 2. Collect D samples:
                if idx_data < D:
                    if idx_data == 0:
                        data["s"] = np.zeros((D, s.size))
                        data["a"] = np.zeros((D,), dtype=int)
                        data["r"] = np.zeros((D,))
                        data["s_next"] = np.zeros((D, s_next.size))
                    data["s"][idx_data] = s
                    data["a"][idx_data] = a
                    data["r"][idx_data] = r
                    data["s_next"][idx_data] = s_next
                    idx_data += 1

                s = s_next  # Move to the next state

            # Reset the sample collection index for the next round
            if idx_data >= D:
                idx_data = 0

        # 3. For K iterations (Policy Improvement):
        for k in range(K):
            # 4. Fix TD targets:
            phi_next = get_phi(data["s_next"])
            td_target = data["r"] + gamma * np.dot(phi_next, weights).max(-1)

            weights_before_fit = weights.copy()
            abs_td_error_history = []

            # 5. For N steps (Gradient Descent on Q-values):
            for n in range(N):
                weights_before_step = weights.copy()

                # 6. Update weights using gradient descent:
                for act in range(n_actions):
                    action_idx = data["a"] == act  # Find which samples correspond to this action
                    if action_idx.any():  # Ensure there are samples for this action
                        phi = get_phi(data["s"][action_idx])  # Get phi for states with action `act`
                        td_prediction = np.dot(phi, weights[:, act])  # Predict TD values
                        td_error_act = td_target[action_idx] - td_prediction  # Calculate TD error
                        abs_td_error = np.abs(td_error_act).mean()  # Average TD error
                        # abs_td_error_history.append(abs_td_error)
                        gradient = (td_error_act[..., None] * phi).mean(0)  # Calculate gradient
                        weights[:, act] += alpha * gradient  # Gradient descent step

                # Append TD error for logging
                abs_td_error_history.append(abs_td_error)

                # 7. If weights didn’t change, break (Convergence Check):
                if np.allclose(weights, weights_before_step, rtol=1e-5, atol=1e-5):
                    break

            # 8. If policy didn’t change, break:
            if np.allclose(weights, weights_before_fit, rtol=1e-5, atol=1e-5):
                break

        # Log progress and TD error
        if tot_steps % log_frequency == 0:
            exp_return = expected_return(env_eval, weights, gamma, episodes_eval)
            mean_td_error = np.mean(abs_td_error_history) if abs_td_error_history else 0
            pbar.set_description(
                f"TDE: {mean_td_error:.3f}, " +
                f"G: {exp_return:.3f}"
            )
        exp_return_history[tot_steps] = exp_return
        td_error_history[tot_steps] = mean_td_error

        pbar.update(ep_steps)

    pbar.close()
    return td_error_history, exp_return_history


# def fqi(seed):
    N = gradient_steps
    K = fitting_iterations
    D = datasize
    S = 3

    data = dict()
    eps = 1.0
    idx_data = 0
    tot_steps = 0
    weights = np.zeros((phi_dummy.shape[1], n_actions))  # Initialize weights
    exp_return = expected_return(env_eval, weights, gamma, episodes_eval)
    exp_return_history = np.zeros((max_steps))
    td_error_history = np.zeros((max_steps))
    pbar = tqdm(total=max_steps)

    while tot_steps < max_steps:
        # 1. For S environment steps:
        step_counter = 0
        while step_counter < S and tot_steps < max_steps:
            s, _ = env.reset(seed=seed + tot_steps)  # reset environment
            done = False
            ep_steps = 0
            while not done and step_counter < S:
                phi = get_phi(s)  # Compute features for state
                Q = np.dot(phi, weights).ravel()  # Estimate Q-values
                a = eps_greedy_action(Q, eps)  # Select action using eps-greedy policy
                s_next, r, terminated, truncated, _ = env.step(a)  # Execute action
                done = terminated or truncated  # Check if episode ends
                step_counter += 1
                tot_steps += 1
                ep_steps += 1
                eps = max(eps - 1.0 / max_steps, 0.5)  # Decaying epsilon

                # 2. Collect D samples:
                if idx_data < D:
                    if idx_data == 0:
                        data["s"] = np.zeros((D, s.size))
                        data["a"] = np.zeros((D,), dtype=int)
                        data["r"] = np.zeros((D,))
                        data["s_next"] = np.zeros((D, s_next.size))
                    data["s"][idx_data] = s
                    data["a"][idx_data] = a
                    data["r"][idx_data] = r
                    data["s_next"][idx_data] = s_next
                    idx_data += 1

                s = s_next  # Move to the next state

            # Reset the sample collection index for the next round
            if idx_data >= D:
                idx_data = 0

        # 3. For K iterations (Policy Improvement):
        for k in range(K):
            # 4. Fix TD targets:
            phi_next = get_phi(data["s_next"])
            td_target = data["r"] + gamma * np.dot(phi_next, weights).max(-1)

            weights_before_fit = weights.copy()
            abs_td_error_history = []
            mean_td_error = 0  # Default value to prevent unbound error

            # 5. For N steps (Gradient Descent on Q-values):
            for n in range(N):
                weights_before_step = weights.copy()

                # 6. Update weights using gradient descent:
                for act in range(n_actions):
                    action_idx = data["a"] == act  # Find which samples correspond to this action
                    if action_idx.any():  # Ensure there are samples for this action
                        phi = get_phi(data["s"][action_idx])  # Get phi for states with action `act`
                        td_prediction = np.dot(phi, weights[:, act])  # Predict TD values
                        td_error_act = td_target[action_idx] - td_prediction  # Calculate TD error
                        abs_td_error = np.abs(td_error_act).mean()  # Average TD error
                        abs_td_error_history.append(abs_td_error)   # Append TD error
                        gradient = (td_error_act[..., None] * phi).mean(0)  # Calculate gradient
                        weights[:, act] += alpha * gradient  # Gradient descent step

                # 7. If weights didn’t change, break (Convergence Check):
                if np.allclose(weights, weights_before_step, rtol=1e-5, atol=1e-5):
                    break

            # 8. If policy didn’t change, break:
            if np.allclose(weights, weights_before_fit, rtol=1e-5, atol=1e-5):
                break

        # Log progress and TD error
        if tot_steps % log_frequency == 0:
            exp_return = expected_return(env_eval, weights, gamma, episodes_eval)
            mean_td_error = np.mean(abs_td_error_history) if abs_td_error_history else 0
            pbar.set_description(
                f"TDE: {mean_td_error:.3f}, " +
                f"G: {exp_return:.3f}"
            )
        exp_return_history[tot_steps] = exp_return
        td_error_history[tot_steps] = mean_td_error

        pbar.update(ep_steps)

    pbar.close()
    return td_error_history, exp_return_history

def fqi(seed):
    N = gradient_steps
    K = fitting_iterations
    D = datasize
    S = 3

    data = dict()
    eps = 1.0
    idx_data = 0
    tot_steps = 0
    weights = np.zeros((phi_dummy.shape[1], n_actions))  # Initialize weights
    exp_return = expected_return(env_eval, weights, gamma, episodes_eval)
    exp_return_history = np.zeros((max_steps))
    td_error_history = np.zeros((max_steps))
    pbar = tqdm(total=max_steps)

    while tot_steps < max_steps:
        # 1. For S environment steps:
        state_counter = 0
        while state_counter < S and tot_steps < max_steps:
            s, _ = env.reset(seed=seed + tot_steps)  # reset environment
            done = False

            ep_steps = 0
            
            while not done and state_counter < S:
                phi = get_phi(s)  # Compute features for state
                Q = np.dot(phi, weights).ravel()  # Estimate Q-values
                a = eps_greedy_action(Q, eps)  # Select action using eps-greedy policy
                s_next, r, terminated, truncated, _ = env.step(a)  # Execute action
                done = terminated or truncated  # Check if episode ends

                state_counter += 1 # why here??

                tot_steps += 1

                # Make sure we don't exceed max_steps
                if tot_steps >= max_steps:
                    break

                ep_steps += 1
                eps = max(eps - 1.0 / max_steps, 0.5)  # Decaying epsilon

                # 2. Collect D samples:
                if idx_data < D:
                    if idx_data == 0:
                        data["s"] = np.zeros((D, s.size))
                        data["a"] = np.zeros((D,), dtype=int)
                        data["r"] = np.zeros((D,))
                        data["s_next"] = np.zeros((D, s_next.size))
                    data["s"][idx_data] = s
                    data["a"][idx_data] = a
                    data["r"][idx_data] = r
                    data["s_next"][idx_data] = s_next
                    idx_data += 1

                s = s_next  # Move to the next state

            # Reset the sample collection index for the next round
            if idx_data >= D:
                idx_data = 0

        # 3. For K iterations (Policy Improvement):
        for k in range(K):
            # 4. Fix TD targets:
            phi_next = get_phi(data["s_next"])
            td_target = data["r"] + gamma * np.dot(phi_next, weights).max(-1)

            weights_before_fit = weights.copy()
            abs_td_error_history = []
            mean_td_error = 0  # Default value to prevent unbound error

            # 5. For N steps (Gradient Descent on Q-values):
            for n in range(N):
                weights_before_step = weights.copy()

                # 6. Update weights using gradient descent:
                for act in range(n_actions):
                    action_idx = data["a"] == act  # Find which samples correspond to this action
                    if action_idx.any():  # Ensure there are samples for this action
                        phi = get_phi(data["s"][action_idx])  # Get phi for states with action `act`
                        td_prediction = np.dot(phi, weights[:, act])  # Predict TD values
                        td_error_act = td_target[action_idx] - td_prediction  # Calculate TD error
                        abs_td_error = np.abs(td_error_act).mean()  # Average TD error
                        abs_td_error_history.append(abs_td_error)   # Append TD error
                        gradient = (td_error_act[..., None] * phi).mean(0)  # Calculate gradient
                        weights[:, act] += alpha * gradient  # Gradient descent step

                # 7. If weights didn’t change, break (Convergence Check):
                if np.allclose(weights, weights_before_step, rtol=1e-5, atol=1e-5):
                    break

            # 8. If policy didn’t change, break:
            if np.allclose(weights, weights_before_fit, rtol=1e-5, atol=1e-5):
                break

        # Log progress and TD error
        if tot_steps % log_frequency == 0:
            exp_return = expected_return(env_eval, weights, gamma, episodes_eval)
            mean_td_error = np.mean(abs_td_error_history) if abs_td_error_history else 0
            pbar.set_description(
                f"TDE: {mean_td_error:.3f}, " +
                f"G: {exp_return:.3f}"
            )
            # Ensure that tot_steps does not exceed max_steps before updating
            if tot_steps < max_steps:
                exp_return_history[tot_steps] = exp_return
                td_error_history[tot_steps] = mean_td_error

        pbar.update(ep_steps)

    pbar.close()
    return td_error_history, exp_return_history


# def fqi(seed):
    N = gradient_steps
    K = fitting_iterations
    D = datasize
    S = 4

    data = dict()
    eps = 1.0
    idx_data = 0
    tot_steps = 0
    weights = np.zeros((phi_dummy.shape[1], n_actions))  # Initialize weights
    exp_return = expected_return(env_eval, weights, gamma, episodes_eval)
    exp_return_history = np.zeros((max_steps))
    td_error_history = np.zeros((max_steps))
    pbar = tqdm(total=max_steps)

    while tot_steps < max_steps:
        # 1. For S environment steps (collecting samples):
        step_counter = 0
        while step_counter < S and tot_steps < max_steps:
            # Reset environment for the new episode
            s, _ = env.reset(seed=seed + tot_steps)  # reset environment
            done = False
            ep_steps = 0

            while not done and step_counter < S:
                # Use epsilon-greedy policy to select action
                phi = get_phi(s)  # Compute features for state
                Q = np.dot(phi, weights).ravel()  # Estimate Q-values
                a = eps_greedy_action(Q, eps)  # Select action using epsilon-greedy policy
                
                # 2. Step in the environment with selected action and observe next state
                s_next, r, terminated, truncated, _ = env.step(a)  # Execute action
                done = terminated or truncated  # Check if episode ends
                step_counter += 1
                tot_steps += 1

                # Store the sample: state, action, reward, next state
                if idx_data < D:
                    if idx_data == 0:
                        data["s"] = np.zeros((D, s.size))
                        data["a"] = np.zeros((D,), dtype=int)
                        data["r"] = np.zeros((D,))
                        data["s_next"] = np.zeros((D, s_next.size))
                    # Collect sample
                    data["s"][idx_data] = s
                    data["a"][idx_data] = a
                    data["r"][idx_data] = r
                    data["s_next"][idx_data] = s_next
                    idx_data += 1

                s = s_next  # Move to the next state

                # Decay epsilon after every step
                eps = max(eps - 1.0 / max_steps, 0.5)  # Decaying epsilon to minimum of 0.5

            # Reset the sample collection index for the next round
            if idx_data >= D:
                # 3. After collecting D samples, perform FQI (Fitted Q-Iteration)
                # 4. For K iterations (Policy Improvement):
                for k in range(K):
                    # 5. Fix TD targets:
                    phi_next = get_phi(data["s_next"])
                    td_target = data["r"] + gamma * np.dot(phi_next, weights).max(-1)

                    weights_before_fit = weights.copy()  # Store weights before fitting
                    abs_td_error_history = []

                    # 6. For N gradient steps (Gradient Descent on Q-values):
                    for n in range(N):
                        weights_before_step = weights.copy()  # Store weights before each step

                        # 7. Update weights using gradient descent:
                        for act in range(n_actions):
                            action_idx = data["a"] == act  # Find which samples correspond to this action
                            if action_idx.any():  # Ensure there are samples for this action
                                phi = get_phi(data["s"][action_idx])  # Get phi for states with action `act`
                                td_prediction = np.dot(phi, weights[:, act])  # Predict TD values
                                td_error_act = td_target[action_idx] - td_prediction  # Calculate TD error
                                abs_td_error = np.abs(td_error_act).mean()  # Average TD error
                                abs_td_error_history.append(abs_td_error)   # Append TD error
                                gradient = (td_error_act[..., None] * phi).mean(0)  # Calculate gradient
                                weights[:, act] += alpha * gradient  # Gradient descent step

                        # 8. Break if weights didn’t change between gradient steps (Convergence Check):
                        if np.allclose(weights, weights_before_step, rtol=1e-5, atol=1e-5):
                            break  # Stop the gradient descent if the weights haven't changed significantly

                    # 9. Break if policy (weights) didn’t change after K iterations:
                    if np.allclose(weights, weights_before_fit, rtol=1e-5, atol=1e-5):
                        break  # Stop the fitting iterations if the weights haven't changed significantly

                idx_data = 0  # Reset the sample collection index for the next round

        # 10. Log progress and TD error
        if tot_steps % log_frequency == 0:
            exp_return = expected_return(env_eval, weights, gamma, episodes_eval)
            mean_td_error = np.mean(abs_td_error_history) if abs_td_error_history else 0
            pbar.set_description(
                f"TDE: {mean_td_error:.3f}, " +
                f"G: {exp_return:.3f}"
            )
            if tot_steps < max_steps:
                exp_return_history[tot_steps] = exp_return
                td_error_history[tot_steps] = mean_td_error

        pbar.update(ep_steps)

    pbar.close()
    return td_error_history, exp_return_history



# def fqi(seed):
    N = gradient_steps
    K = fitting_iterations
    D = datasize
    S = 4

    data = dict()
    eps = 1.0
    idx_data = 0
    tot_steps = 0
    weights = np.zeros((phi_dummy.shape[1], n_actions))  # Initialize weights
    exp_return = expected_return(env_eval, weights, gamma, episodes_eval)
    exp_return_history = np.zeros((max_steps))
    td_error_history = np.zeros((max_steps))
    pbar = tqdm(total=max_steps)

    while tot_steps < max_steps:
        # 1. For S environment steps (collecting samples):
        step_counter = 0
        while step_counter < S and tot_steps < max_steps:
            # Reset environment for the new episode
            s, _ = env.reset(seed=seed + tot_steps)  # reset environment
            done = False
            ep_steps = 0

            while not done and step_counter < S:
                # Use epsilon-greedy policy to select action
                phi = get_phi(s)  # Compute features for state
                Q = np.dot(phi, weights).ravel()  # Estimate Q-values
                a = eps_greedy_action(Q, eps)  # Select action using epsilon-greedy policy
                
                # 2. Step in the environment with selected action and observe next state
                s_next, r, terminated, truncated, _ = env.step(a)  # Execute action
                done = terminated or truncated  # Check if episode ends
                step_counter += 1
                tot_steps += 1

                # Store the sample: state, action, reward, next state
                if idx_data < D:
                    if idx_data == 0:
                        data["s"] = np.zeros((D, s.size))
                        data["a"] = np.zeros((D,), dtype=int)
                        data["r"] = np.zeros((D,))
                        data["s_next"] = np.zeros((D, s_next.size))
                    # Collect sample
                    data["s"][idx_data] = s
                    data["a"][idx_data] = a
                    data["r"][idx_data] = r
                    data["s_next"][idx_data] = s_next
                    idx_data += 1

                s = s_next  # Move to the next state

                # Decay epsilon after every step
                eps = max(eps - 1.0 / max_steps, 0.5)  # Decaying epsilon to minimum of 0.5

            # Reset the sample collection index for the next round
            if idx_data >= D:
                # Initialize TD error history for this batch
                abs_td_error_history = []  # Ensures it's always initialized before use

                # 3. After collecting D samples, perform FQI (Fitted Q-Iteration)
                # 4. For K iterations (Policy Improvement):
                for k in range(K):
                    # 5. Fix TD targets:
                    phi_next = get_phi(data["s_next"])
                    td_target = data["r"] + gamma * np.dot(phi_next, weights).max(-1)

                    weights_before_fit = weights.copy()  # Store weights before fitting

                    # 6. For N gradient steps (Gradient Descent on Q-values):
                    for n in range(N):
                        weights_before_step = weights.copy()  # Store weights before each step

                        # 7. Update weights using gradient descent:
                        for act in range(n_actions):
                            action_idx = data["a"] == act  # Find which samples correspond to this action
                            if action_idx.any():  # Ensure there are samples for this action
                                phi = get_phi(data["s"][action_idx])  # Get phi for states with action `act`
                                td_prediction = np.dot(phi, weights[:, act])  # Predict TD values
                                td_error_act = td_target[action_idx] - td_prediction  # Calculate TD error
                                abs_td_error = np.abs(td_error_act).mean()  # Average TD error
                                abs_td_error_history.append(abs_td_error)   # Append TD error
                                gradient = (td_error_act[..., None] * phi).mean(0)  # Calculate gradient
                                weights[:, act] += alpha * gradient  # Gradient descent step

                        # 8. Break if weights didn’t change between gradient steps (Convergence Check):
                        if np.allclose(weights, weights_before_step, rtol=1e-5, atol=1e-5):
                            break  # Stop the gradient descent if the weights haven't changed significantly

                    # 9. Break if policy (weights) didn’t change after K iterations:
                    if np.allclose(weights, weights_before_fit, rtol=1e-5, atol=1e-5):
                        break  # Stop the fitting iterations if the weights haven't changed significantly

                idx_data = 0  # Reset the sample collection index for the next round

        # 10. Log progress and TD error
        if tot_steps % log_frequency == 0:
            exp_return = expected_return(env_eval, weights, gamma, episodes_eval)
            mean_td_error = np.mean(abs_td_error_history) if abs_td_error_history else 0
            pbar.set_description(
                f"TDE: {mean_td_error:.3f}, " +
                f"G: {exp_return:.3f}"
            )
            if tot_steps < max_steps:
                exp_return_history[tot_steps] = exp_return
                td_error_history[tot_steps] = mean_td_error

        pbar.update(ep_steps)

    pbar.close()
    return td_error_history, exp_return_history



env_id = "Gym-Gridworlds/Empty-2x2-v0"
env = gymnasium.make(env_id, coordinate_observation=True, random_action_prob=0.1, reward_noise_std=0.01)
env_eval = gymnasium.make(env_id, coordinate_observation=True, max_episode_steps=10)  # 10 steps only for faster eval
episodes_eval = 10  # max expected return will be 0.994

state_dim = env.observation_space.shape[0]
n_actions = env.action_space.n

# automatically set centers and sigmas
n_centers = [2, 2]
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
gradient_steps_sweep = [1, 100]  #[1000] #[1, 100, 1000]  # N in pseudocode
fitting_iterations_sweep = [1, 100]  #[1000] # [1, 100, 1000]  # K in pseudocode
datasize_sweep = [1, 100]  #[1000] # [1, 100, 1000]  # D in pseudocode
gamma = 0.99
alpha = 0.05
max_steps = 10000
log_frequency = 100
n_seeds = 1

results_ret = np.zeros((
    len(gradient_steps_sweep),
    len(fitting_iterations_sweep),
    len(datasize_sweep),
    n_seeds,
    max_steps,
))
results_tde = np.zeros_like(results_ret)

for i, gradient_steps in enumerate(gradient_steps_sweep):
    for j, fitting_iterations in enumerate(fitting_iterations_sweep):
        for k, datasize in enumerate(datasize_sweep):
            label = f"Grad Steps: {gradient_steps}, " + \
                    f"Fit Iters: {fitting_iterations}, " + \
                    f"Datasize: {datasize}"
            print(label)
            for seed in range(n_seeds):
                td_error, exp_return = fqi(seed)
                results_tde[i, j, k, seed] = td_error
                results_ret[i, j, k, seed] = exp_return

fig, axs = plt.subplots(3*2, 3*3)
for i, N in enumerate(gradient_steps_sweep):
    for j, K in enumerate(fitting_iterations_sweep):
        for k, D in enumerate(datasize_sweep):
            axs[j][k+i*3].plot(results_ret[i, j, k].mean(0), label=f"N:{N}  K:{K}  D:{D}", color="g")
            axs[j][k+i*3].legend(prop={'size': 6}, loc="lower right")
            axs[j][k+i*3].set_ylim([0.0, 1.1])
            axs[j][k+i*3].tick_params(labelsize=6)
            axs[j+3][k+i*3].plot(results_tde[i, j, k].mean(0), label=f"N:{N}  K:{K}  D:{D}", color="b")
            axs[j+3][k+i*3].legend(prop={'size': 6}, loc="lower right")
            axs[j+3][k+i*3].tick_params(labelsize=6)
            if i == 0 and j == 1 and k == 0:
                axs[j][k+i*3].set_ylabel("Return")
                axs[j+3][k+i*3].set_ylabel("TD Error")
            if j == 2:
                axs[j+3][k+i*3].set_xlabel("Steps")

fig.tight_layout()
plt.show()