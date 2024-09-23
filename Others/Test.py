def monte_carlo(env, Q, gamma, eps_decay, max_steps, epsilon, episodes_per_iteration,seed = 0, use_is = False):    
    policy = np.ones((n_states, n_actions))*(1/n_actions)
    #we simply call the target policy, policy. 
    behavior_policy = copy.deepcopy(policy)
    be = [] 
    #let's keep track of the steps:
    steps = 0    
    #Calculate first bellman error
    Q_true = bellman_q(policy, gamma)
    current_bellman_error = np.sum(np.abs(Q_true - Q))
    
    target_epsilon = 0.01
    while steps <  max_steps: 
        #Here we keep the n iterations of episodes to calculate returns over: 
        c = np.zeros((n_states,n_actions))

        if use_is:
            behavior_policy = update_soft_policy(behavior_policy, Q, epsilon)#if using importance sampling

        for _ in range(episodes_per_iteration):
            data = episode(env, Q, epsilon, seed) #we use the Behavior policy to generate the episodes, by
            #using epsilon

            states = data['s']
            actions = data['a']
            rewards = data['r']

            #the importance sampling for episode, state, action pair
            episode_steps = len(states) #count the steps taken in last episode
            steps += episode_steps #Keep track of steps

            #decay epsilon after every episode(if not importance sampling)
            epsilon = max(epsilon - eps_decay / max_steps * episode_steps, 0.01)
            
            _return = 0
            _weight = 1

            for time_step in range(episode_steps-1,-1,-1): #The loop is backward
                
                    #Update the correponding state and action:
                action = actions[time_step]
                state = states[time_step]
                reward = rewards[time_step]
                   
                _return = gamma*_return + reward
                
                c[state][action] = c[state][action] + _weight
                Q[state][action] = Q[state][action] + (_weight/c[state][action])*(_return-Q[state][action])

                #update the policy 
                if use_is:
                    _weight = _weight * (policy[state][action]/behavior_policy[state][action])
                be.append(current_bellman_error)

        update_soft_policy(policy, Q, target_epsilon)
        #update_soft_policy(behavior_policy, Q, epsilon)
         #Here we use the target epsilon for the target policy

        #Now, for the time steps that
        Q_true = bellman_q(policy, gamma)
        current_bellman_error = np.sum(np.abs(Q_true - Q)) #keep the current bellman error
            #be.append(current_bellman_error)
        
                #log the bellman error for each time step. (If the policy is not changing, we should log the same error?)
    return Q, be, policy






# Monte Carlo Control with Off-policy E-soft Policy
# def off_policy_mc_control(env, Q, gamma, eps_decay, max_steps, episodes_per_iteration):
#     eps = 1.0
#     total_steps = 0
#     bellman_errors = []
#     C = np.zeros_like(Q)  # To store cumulative weights for each (state, action)
#     pi = eps_greedy_probs(Q, eps)  # Greedy policy
#     Q_true = bellman_q(pi, gamma)
#     bellman_error = np.abs(Q - Q_true).sum()
#     bellman_errors.append(bellman_error)
    
#     while total_steps < max_steps:
#         for _ in range(episodes_per_iteration):
#             # Generate an episode using a soft behavior policy `b`
#             episode_data = episode(env, Q, eps, int(seed))  # Assume `episode()` returns states, actions, rewards
#             G = 0
#             W = 1

#             # Loop over episode steps in reverse
#             for t in range(len(episode_data["s"]) - 1, -1, -1):
#                 state, action, reward = episode_data["s"][t], episode_data["a"][t], episode_data["r"][t]
#                 G = gamma * G + reward

#                 # Update cumulative weight
#                 C[state][action] += W
                
#                 # Update Q-value with weighted importance sampling
#                 Q[state][action] += W / C[state][action] * (G - Q[state][action])
                
#                 # Update the greedy policy `pi`
#                 pi[state] = np.argmax(Q[state])  # Greedy action for current state

#                 # If action taken is not the greedy action, stop updating
#                 if action != pi[state]:
#                     break
                
#                 # Update importance sampling weight
#                 W *= 1.0 / eps_greedy_probs(Q, eps)[action][state]

#             # Decay epsilon after each episode (soft behavior policy)
#             eps = max(eps - eps_decay * len(episode_data["s"]) / max_steps, 0.01)
#             total_steps += len(episode_data["s"])

#         # Update the epsilon-greedy policy progressively
#         pi = eps_greedy_probs(Q, eps)
#         Q_true = bellman_q(pi, gamma)
#         bellman_error = np.abs(Q - Q_true).sum()
#         bellman_errors.append(bellman_error)

#     # Ensure bellman_errors has the same length as max_steps
#     if len(bellman_errors) > max_steps:
#         bellman_errors = bellman_errors[:max_steps]  # Trim to match max_steps
#     else:
#         # Pad with the last bellman error if the length is shorter
#         bellman_errors += [bellman_errors[-1]] * (max_steps - len(bellman_errors))

#     return Q, bellman_errors

# # Monte Carlo Control with Off-policy E-soft Policy
# def off_policy_mc_control(env, Q, gamma, eps_decay, max_steps, episodes_per_iteration):
#     eps = 1.0
#     total_steps = 0
#     bellman_errors = []
#     C = np.zeros_like(Q)  # To store cumulative weights for each (state, action)
#     pi = eps_greedy_probs(Q, eps)  # Greedy policy
#     Q_true = bellman_q(pi, gamma)
#     bellman_error = np.abs(Q - Q_true).sum()
#     bellman_errors.append(bellman_error)
    
#     while total_steps < max_steps:
#         for _ in range(episodes_per_iteration):
#             # Generate an episode using a soft behavior policy `b`
#             episode_data = episode(env, Q, eps, int(seed))
#             G = 0
#             W = 1

#             # Loop over episode steps in reverse
#             for t in range(len(episode_data["s"]) - 1, -1, -1):
#                 state, action, reward = episode_data["s"][t], episode_data["a"][t], episode_data["r"][t]
#                 G = gamma * G + reward

#                 # Update cumulative weight
#                 C[state][action] += W
                
#                 # Update Q-value with weighted importance sampling
#                 Q[state][action] += W / C[state][action] * (G - Q[state][action])
                
#                 # Update the greedy policy `pi`
#                 best_action = np.argmax(Q[state])  # Get the best action for the current state
#                 pi[state] = best_action  # Update the policy
                
#                 # If action taken is not the greedy action, stop updating
#                 if action != best_action:  # Compare with the best action
#                     break
                
#                 # Update importance sampling weight
#                 W *= 1.0 / eps_greedy_probs(Q, eps)[state][action]

#             # Decay epsilon after each episode (soft behavior policy)
#             eps = max(eps - eps_decay * len(episode_data["s"]) / max_steps, 0.01)
#             total_steps += len(episode_data["s"])

#         # Update the epsilon-greedy policy progressively
#         pi = eps_greedy_probs(Q, eps)
#         Q_true = bellman_q(pi, gamma)
#         bellman_error = np.abs(Q - Q_true).sum()
#         bellman_errors.append(bellman_error)

#     # Ensure bellman_errors has the same length as max_steps
#     if len(bellman_errors) > max_steps:
#         bellman_errors = bellman_errors[:max_steps]  # Trim to match max_steps
#     else:
#         # Pad with the last bellman error if the length is shorter
#         bellman_errors += [bellman_errors[-1]] * (max_steps - len(bellman_errors))

#     return Q, bellman_errors