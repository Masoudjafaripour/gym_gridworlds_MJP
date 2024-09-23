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
