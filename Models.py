class QModel():
    def __init__(self, n_actions, memory_size=10000, hidden_units=128, demands_type='deterministic'):
        
        
        self.memory_size = memory_size
        self.hidden_units = hidden_units
        
        self.env = build_beer_game(demands_type=demands_type) # build and initialize the environment
        
        self.n_actions = n_actions
        self.policy_net = DuelingQNet(n_actions, hidden_units).to(device)
        self.target_net = DuelingQNet(n_actions, hidden_units).to(device)


        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(memory_size)

        self.steps_done = 0
        self.eps_threshold = EPS_START
        self.episode = 1
        self.episode_history = []
        self.eps_history = []
        
        
        self.sample_qs = []
        self.test_rewards = []
        
        self.best_cum_rewards = -999999
        self.best_actions = None
        self.latest_actions = None
        
    def train(self, num_episodes=1000, target_update=10, batch_size=128, seed=None):
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            
        for i_episode in tqdm(range(self.episode, self.episode+num_episodes), leave=False):
            # Initialize the environment and state

            state = self.env.reset()
            state = torch.tensor(list(state.values())).float().to(device) # convert state dict to list of numbers

            cum_rewards = 0
            loss = 0
            self.latest_actions = []
            
            for t in count():
                # Select and perform an action
                action = self.select_action(state)
                
                self.latest_actions.append(action.item()*2)
                
                next_state, reward, done = self.env.step(action.item()*2)
                reward = torch.tensor([reward], device=device)

                cum_rewards += reward

                # Observe new state
                if not done:
                    next_state = torch.tensor(list(next_state.values())).float().to(device) # convert state dict to list of numbers

                else:
                    next_state = None

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the target network)
                current_loss = self.optimize_model(t, batch_size=batch_size)
                if current_loss is None:
                    current_loss = 0
                    
                loss += current_loss
                
                
                if done:
                    break
            
        
            loss = loss/t

            # Update the target network, copying all weights and biases in DQN
            if i_episode % target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                
                
            if cum_rewards.item() >= self.best_cum_rewards:
                self.best_cum_rewards = cum_rewards.item()
                self.best_actions = self.latest_actions

            if self.episode % 100 ==0:
                qs = self.policy_net(sample_states_4)
                self.sample_qs.append([self.episode, qs[0,0].item(), qs[1,0].item(), qs[2,4].item(), qs[3,4].item()])
                
                actions, rewards = self.test()
                self.test_rewards.append([self.episode, rewards])
                
            self.episode_history.append([seed, i_episode, self.eps_threshold, batch_size, self.memory_size, self.hidden_units, cum_rewards.item(), loss])
            self.episode += 1
        
            
        return self.sample_qs
        
        

    def select_action(self, state, test=False):
        
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        self.eps_threshold = eps_threshold


        if test or (sample > eps_threshold):
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.

                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=device, dtype=torch.long)
        
    def optimize_model(self, t, batch_size=128):

        if len(self.memory) < batch_size:
            return
        transitions = self.memory.sample(batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=device, dtype=torch.bool)

        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        
        
        
        next_state_values = torch.zeros(batch_size, device=device)
#         next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        
        
        actions = self.policy_net(non_final_next_states).max(1)[1].detach()
        q_values = self.target_net(non_final_next_states).detach()        
        next_state_values[non_final_mask] = q_values.gather(1, actions.unsqueeze(1)).squeeze()

        
        
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch


        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-5, 5)
        self.optimizer.step()
        
        return loss.item()
    
    
    def test(self):

        state = self.env.reset()
        state = torch.tensor(list(state.values()), device=device).float()

        actions = []
        rewards = 0 
        for t in count():

            quantity = self.select_action(state, test=True).item()*2
            state, reward, done = self.env.step(quantity)
            state = torch.tensor(list(state.values()), device=device).float()

            actions.append(quantity)
            rewards += reward

            if done:
                break

        return actions, rewards