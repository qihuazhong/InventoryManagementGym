import numpy as np
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    

class SumTree(object):
    """
    This SumTree code is originally from 
    Jaromiru: 
    https://github.com/jaromiru/AI-blog/blob/master/SumTree.py
    
    Modified by:
    1. Morvan Zhou: 
    https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py
    
    2. simoninithomas:
    https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Dueling%20Double%20DQN%20with%20PER%20and%20fixed-q%20targets/Dueling%20Deep%20Q%20Learning%20with%20Doom%20(%2B%20double%20DQNs%20and%20Prioritized%20Experience%20Replay).ipynb
    
    3. Qihua Zhong
    - Changed updating operations to explicit sums to avoid float-point errors accumulation.
    - Use numpy vector operations for batch priority updates instead of for loops (about 8x faster when batch size is 128 and memory size is 2^17).
    - Use numpy vector operations for batch sampling instead of for loops (about 10x faster when batch size is 128 and memory size is 2^17)
    
    
    """
    
    def __init__(self, capacity):
        
        self.position = 0
        self.capacity = capacity
        
        # parent nodes = capacity - 1
        # leaf nodes = capacity
        self.tree = np.zeros(2 * capacity -1)
        self.data = np.zeros(capacity, dtype=object)
        
        
    def push(self, priority, *experience):
        
        tree_index = self.position + self.capacity - 1
        self.data[self.position] = Transition(*experience)
        self.update(tree_index, priority)
        
        self.position += 1
        
        if self.position >= self.capacity:
            self.position = 0
            
            
            
    def update(self, tree_index, priority):
        '''
        The previoius implementations were prone to accumulating float-point operation errors.
        The accumulated errors caused discrepencies between parents and their children's sums, therefore
        a small probability of sampling from an unfilled position in the memory.
        
        Explicitly summing up the two children at each level instead of 
        adding the difference (obtained at the bottom level) solved the issue.
        '''

        self.tree[tree_index] = priority
        
        while tree_index != 0:
            
            tree_index = (tree_index - 1) // 2
            left_child_index = 2 * tree_index + 1
            right_child_index = left_child_index + 1
            
            self.tree[tree_index] = self.tree[left_child_index] + self.tree[right_child_index]
            
            
            
    def batch_update(self, tree_idx, ps):

        
        self.tree[tree_idx] = ps
        
        tree_idx = np.array(tree_idx)
        
        while tree_idx.sum() != 0:
            tree_idx = np.unique((tree_idx - 1) // 2)
            
            left_child_index = 2 * tree_idx + 1
            right_child_index = left_child_index + 1
            self.tree[tree_idx] = self.tree[left_child_index] + self.tree[right_child_index]
            
        
        
    def get_leaf(self, v):
        
        parent_index = 0
        
        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1
            
            # If we reach bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else: # downward search, always search for a higher priority node
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index
        
        data_index = leaf_index - self.capacity + 1
        
        return leaf_index, self.tree[leaf_index], self.data[data_index]
    
    
    def get_batch_leaves(self, values):
        parent_index = np.zeros(shape = (len(values)), dtype=np.int32)
        
        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1
            
            # If we reach bottom, end the search
            if (left_child_index >= len(self.tree)).all():
                leaf_index = parent_index
                break
            else: # downward search, always search for a higher priority node
                
                left_mask = values <= self.tree[left_child_index]
                right_mask = ~left_mask
                
                
                parent_index[left_mask] = left_child_index[left_mask]
                
                values[right_mask] -= self.tree[left_child_index][right_mask]
                parent_index[right_mask] = right_child_index[right_mask]
                
        data_index = leaf_index - self.capacity + 1
        
        return leaf_index, self.tree[leaf_index], self.data[data_index]
    
    @property
    def total_priority(self):
        return self.tree[0]
    
    
class PERMemory(object):
    
    PER_e = 0.01  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
    PER_a = 0.8  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
    PER_b = 0.4  # importance-sampling, from initial value increasing to 1
    
    min_priority = np.power(PER_e, PER_a)
    
    PER_b_increment_per_sampling = 0.0000005
    
    absolute_error_upper = 10000.  # clipped abs error
    
    
    def __init__(self, capacity):
        
        self.capacity = capacity
        self.tree = SumTree(capacity)
        self.filled_length = 0
        
    def push(self, *experience):
        
        # Find the max priority
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        if max_priority == 0:
            max_priority = self.absolute_error_upper
        
        self.tree.push(max_priority, *experience)
        
        if self.filled_length < self.capacity:
            self.filled_length += 1
        
        
    def sample_old(self, batch_size):
        
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])
        
        # Create a sample array that will contains the minibatch
        memory_batch = []
        batch_idx, batch_ISWeights = np.empty((batch_size,), dtype=np.int32), np.empty((batch_size), dtype=np.float32)
    
        # Calculate the priority segment
        priority_segment = self.tree.total_priority / batch_size
        
        
        # Calculate the max weight
        p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority

        if p_min == 0:
            max_weight = 1000.
        else:
            max_weight = (p_min * batch_size) ** (-self.PER_b)
        
        for i in range(batch_size):
            
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)
            
            # Retrieve the experience that corresponds to each value
            index, priority, data = self.tree.get_leaf(value)
            
            # P(j)
            sampling_probabilities = max(self.min_priority, priority) / self.tree.total_priority
            
            # IS = (1/batch_size * 1/P(i)**b /max wi == batch_size*P(i)**-b /max wi)
            batch_ISWeights[i] = np.power(batch_size * sampling_probabilities, -self.PER_b)/max_weight
            
            batch_idx[i] = index
            experience = data
            memory_batch.append(experience)
            
        return batch_idx, memory_batch, batch_ISWeights
    
    
    def sample(self, batch_size):
        
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])
        

        p_min = np.min(self.tree.tree[-self.tree.capacity:-self.tree.capacity+self.filled_length-1]) / self.tree.total_priority

        max_weight = (p_min * batch_size) ** (-self.PER_b)
                
        values = np.random.uniform(self.tree.total_priority, size=batch_size)
        
        batch_idx, priority, data = self.tree.get_batch_leaves(values)

        # P(j)
        sampling_probabilities = np.maximum(self.min_priority, priority) / self.tree.total_priority
        batch_ISWeights = np.power(batch_size * sampling_probabilities, -self.PER_b)/max_weight
        
        return batch_idx, data.tolist(), batch_ISWeights.astype(np.float32)
        
        

    
    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.PER_e
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        
        ps = np.power(clipped_errors, self.PER_a).squeeze()
        
        self.tree.batch_update(tree_idx, ps)
        

    def batch_update_old(self, tree_idx, abs_errors):
        abs_errors += self.PER_e
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        
        ps = np.power(clipped_errors, self.PER_a)

        
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)
        
        
    def __len__(self):
        return self.filled_length
    