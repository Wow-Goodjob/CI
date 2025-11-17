import torch
import numpy as np
from collections import deque
import sys
from algo.sum_tree import SumTree
from net import ReplayBuffer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, args, net_config=None):
        super(PrioritizedReplayBuffer, self).__init__(args, net_config=net_config)
        self.alpha = 0.6
        self.beta_init = 0.4
        self.beta = 0.4
        self.batch_size = net_config["BATCH_SIZE"]
        self.buffer_capacity = net_config["BUFFER_SIZE"]
        self.sum_tree = SumTree(net_config["BUFFER_SIZE"])
        self.current_size = 0
        self.count = 0

    def write(self, state, action, reward, next_state, done, q=None):
        index = self.cnt % self.buffer_size
        self.memory_buf[index] = self.memory(state, action, reward, next_state, done)

        priority = 1.0 if self.cnt == 0 else self.sum_tree.priority_max
        self.sum_tree.update(data_index=self.count, priority=priority)  # 更新当前经验在sum_tree中的优先级
        self.count = (self.count + 1) % self.batch_size  # When the 'count' reaches buffer_capacity, it will be reset to 0.
        self.current_size = min(self.cnt + 1, self.buffer_capacity)
        self.cnt += 1

    def sample(self):
        batch_index, IS_weight = self.sum_tree.get_batch_index(current_size=self.current_size, batch_size=self.batch_size, beta=self.beta)
        self.beta = self.beta_init + (1 - self.beta_init) * (min(self.cnt, self.buffer_capacity * 2) / self.buffer_capacity * 2)  # beta：beta_init->1.0
        experiences = [self.memory_buf[i] for i in batch_index]
        if self.args.agent_type in {"CCGN"}:  # S
            states = torch.tensor(np.array([e.state for e in experiences if e is not None])).float().to(device)
        else:
            states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)

        if self.args.agent_type == "DQN":  # A
            actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        elif self.args.agent_type in {"CCGN"}:
            actions = torch.tensor(np.array([e.action for e in experiences if e is not None])).long().to(device)
        else:
            actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)

        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)  # R

        if self.args.agent_type in {"CCGN"}:  # S'
            next_states = torch.tensor(np.array([e.next_state for e in experiences if e is not None])).float().to(device)
        else:
            next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
                device)

        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)  # D

        return states, actions, rewards, next_states, dones, batch_index, IS_weight

    def update_batch_priorities(self, batch_index, td_errors):  # 根据传入的td_error，更新batch_index所对应数据的priorities
        priorities = (np.abs(td_errors) + 0.01) ** self.alpha
        for index, priority in zip(batch_index, priorities):
            self.sum_tree.update(data_index=index, priority=priority)
