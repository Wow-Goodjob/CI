import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        state = torch.from_numpy(state).float().to(device)
        probs = self.actor(state)
        dist = Categorical(probs)
        action = dist.sample()

        return action.item(), dist.log_prob(action), self.critic(state)

    def evaluate(self, state, action):
        state = state.to(device)
        action = action.to(device)

        probs = self.actor(state)
        dist = Categorical(probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=0.0003, gamma=0.99, K_epochs=4, eps_clip=0.2):
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.mse_loss = nn.MSELoss()

        self.buffer = {'states': [], 'actions': [], 'logprobs': [], 'rewards': [], 'state_values': [],
                       'is_terminals': []}

    def select_action(self, state):
        with torch.no_grad():
            state = np.array(state)
            action, log_prob, _ = self.policy_old.act(state)

        return action, log_prob.cpu().detach().numpy()

    def store_transition(self, state, action, log_prob, reward, done):
        self.buffer['states'].append(state)
        self.buffer['actions'].append(action)
        self.buffer['logprobs'].append(log_prob)
        self.buffer['rewards'].append(reward)
        self.buffer['is_terminals'].append(done)

    def update(self):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer['rewards']), reversed(self.buffer['is_terminals'])):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        old_states = torch.tensor(np.array(self.buffer['states']), dtype=torch.float32).to(device)
        old_actions = torch.tensor(np.array(self.buffer['actions']), dtype=torch.float32).to(device)
        old_logprobs = torch.tensor(np.array(self.buffer['logprobs']), dtype=torch.float32).to(device)

        batch_size = 128
        dataset_size = len(old_states)

        for _ in range(self.K_epochs):

            indices = torch.randperm(dataset_size).to(device)

            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]

                batch_states = old_states[batch_indices]
                batch_actions = old_actions[batch_indices]
                batch_logprobs = old_logprobs[batch_indices]
                batch_rewards = rewards[batch_indices]

                logprobs, state_values, dist_entropy = self.policy.evaluate(batch_states, batch_actions)
                state_values = torch.squeeze(state_values)

                ratios = torch.exp(logprobs - batch_logprobs)

                advantages = batch_rewards - state_values.detach()

                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

                loss = -torch.min(surr1, surr2) + 0.5 * self.mse_loss(state_values, batch_rewards) - 0.01 * dist_entropy

                self.optimizer.zero_grad()
                loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.clear_buffer()

    def clear_buffer(self):
        self.buffer = {'states': [], 'actions': [], 'logprobs': [], 'rewards': [], 'state_values': [],
                       'is_terminals': []}
    def save(self,path):
        directory = os.path.dirname(path)


import traci


class TrafficLightManager:
    def __init__(self, tls_ids):
        self.tls_ids = tls_ids  # 16个路口的 ID 列表
        self.phase_map = {0: 0, 1: 2, 2: 4, 3: 6}  # 简单的 Action 到 Phase 映射示例
        self.lane_to_tls={}
        self.edge_to_tls = {}
        self.build_lane_map()

    def get_state(self, tls_id):
        lanes = traci.trafficlight.getControlledLanes(tls_id)
        queues = []
        for lane in lanes:
            number=traci.lane.getLastStepHaltingNumber(lane)
            queues.append(min(number/25,1.0))
            #queues.append(traci.lane.getLastStepOccupancy(lane))
        current_phase=traci.trafficlight.getPhase(tls_id)
        phase_one_shot=[0]*8
        phase_one_shot[current_phase]=1
        state=np.array(queues+phase_one_shot)
        return np.array(state)

    def build_lane_map(self):
        for tls_id in self.tls_ids:
            lanes = traci.trafficlight.getControlledLanes(tls_id)
            unique_lanes=list(set(lanes))
            for lane in unique_lanes:
                self.lane_to_tls[lane]=tls_id
                edge=traci.lane.getEdgeID(lane)
                self.edge_to_tls[edge]=tls_id

    def get_reward(self, tls_id):
        lanes = traci.trafficlight.getControlledLanes(tls_id)
        queue_length=sum([traci.lane.getLastStepHaltingNumber(lane) for lane in lanes])
        #waiting_time = sum([traci.lane.getWaitingTime(lane) for lane in lanes])
        #print(f"Waiting time: {waiting_time}")
        #return -waiting_time
        return -queue_length/50

    def apply_action(self, tls_id, action_idx):
        current_phase = traci.trafficlight.getPhase(tls_id)
        target_phase = self.phase_map.get(action_idx, 0)
        if current_phase != target_phase:
            traci.trafficlight.setPhase(tls_id, target_phase)

    def set_phase(self,tls_id, phase):
        traci.trafficlight.setPhase(tls_id, phase)

    def get_yellow_phase(self,current_phase):
        return current_phase//2*2+1