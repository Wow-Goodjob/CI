import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import random

import net
import config
from agent import Agent
from algo.self_org_agent import DDQNNet
from algo.iql_b_agent import IQLNet
from algo.astar_dqn import ADQNNet
from algo.adaptive_dqn import AdaptiveQNet
from algo.nav_agent import NavNet
from algo.dso_agent import DSOQNet
# from torch.utils.tensorboard import SummaryWriter
import os
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(1)


# model = 'DQN'
# tm = str(time.strftime('%Y-%m-%d-%H-%M', time.localtime()))
# os.makedirs(f'./log/{model}/{tm}', exist_ok=True)
# writer = SummaryWriter(f'./log/{model}/{tm}')


class DQNAgent(Agent):
    def __init__(self, state_dim, action_dim, args, tl_id, net_type, lr=1e-3, n_steps=1, execution=False,
                 net_config=config.RL):
        super(DQNAgent, self).__init__(state_dim, action_dim, args, tl_id, net_type, n_steps=1, execution=execution,
                                       net_config=net_config)

        lr = net_config["Q_LR"]
        self.update = net_config["UPDATE"]
        self.gamma = net_config["GAMMA"]
        self.config = net_config
        if args.algo == "self_org":
            self.q_network = DDQNNet(state_dim, action_dim, args, None, tl_id).to(device)
            self.target_q_network = DDQNNet(state_dim, action_dim, args, None, tl_id).to(device)
        elif args.algo == "iql_b":
            self.q_network = IQLNet(state_dim, action_dim, args, None, tl_id).to(device)
            self.target_q_network = IQLNet(state_dim, action_dim, args, None, tl_id).to(device)
        elif args.algo == "astar_dqn":
            self.q_network = ADQNNet(state_dim, action_dim, args, None, tl_id).to(device)
            self.target_q_network = ADQNNet(state_dim, action_dim, args, None, tl_id).to(device)
        elif args.algo == "adaptive":
            self.q_network = AdaptiveQNet(state_dim, action_dim, args, None, tl_id).to(device)
            self.target_q_network = AdaptiveQNet(state_dim, action_dim, args, None, tl_id).to(device)
        elif args.algo == "nav":
            self.q_network = NavNet(state_dim, action_dim, args, None, tl_id).to(device)
            self.target_q_network = NavNet(state_dim, action_dim, args, None, tl_id).to(device)
        elif args.algo == "dso":
            self.q_network = DSOQNet(state_dim, action_dim, args, None, tl_id).to(device)
            self.target_q_network = DSOQNet(state_dim, action_dim, args, None, tl_id).to(device)
        else:
            self.q_network = net.DQN(state_dim, action_dim, args, None, tl_id).to(device)
            self.target_q_network = net.DQN(state_dim, action_dim, args, None, tl_id).to(device)
        # self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.optimizer = torch.optim.RMSprop(self.q_network.parameters(),
                                             lr=lr, alpha=0.99, eps=1e-5)
        if args.algo == "adaptive":
            self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)

        if net_config["DECAY"]:
            from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=args.episode, eta_min=1e-5)
            # self.scheduler = ExponentialLR(self.optimizer, gamma=0.97)

        self.MSELoss = nn.MSELoss()
        self.buf_cnt = 0
        self.buf_sz = net_config["BUFFER_SIZE"]
        self.n_steps = n_steps
        self.replay = args.replay
        self.batch_size = net_config["BATCH_SIZE"]

        self.tau = args.tau

        self.next_action = 0
        self.action = 0

        self.writer_step = 0

    def store(self, state, reward, next_state, done, q=None, actions=None):
        self.buf_cnt += 1
        if self.args.algo == "nav":
            target = self.q_network(Variable(torch.FloatTensor(state).unsqueeze(dim=0))).data
            old_val = target[0][actions]
            target_val = self.target_q_network(Variable(torch.FloatTensor(next_state).unsqueeze(dim=0))).data
            if done:
                target[0][actions] = reward
            else:
                target[0][actions] = reward + self.gamma * torch.max(target_val)

            error = abs(old_val - target[0][actions])

            self.buf.add(error, (state, actions, reward, next_state, done))
        else:
            # self.pfrl_buf.append(state, self.action, reward, next_state=next_state, is_state_terminal=done)
            if actions is not None:
                self.buf.write(state, actions, reward, next_state, done, q)
            else:
                self.buf.write(state, self.action, reward, next_state, done, q)
                self.action = self.next_action

    def act(self, state, qnet=None, avail_actions=None):
        epsilon = self.config["EPSILON"]
        state = torch.from_numpy(state).float().to(device)
        state = state.unsqueeze(0)
        if avail_actions is not None:
            avail_actions = torch.tensor(avail_actions, dtype=torch.float32)

        self.q_network.eval()
        with torch.no_grad():
            if self.buf_sz == 0 or np.random.uniform() >= epsilon:
                actions_value = self.q_network(state).cpu().squeeze(dim=0)
                min_a = torch.min(actions_value)
                max_a = torch.max(actions_value)
                actions_value = (actions_value - min_a) / (max_a - min_a)
                if avail_actions is not None:
                    actions_value[avail_actions == 0] = -1
                a = int(actions_value.argmax())
            else:
                rand_nums = [random.random() for _ in range(self.action_dim)]
                total = sum(rand_nums)
                actions_value = torch.tensor([num / total for num in rand_nums], dtype=torch.float32)
                if avail_actions is not None:
                    actions_value[avail_actions == 0] = -1
                a = int(actions_value.argmax())
                if self.args.algo == "nav":
                    tau = 1
                    avail_actions = (torch.mean(avail_actions) - avail_actions) / torch.var(avail_actions) / tau
                    avail_actions = torch.exp(avail_actions) / torch.sum(torch.exp(avail_actions))
                    a = int(torch.multinomial(avail_actions, 1))
        self.q_network.train()
        self.next_action = a
        return a

    def learn(self):
        if self.buf_cnt == self.buf_sz - 1:
            print("***************************** Training! **********************************", self.tl_id)
        if self.buf_cnt < self.args.threshold:
            return False
        self.training = True
        # sample from replay buffer
        if self.args.algo == "nav":
            mini_batch, idxs, is_weights = self.buf.sample(self.batch_size)
            mini_batch = np.array(mini_batch, dtype=object).transpose()
            #states = np.vstack(mini_batch[0])
            states=mini_batch[0]
            actions = mini_batch[1]
            rewards = mini_batch[2]
            #new_states = np.vstack(mini_batch[3])
            new_states=mini_batch[3]
            dones = mini_batch[4]
            states=np.vstack(states)
            new_states=np.vstack(new_states)
            #states=np.array(states,dtype=np.float32)
            #actions = np.array(actions, dtype=np.float64)
            #rewards = np.array(rewards, dtype=np.float32)
            #new_states = np.array(new_states, dtype=np.float32)

            dones=np.array([1.0 if done else 0.0 for done in dones],dtype=np.float32)

            states = torch.from_numpy(states).float().to(device)
            actions = torch.from_numpy(actions.astype(np.int64)).unsqueeze(dim=-1).to(device)
            rewards = torch.from_numpy(rewards.astype(np.float32)).float().unsqueeze(dim=-1).to(device)  # R
            new_states = torch.from_numpy(new_states).float().to(device)
            dones = torch.from_numpy(dones.astype(np.int64)).float().unsqueeze(dim=-1).to(device)  # D
        else:
            states, actions, rewards, new_states, dones = self.buf.sample()
        # memory = self.pfrl_buf.sample(self.args.batch_size)
        # states = torch.from_numpy(np.array([elem[0]["state"] for elem in memory])).float().to(device)
        # actions = torch.from_numpy(np.array([elem[0]["action"] for elem in memory])).unsqueeze(dim=1).long().to(device)
        # rewards = torch.from_numpy(np.array([elem[0]["reward"] for elem in memory])).unsqueeze(dim=1).float().to(device)
        # new_states = torch.from_numpy(np.array([elem[0]["next_state"] for elem in memory])).float().to(device)
        # dones = torch.from_numpy(np.array([1 if elem[0]["is_state_terminal"] else 0 for elem in memory])).unsqueeze(dim=1).float().to(device)

        # current q, return related value according to current state and action
        actions = actions.long()
        q = self.q_network(states, bl=True).gather(1, actions)

        # target q, return all value according to next state
        q_next = self.target_q_network(new_states, bl=True).detach()

        # y = r + gamma * target_q, select action-value pair in target q which maximize the q
        y = rewards + (1 - dones) * self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
        # MSE loss
        if self.args.algo == "nav":
            errors = torch.abs(q - y).data.numpy()
            for i in range(self.batch_size):
                idx = idxs[i]
                self.buf.update(idx, errors[i])
            loss = (torch.FloatTensor(is_weights) * F.mse_loss(q, y)).mean()
        else:
            loss = self.MSELoss(q, y)

        # writer.add_scalar(f'dqn_loss/{self.tl_id}', loss, self.writer_step)
        # self.writer_step += 1

        # # if prioritized replay, return IS-weight from sample
        # if self.replay == "PER":
        #     weights = torch.from_numpy(np.array([elem[0]["weight"] for elem in memory])).float().to(device)
        #     loss = (weights * loss).mean()
        #
        #     # TD-Errors, update sum-tree in prioritized replay buffer
        #     self.pfrl_buf.update_errors(y - q)

        # back propagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the target q network
        if self.buf_cnt % self.update == 0:  # update timely
            self.soft_update(self.q_network, self.target_q_network, tau=self.tau)

        return loss.item()

    def update_w(self, s_series, s_index, state):
        if self.args.temporal != "":
            return (s_index + 1) % len(s_series), True
        if len(s_series) < self.args.history:
            s_series.append(state)
            return (s_index + 1) % len(s_series), False
        # GRU embedded GAT
        s_series[s_index] = state
        time_series = state
        for i in range(len(s_series) - 1):
            time_series = np.concatenate((time_series, s_series[((s_index + i + 1) % len(s_series))]), axis=0)
        self.q_network.update_w(time_series)
        if self.buf_cnt % self.update == 0:
            self.target_q_network.update_w(time_series)
        return (s_index + 1) % len(s_series), True

    def force_switch(self):
        self.action = (self.action + 1) % 4
        return self.action

    def soft_update(self, network, target_network, tau):
        if self.args.algo == "nav":
            tau = 1.0
            for target_param, local_param in zip(target_network.parameters(), network.parameters()):
                target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
        else:
            for target_param, local_param in zip(target_network.parameters(), network.parameters()):
                target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def load_model(self, name):
        # print(torch.load('./model/' + name + '/q_net'
        #   + self.tl_id[4] + '.pt'))
        self.q_network.load_state_dict(torch.load('./model/' + name + '/q_net'
                                                  + self.tl_id + '.pt'))
        self.target_q_network.load_state_dict(torch.load('./model/' + name + '/target_q_net'
                                                         + self.tl_id + '.pt'))
        self.optimizer.load_state_dict(torch.load('./model/' + name + '/opt'
                                                  + self.tl_id + '.pt'))

    def save_model(self, name):
        torch.save(self.q_network.state_dict(), './model/' + name + '/q_net'
                   + self.tl_id + '.pt')
        torch.save(self.target_q_network.state_dict(), './model/' + name + '/target_q_net'
                   + self.tl_id + '.pt')
        torch.save(self.optimizer.state_dict(), './model/' + name + '/opt'
                   + self.tl_id + '.pt')

    def load_buffer(self, name):
        with open(f'./model/{name}/buffer.pkl', 'wb') as f:
            pickle.dump(self.buf, f)
        # pass
        # self.pfrl_buf.load('./model/' + name + '/replay_buffer' + self.tl_id[4])
        # self.buf_cnt = len(self.pfrl_buf)

    def save_buffer(self, name):
        with open(f'./model/{name}/buffer.pkl', 'rb') as f:
            self.buf = pickle.load(f)
        # pass
        # self.pfrl_buf.save('./model/' + name + '/replay_buffer' + self.tl_id[4])


class CDQNAgent(DQNAgent):
    def __init__(self, state_dim, action_dim, args, tl_id, net_type, lr=1e-3, n_steps=1, execution=False,
                 net_config=config.RL):
        super(CDQNAgent, self).__init__(state_dim, action_dim, args, tl_id, net_type, n_steps=1, execution=execution,
                                        net_config=net_config)
        self.forward_scale = args.forward_scale  # 前向预测模型损失函数的比例为 0.8
        self.inverse_scale = args.inverse_scale  # 反向预测模型损失函数的比例为 0.2
        self.Qloss_scale = args.Qloss_scale  # Q 值损失函数比例为1
        self.intrinsic_scale = args.intrinsic_scale  # 内在奖励的比例为 1
        self.use_extrinsic = args.use_extrinsic  # 是否使用外在奖励, 如果为 False , 模型只接受来自 ICM 的损失

        self.ICM = net.ICM(state_dim, action_dim, args, None, tl_id).to(device)
        self.optimizer = torch.optim.Adam(list(self.q_network.parameters()) + list(self.ICM.parameters()), lr=lr)

    def learn(self):
        if self.buf_cnt == self.buf_sz - 1:
            print("***************************** Training! **********************************", self.tl_id)
        if self.buf_cnt < self.buf_sz:
            return False
        self.training = True
        # sample from replay buffer
        states, actions, rewards, new_states, dones = self.buf.sample()

        '''ICM'''
        a_vec = F.one_hot(actions.squeeze(dim=-1), num_classes=self.action_dim)  # 将动作转化为 one-hot 向量
        pred_s_next, pred_a_vec, feature_next = self.ICM.Get_Full(states, new_states, a_vec)

        # 计算前向模型和反向动作预测的损失
        forward_loss = F.mse_loss(pred_s_next, feature_next.detach(), reduction='none')
        inverse_pred_loss = F.cross_entropy(pred_a_vec, torch.tensor(F.one_hot(actions.squeeze(dim=-1), num_classes=self.action_dim).detach(), dtype=torch.float32), reduction='none')

        # 计算奖励
        intrinsic_rewards = self.intrinsic_scale * forward_loss.mean(-1)
        total_rewards = intrinsic_rewards

        if self.use_extrinsic:
            total_rewards = rewards + intrinsic_rewards.unsqueeze(dim=-1)

        # current q, return related value according to current state and action
        actions = actions.long()
        q = self.q_network(states, bl=True).gather(1, actions)

        # target q, return all value according to next state
        # q_next = self.target_q_network(new_states, bl=True).detach()

        # y = r + gamma * target_q, select action-value pair in target q which maximize the q
        # y = rewards + (1 - dones) * self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
        # MSE loss
        # loss = self.MSELoss(q, y)

        '''ICM'''
        # 选择动作对应的 Q 值
        predicted_qvalues_for_actions = torch.sum(torch.multiply(a_vec, q), axis=1)

        # 计算现态所有动作的 Q 值
        predicted_next_qvalues = self.target_q_network(new_states)

        # 计算次态所有动作的 Q 值
        next_state_values = predicted_next_qvalues.max(-1)[0]

        # Q 学习核心公式
        target_qvalues_for_actions = total_rewards + self.gamma * next_state_values.unsqueeze(dim=-1)

        # 对于最后的状态，可以进行一些简化Q(s,a) = r(s,a), 因为 s' 不存在
        target_qvalues_for_actions = torch.where(torch.tensor(dones, dtype=bool), total_rewards, target_qvalues_for_actions)

        # Lasso 损失
        Q_loss = F.smooth_l1_loss(predicted_qvalues_for_actions, target_qvalues_for_actions.detach())
        loss = self.Qloss_scale * Q_loss + self.forward_scale * forward_loss.mean() + self.inverse_scale * inverse_pred_loss.mean()

        # writer.add_scalar(f'dqn_loss/{self.tl_id}', loss, self.writer_step)
        self.writer_step += 1

        # back propagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the target q network
        if self.buf_cnt % self.update == 0:  # update timely
            self.soft_update(self.q_network, self.target_q_network, tau=self.tau)

        return loss.item()
