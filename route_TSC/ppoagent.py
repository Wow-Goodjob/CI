import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
import threading
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import Categorical
from agent import Agent
from net import Actor, Critic, PPOReplayBuffer
import config

# from torch.utils.tensorboard import SummaryWriter
import time
import os

Q_LR = config.RL["Q_LR"]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# model = 'PPO'
# tm = str(time.strftime('%Y-%m-%d-%H-%M', time.localtime()))
# os.makedirs(f'./log/{model}/{tm}', exist_ok=True)
# writer = SummaryWriter(f'./log/{model}/{tm}')


class PPOAgent(Agent):
    def __init__(self, state_dim, action_dim, args, tl_id, net_type, lr=Q_LR, n_steps=1, execution=False,
                 net_config=config.RL):
        super(PPOAgent, self).__init__(state_dim, action_dim, args, tl_id, net_type, n_steps=1, execution=execution,
                                       net_config=config.RL)
        self.batch_size = args.buffer_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr_a = args.lr_a  # Learning rate of actor
        self.lr_c = args.lr_c  # Learning rate of critic
        self.gamma = args.gamma  # Discount factor
        self.lamda = args.lamda  # GAE parameter
        self.epsilon = args.epsilon  # PPO clip parameter
        self.K_epochs = args.K_epochs  # PPO parameter
        self.entropy_coef = args.entropy_coef  # Entropy coefficient
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm
        self.replay_buffer = PPOReplayBuffer(args, state_dim)
        self.replay_buffers = []
        self.total_steps = 0
        self.args = args
        self.tl_id = tl_id

        self.actor = Actor(args, state_dim, action_dim).to(device)
        self.critic = Critic(args, state_dim, action_dim).to(device)
        if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

        self.a = 0
        self.a_logprob = 0.0

        self.writer_step = 0


    def test_buffer(self):
        print(f"[PPOAgent {self.tl_id}] Testing buffer...")

        test_state = np.random.randn(self.state_dim)
        test_action = 1
        test_logprob = 0.5
        test_reward = 1.0
        test_next_state = np.random.randn(self.state_dim)

        self.write(test_state, test_action, test_logprob, test_reward, test_next_state, False, False, id=0)

        if self.replay_buffers:
            print(f"[PPOAgent {self.tl_id}] Buffer test passed. Size: {len(self.replay_buffers[0].s)}")
        else:
            print(f"[PPOAgent {self.tl_id}] Buffer test failed. No buffers.")

    def evaluate(self, s):  # When evaluating the policy, we select the action with the highest probability
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        a_prob = self.actor(s).detach().numpy().flatten()
        a = np.argmax(a_prob)
        return a

    def choose_action(self, s, avail_actions=None):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0).to(device)
        with torch.no_grad():
            probs = self.actor(s).cpu()
            if avail_actions is not None:
                avail_actions = torch.tensor(avail_actions, dtype=bool)
                probs = torch.where(avail_actions, probs, torch.tensor(-1e8))
            dist = Categorical(logits=probs)
            a = dist.sample()
            a_logprob = dist.log_prob(a)
        return a.numpy()[0], a_logprob.numpy()[0]

    def write(self, s, a, a_logprob, r, s_, dw, done, id=-1):
        # self.replay_buffer.store(s, self.a, self.a_logprob, r, s_, dw, done)
        while len(self.replay_buffers) <= id:
            self.replay_buffers.append(PPOReplayBuffer(self.args, self.state_dim))
        self.replay_buffers[id].store(s, a, a_logprob, r, s_, dw, done)
        self.total_steps += 1
        self.a = a
        self.a_logprob = a_logprob

    def learn(self, buf=None):
        # 遍历所有的 ReplayBuffer
        for rb in self.replay_buffers:
            # 1. 获取数据长度
            sz = len(rb.s)
            if sz == 0:  # 如果没有数据，直接跳过
                continue

            # 2. 提取数据
            s, a, a_logprob, r, s_, dw, done = rb.s, rb.a, rb.a_logprob, rb.r, rb.s_, rb.dw, rb.done

            # 3. 【性能修复】先转为 numpy array，再转 tensor，消除 UserWarning
            s = torch.tensor(np.array(s), dtype=torch.float).to(device)
            a = torch.tensor(np.array(a), dtype=torch.long).to(device)
            a_logprob = torch.tensor(np.array(a_logprob), dtype=torch.float).to(device)
            r = torch.tensor(np.array(r), dtype=torch.float).to(device)
            s_ = torch.tensor(np.array(s_), dtype=torch.float).to(device)
            dw = torch.tensor(np.array(dw), dtype=torch.float).to(device)
            done = torch.tensor(np.array(done), dtype=torch.float).to(device)

            """
                Calculate the advantage using GAE
            """
            adv = []
            gae = 0
            with torch.no_grad():
                vs = self.critic(s)
                vs_ = self.critic(s_)
                deltas = r + self.gamma * (1.0 - dw) * vs_ - vs
                # 注意：这里要转回 cpu numpy 进行循环计算
                deltas_np = deltas.cpu().numpy().flatten()
                done_np = done.cpu().numpy().flatten()

                for delta, d in zip(reversed(deltas_np), reversed(done_np)):
                    gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                    adv.insert(0, gae)

                adv = torch.tensor(adv, dtype=torch.float).view(-1, 1).to(device)
                v_target = adv + vs
                if self.use_adv_norm:
                    adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

            # Optimize policy for K epochs:
            for _ in range(self.K_epochs):
                # 4. 【逻辑修复】BatchSampler 的 range 必须是数据长度 sz，而不是 1
                # 建议使用 self.mini_batch_size 进行小批次更新
                batch_size_to_use = self.mini_batch_size if hasattr(self, 'mini_batch_size') else sz

                for index in BatchSampler(SubsetRandomSampler(range(sz)), batch_size_to_use, False):
                    # 重新计算 log_prob 和 entropy
                    dist_now = Categorical(probs=self.actor(s[index]))
                    dist_entropy = dist_now.entropy().view(-1, 1)
                    a_logprob_now = dist_now.log_prob(a[index].squeeze()).view(-1, 1)

                    # r = p_new / p_old
                    ratios = torch.exp(a_logprob_now - a_logprob[index])

                    surr1 = ratios * adv[index]
                    surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]

                    # Actor Loss
                    actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy

                    self.optimizer_actor.zero_grad()
                    actor_loss.mean().backward()
                    if self.use_grad_clip:
                        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                    self.optimizer_actor.step()

                    # Critic Loss
                    v_s = self.critic(s[index])
                    critic_loss = F.mse_loss(v_target[index], v_s)

                    self.optimizer_critic.zero_grad()
                    critic_loss.backward()
                    if self.use_grad_clip:
                        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                    self.optimizer_critic.step()

            if hasattr(rb, 'clean'):
                rb.clean()
            else:
                # 假设你的 buffer 内部是用 list 存储的，手动清空
                rb.s = []
                rb.a = []
                rb.a_logprob = []
                rb.r = []
                rb.s_ = []
                rb.dw = []
                rb.done = []
                rb.count = 0

    def lr_decay(self, total_steps):
        lr_a_now = self.lr_a * (1 - total_steps / self.max_train_steps)
        lr_c_now = self.lr_c * (1 - total_steps / self.max_train_steps)
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now

    def load_model(self, name):
        self.actor.load_state_dict(torch.load('./model/' + name + '/actor' + self.tl_id + '.pt'))
        self.critic.load_state_dict(torch.load('./model/' + name + '/critic' + self.tl_id + '.pt'))
        self.optimizer_actor.load_state_dict(torch.load('./model/' + name + '/opt_a' + self.tl_id + '.pt'))
        self.optimizer_critic.load_state_dict(torch.load('./model/' + name + '/opt_c' + self.tl_id + '.pt'))

    def load_buffer(self, name):
        pass
        # self.replay_buffer.load(name, self.tl_id[4])

    def save_model(self, name):
        torch.save(self.actor.state_dict(), './model/' + name + '/actor' + self.tl_id + '.pt')
        torch.save(self.critic.state_dict(), './model/' + name + '/critic' + self.tl_id + '.pt')
        torch.save(self.optimizer_actor.state_dict(), './model/' + name + '/opt_a' + self.tl_id + '.pt')
        torch.save(self.optimizer_critic.state_dict(), './model/' + name + '/opt_c' + self.tl_id + '.pt')

    def save_buffer(self, name):
        pass
        # self.replay_buffer.save(name, self.tl_id[4])
