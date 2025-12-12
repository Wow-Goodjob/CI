import numpy as np
import math
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
import threading
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import Categorical
import config
import pickle


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, args, agent_num, tl_id):
        super(DQN, self).__init__()

        self.args = args
        self.hid_dim = 32  # for GRU based attention, N*r*1 mul 1*g = N*(r*g) [N*hid]
        self.state_dim = state_dim
        self.agent_num = agent_num
        self.tl_id = tl_id
        # self.tl_int = int(self.tl_id[4])
        # self.neighbor = config.COP[tl_id]
        self.neighbor = 1
        self.agent_num = args.agent_num

        self._state = args.state_dim  # single junction state
        # self.gat = GATModel(self._state, self.hid_dim, args, agent_num, tl_id)
        # self.gat = DglGraphAttention(self._state, self.hid_dim, args, tl_id)
        # self.gat = GAT(self._state, self.hid_dim, args)
        self.fc0 = nn.Linear(state_dim, self.hid_dim)
        if args.temporal == "GRU" and args.spatial == "FC":
            self.fc0 = nn.Linear(self._state, self.hid_dim)

        self.attn = nn.Parameter(torch.FloatTensor(1, self.neighbor + 1, self._state))

        # self.gru = ConvGRU(num_cells=2, in_channels=1, intermediate_channels=1, kernel_size=(1, 1), return_sequence=False)
        if args.temporal == "GRU":
            self.gru = GRUModel(self._state * args.history, self.hid_dim, batch_size=args.batch_size)
        # self.gru = []
        # for _ in range(self.neighbor + 1):
            # self.gru.append(GRUModel(self._state * args.history, self.hid_dim, batch_size=args.batch_size))

        self.fc1 = nn.Linear(self._state * self.hid_dim, self.hid_dim) \
            if self.args.temporal == "GRU" else nn.Linear(self.hid_dim, self.hid_dim)  # treat (r*g) as hidden dim
        if args.temporal == "GRU" and args.spatial == "FC":
            self.fc1 = nn.Linear(self.hid_dim, self.hid_dim)
        f1 = 1.0 / np.sqrt(self.hid_dim)
        self.fc1.weight.data.uniform_(-f1, f1)

        self.fc2 = nn.Linear(self.hid_dim, action_dim)
        f2 = 0.003
        self.fc2.weight.data.uniform_(-f2, f2)

        self.h = torch.randn(1, 1, self.hid_dim)
        self.b_h = torch.randn(1, args.batch_size, self.hid_dim)
        self.w = None

        self.use_attention = False
        if tl_id == "veh" and args.use_attention:
            self.use_attention = True
            self.n_road = args.road_feature * 4 * 3
            self.n_intersection = args.agent_num
            n_embd = 64
            self.road_emb = nn.Sequential(nn.LayerNorm(self.n_road),
                                          init_(nn.Linear(self.n_road, n_embd), activate=True), nn.GELU())
            self.cav_emb = nn.Sequential(nn.LayerNorm(args.cav_feature),
                                         init_(nn.Linear(args.cav_feature, n_embd), activate=True), nn.GELU())
            self.att = SelfAttention(n_embd=n_embd, n_head=1, n_agent=self.n_road)
            self.out_layer = nn.Sequential(init_(nn.Linear(n_embd, self.hid_dim), activate=True), nn.GELU())

        # self.gru_fc = nn.Linear(self.hid_dim, self.hid_dim)

    # return value of each action
    def forward(self, x, bl=False):
        if self.use_attention:
            B, LD = x.shape
            road = self.road_emb(x[:, :-self.args.cav_feature].reshape(B, self.n_intersection, self.n_road))  # B*N*F -> B*N*emb
            cav = self.cav_emb(x[:, -self.args.cav_feature:]).unsqueeze(dim=1)  # B*1*F' -> B*1*Emb
            att = self.att(road, road, cav).squeeze(dim=1)  # K, V, Q
            x0 = self.out_layer(att)
        elif self.args.temporal == "FC" and self.args.spatial == "FC":
            x0 = F.relu(self.fc0(x))
        elif self.args.temporal == "FC" and self.args.spatial == "GAT":
            if bl:
                bn = x.size()[0]
                x = x.reshape(bn, -1, self._state)
                x0 = self.gat(x)
            else:
                x = x.reshape(1, -1, self._state)
                x0 = self.gat(x)
            x0 = x0.reshape(-1, self.hid_dim)
            # x1 = F.relu(self.fc1(x0))
            return self.fc2(x0)
        elif self.args.temporal == "GRU" and self.args.spatial == "GAT":
            # s_state = x.clone()
            # self.w = self.gru(s_state)  # update GAT parameter self.w
            if bl:
                bn = x.size()[0]
                x = x.reshape(bn, -1, self._state)
                s_state = x.clone()
                w = self.gru(s_state)
                x0 = self.gat(x, w)
            else:
                x = x.reshape(1, -1, self._state)
                s_state = x.clone()
                w = self.gru(s_state)
                x0 = self.gat(x, w)
            x0 = x0.reshape(-1, self.hid_dim)
            # x1 = F.relu(self.fc1(x0))
            return self.fc2(x0)
        elif self.args.temporal == "GRU" and self.args.spatial == "FC":
            bn = x.size()[0]
            x = x.reshape(-1, self._state)  # (B*N, state)
            x0 = self.fc0(x)  # (B*N, hidden)
            x0 = x0.reshape(-1, self.neighbor + 1, self.hid_dim)  # (B, N, hidden)
            x = x.reshape(bn, -1, self._state)
            s_state = x.clone()
            an = s_state.size()[1]
            for i in range(an):
                if i == 0:
                    w = self.gru[i](s_state[:, i, :].unsqueeze(dim=1))
                else:
                    w = torch.cat([w, self.gru[i](s_state[:, i, :].unsqueeze(dim=1))], dim=1)
            # print("b", x0.shape)
            x0 = x0 * w  # (B, N, hidden) * (N, hidden)
            # print("a", x0.shape)
            x0 = x0.mean(dim=1, keepdim=False)
            x0 = x0.reshape(-1, self.hid_dim)
            x1 = F.relu(self.fc1(x0))
            # print(x1.shape)
            return self.fc2(x1)

        x1 = F.relu(self.fc1(x0))
        x2 = self.fc2(x1)
        # if x2.requires_grad:
        #     g = make_dot(x2)
        #     g.view()
        return x2

    def update_w(self, state):
        s_state = state.clone()
        s_state = s_state.unsqueeze(dim=0)
        self.w = self.gru(s_state, False)


def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class Actor(nn.Module):
    def __init__(self, args, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc3 = nn.Linear(args.hidden_width, action_dim)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        a_prob = torch.softmax(self.fc3(s), dim=1)
        return a_prob


class Critic(nn.Module):
    def __init__(self, args, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, args.critic_hidden_width)
        self.fc2 = nn.Linear(args.critic_hidden_width, args.critic_hidden_width)
        self.fc3 = nn.Linear(args.critic_hidden_width, 1)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh
        self.layer_norm = nn.LayerNorm(state_dim)

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, s):
        s = self.layer_norm(s)
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        v_s = self.fc3(s)
        return v_s


class ReplayBuffer:
    def __init__(self, args, net_config=None):
        self.cnt = 0
        self.buffer_size = net_config["BUFFER_SIZE"]
        self.batch_size = net_config["BATCH_SIZE"]
        self.args = args
        self.memory = namedtuple("memory", field_names=["state", "action", "reward", "next_state", "done"])
        self.memory_buf = [0] * self.buffer_size

    def write(self, state, action, reward, next_state, done, q=None):
        index = self.cnt % self.buffer_size
        self.memory_buf[index] = self.memory(state, action, reward, next_state, done)
        self.cnt += 1

    def sample(self):
        experiences = random.sample(self.memory_buf[:min(self.cnt, self.buffer_size)], self.batch_size)

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

        return states, actions, rewards, next_states, dones


class QReplayBuffer(ReplayBuffer):
    def __init__(self, args):
        super(QReplayBuffer, self).__init__(args)
        self.memory = namedtuple("memory", field_names=["state", "action", "reward", "next_state", "done", "q"])

    def write(self, state, action, reward, next_state, done, q=None):
        index = self.cnt % self.buffer_size
        self.memory_buf[index] = self.memory(state, action, reward, next_state, done, q)
        self.cnt += 1

    def sample(self):
        experiences = random.sample(self.memory_buf, self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        if self.args.agent_type == "DQN":
            actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        else:
            actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        q = torch.tensor
        for i in range(len(experiences)):
            if i == 0:
                q = experiences[i].q
            else:
                q = torch.cat((q, experiences[i].q), dim=0)

        return states, actions, rewards, next_states, dones, q


class PPOReplayBuffer:
    def __init__(self, args, state_dim):
        self.size = args.buffer_size
        self.s = []
        self.a = []
        self.a_logprob = []
        self.r = []
        self.s_ = []
        self.dw = []
        self.done = []
        self.args = args
        self.state_dim = state_dim
        self.count = 0

    def clear(self):
        self.s = []
        self.a = []
        self.a_logprob = []
        self.r = []
        self.s_ = []
        self.dw = []
        self.done = []
        self.count = 0

    def store(self, s, a, a_logprob, r, s_, dw, done):

        self.s.append(s)
        self.a.append(a)
        self.a_logprob.append(a_logprob)
        self.r.append(r)
        self.s_.append(s_)
        self.dw.append(dw)
        self.done.append(done)
        self.count = (self.count + 1) % self.size

    @staticmethod
    def numpy_to_tensor(s, a, a_logprob, r, s_, dw, done):
        s = torch.tensor(s, dtype=torch.float)
        a = torch.tensor(a, dtype=torch.long)  # In discrete action space, 'a' needs to be torch.long
        a_logprob = torch.tensor(a_logprob, dtype=torch.float)
        r = torch.tensor(r, dtype=torch.float)
        s_ = torch.tensor(s_, dtype=torch.float)
        dw = torch.tensor(dw, dtype=torch.float)
        done = torch.tensor(done, dtype=torch.float)

        return s, a, a_logprob, r, s_, dw, done

    def sample(self, batch_size):
        idx = np.random.randint(0, self.count, batch_size)
        s = self.s[idx]
        a = self.a[idx]
        a_logprob = self.a_logprob[idx]
        r = self.r[idx]
        s_ = self.s_[idx]
        dw = self.dw[idx]
        done = self.done[idx]
        return self.numpy_to_tensor(s, a, a_logprob, r, s_, dw, done)

    def load(self, path, tl_id):
        with open(f"./model/{path}/buffers{tl_id}.pkl", "rb") as f:
            buffer_list = pickle.load(f)
            self.s = buffer_list[0]["s"]
            self.a = buffer_list[0]["a"]
            self.a_logprob = buffer_list[0]["a_logprob"]
            self.r = buffer_list[0]["r"]
            self.s_ = buffer_list[0]["s_"]
            self.dw = buffer_list[0]["dw"]
            self.done = buffer_list[0]["done"]
            self.count = buffer_list[0]["count"]

    def save(self, path, tl_id):
        with open(f"./model/{path}/buffers{tl_id}.pkl", "wb") as f:
            pickle.dump([{"s": self.s,
                          "a": self.a,
                          "a_logprob": self.a_logprob,
                          "r": self.r,
                          "s_": self.s_,
                          "dw": self.dw,
                          "done": self.done,
                          "count": self.count}], f)


class ValueNorm(nn.Module):
    """Normalize a vector of observations - across the first norm_axes dimensions"""

    def __init__(
        self,
        input_shape,
        norm_axes=1,
        beta=0.99999,
        per_element_update=False,
        epsilon=1e-5,
        device=torch.device("cpu"),
    ):
        super(ValueNorm, self).__init__()

        self.input_shape = input_shape
        self.norm_axes = norm_axes
        self.epsilon = epsilon
        self.beta = beta
        self.per_element_update = per_element_update
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.running_mean = nn.Parameter(
            torch.zeros(input_shape), requires_grad=False
        ).to(**self.tpdv)
        self.running_mean_sq = nn.Parameter(
            torch.zeros(input_shape), requires_grad=False
        ).to(**self.tpdv)
        self.debiasing_term = nn.Parameter(torch.tensor(0.0), requires_grad=False).to(
            **self.tpdv
        )

    def running_mean_var(self):
        """Get running mean and variance."""
        debiased_mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.running_mean_sq / self.debiasing_term.clamp(
            min=self.epsilon
        )
        debiased_var = (debiased_mean_sq - debiased_mean**2).clamp(min=1e-2)
        return debiased_mean, debiased_var

    @torch.no_grad()
    def update(self, input_vector):
        if isinstance(input_vector, np.ndarray):
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        batch_mean = input_vector.mean(dim=tuple(range(self.norm_axes)))
        batch_sq_mean = (input_vector**2).mean(dim=tuple(range(self.norm_axes)))

        if self.per_element_update:
            batch_size = np.prod(input_vector.size()[: self.norm_axes])
            weight = self.beta**batch_size
        else:
            weight = self.beta

        self.running_mean.mul_(weight).add_(batch_mean * (1.0 - weight))
        self.running_mean_sq.mul_(weight).add_(batch_sq_mean * (1.0 - weight))
        self.debiasing_term.mul_(weight).add_(1.0 * (1.0 - weight))

    def normalize(self, input_vector):
        if isinstance(input_vector, np.ndarray):
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        mean, var = self.running_mean_var()
        out = (input_vector - mean[(None,) * self.norm_axes]) / torch.sqrt(var)[
            (None,) * self.norm_axes
        ]

        return out

    def denormalize(self, input_vector):
        """Transform normalized data back into original distribution"""
        if isinstance(input_vector, np.ndarray):
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        mean, var = self.running_mean_var()
        out = (
            input_vector * torch.sqrt(var)[(None,) * self.norm_axes]
            + mean[(None,) * self.norm_axes]
        )

        out = out.cpu().numpy()

        return out


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module


def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)


class SelfAttention(nn.Module):

    def __init__(self, n_embd, n_head, n_agent, masked=False):
        super(SelfAttention, self).__init__()

        assert n_embd % n_head == 0
        self.masked = masked
        self.n_head = n_head
        # key, query, value projections for all heads
        self.key = init_(nn.Linear(n_embd, n_embd))
        self.query = init_(nn.Linear(n_embd, n_embd))
        self.value = init_(nn.Linear(n_embd, n_embd))
        # output projection
        self.proj = init_(nn.Linear(n_embd, n_embd))
        # if self.masked:
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(n_agent + 1, n_agent + 1))
                             .view(1, 1, n_agent + 1, n_agent + 1))

        self.att_bp = None

    def forward(self, key, value, query):
        Bq, Lq, Dq = query.size()
        B, L, D = key.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(key).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        q = self.query(query).view(Bq, Lq, self.n_head, Dq // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        v = self.value(value).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)

        # causal attention: (B, nh, L, hs) x (B, nh, hs, L) -> (B, nh, L, L)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # self.att_bp = F.softmax(att, dim=-1)

        if self.masked:
            att = att.masked_fill(self.mask[:, :, :L, :L] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        y = att @ v  # (B, nh, L, L) x (B, nh, L, hs) -> (B, nh, L, hs)
        y = y.transpose(1, 2).contiguous().view(Bq, Lq, Dq)  # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)
        return y


class ICM(DQN):
    def __init__(self, state_dim, action_dim, args, agent_num, tl_id):
        super(ICM, self).__init__(state_dim, action_dim, args, agent_num, tl_id)

        self.pred_module = nn.Sequential(
            init_(nn.Linear(self.hid_dim + action_dim, self.hid_dim)),
            nn.GELU(),
            init_(nn.Linear(self.hid_dim, self.hid_dim)),
            nn.GELU()
        )

        self.invpred_module = nn.Sequential(
            init_(nn.Linear(self.hid_dim * 2, self.hid_dim)),
            nn.GELU(),
            init_(nn.Linear(self.hid_dim, action_dim)),
            nn.GELU()
        )

    def Encoder(self, x):
        B, LD = x.shape
        road = self.road_emb(
            x[:, :-self.args.cav_feature].reshape(B, self.args.agent_num, self.n_feature))  # B*N*F -> B*N*emb
        cav = self.cav_emb(x[:, -self.args.cav_feature:]).unsqueeze(dim=1)  # B*1*F' -> B*1*Emb
        att = self.att(road, road, cav).squeeze(dim=1)  # K, V, Q
        x = self.out_layer(att)
        return x

    def forward(self, x):  # Encoder 的前向结构：
        # 得到特征
        feature = self.Encoder(x)
        return feature

    def Predict(self, feature_x, a_vec):
        # 输入当前的状态特征和动作(以 one-hot 的形式), 生成下一特征状态(编码后的特征)
        pred_s_next = torch.concat([feature_x, a_vec], axis=-1).detach()
        pred_s_next = self.pred_module(pred_s_next)
        return pred_s_next

    def Invpred(self, feature_x, feature_x_next):
        # 相反动作预测: 输入现态和次态, 输出预测动作(one-hot 形式)
        pred_a_vec = torch.concat([feature_x, feature_x_next], axis=-1)
        pred_a_vec = self.invpred_module(pred_a_vec)
        return F.softmax(pred_a_vec/1e-20, dim=-1)

    def Get_Full(self, x, x_next, a_vec):
        # 得到所有需要的特征(下一特征状态，下一动作)
        feature = self.Encoder(x)  # => 32
        feature_next = self.Encoder(x_next)  # => 32

        pred_s_next = self.Predict(feature, a_vec)  # 预测下一状态
        pred_a_vec = self.Invpred(feature, feature_next)  # （反向）动作预测：该动作

        return pred_s_next, pred_a_vec, feature_next

