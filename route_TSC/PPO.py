import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import traci

class ActorCritic(nn.Module):
    def __init__(self,state_dim,action_dim):
        super(ActorCritic,self).__init__()
        self.net=nn.Sequential(
            nn.Linear(state_dim,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
        )

        self.actor=nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,action_ddim),
            nn.Softmax(dim=-1)
        )

        self.critic=nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,1)
        )

    def forward(self, x):
        shared_features = self.net(x)
        action_probs=self.actor(shared_features)
        state=self.critic(shared_features)
        return action_probs, state

class PPOAgent:
    def __init__(self, state_dim,action_dim,lr=0.001,gamma=0.95,clip_epsilon=0.2):
        self.network=ActorCritic(state_dim,actor_dim)
        self.optimizer=optim.Adam(self.network.parameters(),lr=lr)
        self.gamma=gamma
        self.clip_epsilon=clip_epsilon

    def select_action(self,state):
        state=torch.FloatTensor(state).unsqueeze(0)
        probs,value=self.network(state)
        distribution=Categorical(probs)
        action=distribution.sample()
        return action.item(), distribution, log_prob(action), value

    def update(self,rewards,log_probs,values,next_value):
        returns=[]
        advantages=[]
        R=next_value

        for r in reversed(rewards):
            R=r+self.gamma*R
            returns.insert(0,R)

        returns=torch.FloatTensor(returns)
        values=torch.cat(values)

        advantages=returns-values

        log_probs=torch.cat(log_probs)
        ratio=torch.exp(log_probs-log_probs.detach())

        para1=ratio*advantages
        para2=torch.clamp(ratio,1-self.clip_epsilon,1+self.clip_epsilon)+advantages

        actor_loss=-torch.min(para1,para2).mean()
        critic_loss=advantages.pow(2).mean()

        total_loss=actor_loss+0.5*critical_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizar.step()
