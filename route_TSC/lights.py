import numpy as np

import config
import env
import traffic_light as tl
import os
import sys
import platform
if platform.system().lower() == 'linux':
    os.environ['SUMO_HOME'] = "/usr/share/sumo"
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME"), 'tools'))
from matplotlib import pyplot as plt
import numpy as np
import traci
import torch
import statistics
import argparse
import time
# from torchsummary import summary
import traffic_light
import random
import pandas as pd
import csv

import config
import episode
import env
import lights
import tripinfo
from traffic_light import TLight
from utils import XmlGenerator, CsvInterpreter, Visualization
from arguments import *
import utils


# implement global agent
class Lights:
    def __init__(self, agent_list, args, action_set, execute=False):
        self.agent_num = len(agent_list)
        self.agent_list = agent_list
        self.rl_model = None
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        #self.agent_num = args.agent_num
        self.agent_type = args.agent_type
        self.args = args

        self.state = [0] * (self.state_dim * self.agent_num)
        self.action = [0] * (self.action_dim * self.agent_num)
        self.new_state = [0] * (self.state_dim * self.agent_num)
        self.global_state = None
        self.action_set = action_set

        # MAT info
        self.actions = None
        self.values = None
        self.action_log_probs = None
        self.rnn_states = None
        self.rnn_states_critic = None

        self.slot = 15
        self.rl_step = 0
        self.ob = None

        self.a = None
        self.a_logprob = None

        self.router_net=None
        self.adj_edge=None
        self.gen_cav=[]
        self.cav_list=[]
        self.leaving_cav_set=set()

        self.road_list=None
        self.light_count=None
        self.lane_dict=None
        self.greens=None

        self.joint_training=args.joint_training
        self.router_training_step = 0

        self.min_green=10
        self.last_action=[0]*self.agent_num
        self.phase_duration=[0]*self.agent_num

    def init_router(self,args):
        if args.router:
            from dqnagent import DQNAgent
            from ppoagent import PPOAgent

            self.adj_edge=env.find_adj_edge(f'./res/{args.map}/4_Phase.net.xml')
            state_dim=args.agent_num*args.road_feature*4*3+args.cav_feature

            if config.ROUTER_RL["ALGO"]=="DQN":
                self.router_net=DQNAgent(
                    state_dim=state_dim,action_dim=args.directions,
                    args=args,tl_id="veh",new_type="",net_config=config.ROUTER_RL
                )

            lanes=traci.lane.getIDList()
            lanes=[lane_id for lane_id in lanes if ":" not in lane_id]
            self.road_list,self.light_count,self.lane_dict=env.get_map_lanes(self.agent_list,lanes)
            if self.greens is None:
                self.greens=[0]*len(self.agent_list)

    def d_step(self, tm=-1):
        rewards = 0
        q_values = []
        is_terminated = False
        if self.args.agent_type == "PS-PPO":
            c_model = self.agent_list[0]
            if tm == 0:  # init step
                a = np.zeros(self.agent_num)
                a_logprob = np.zeros(self.agent_num)
                if self.global_state is None:
                    self.global_state = np.array([env.get_global_state(self.agent_list)]).reshape(self.args.agent_num,-1)
                for i in range(self.agent_num):
                    a[i], a_logprob[i] = c_model.rl_model.choose_action(self.global_state[i])
                    #---
                    self.last_action[i] = a[i]
                    self.phase_duration[i]=self.slot

                self.a = a
                self.a_logprob = a_logprob
                c_model.rl_model.a = self.a
                c_model.rl_model.a_logprob = self.a_logprob

                for i, a in enumerate(self.action_set):
                    self.action_set[i] = (self.slot, self.a[i], 'G')
            rewards_array = np.zeros(self.args.agent_num)
            for i in range(len(self.agent_list)):
                reward, self.action_set[i] = \
                    self.agent_list[i].step(self.action_set[i], self.agent_list, global_state=self.global_state,
                                            mat_action=self.a[i])
                duration, phase_index, phase_type = self.action_set[i]
                env.set_phase(self.agent_list[i], 2 * phase_index if phase_type == 'G' else 2 * phase_index + 1)

            #env.step()
            if env.get_time() % self.slot == self.slot - 1:
                next_state = np.array([env.get_global_state(self.agent_list)]).reshape(self.args.agent_num, -1)
                for i in range(len(self.agent_list)):
                    _, rewards_array[i] = env.get_reward(self.agent_list[i])
                done = env.get_time() >= 1799

                for i in range(self.agent_num):
                    self.agent_list[i].rl_model.write(self.global_state[i], a=self.a[i], a_logprob=self.a_logprob[i],
                                                      r=rewards_array[i], s_=next_state[i], dw=False, done=done,id=0)
                for i in range(self.agent_num):
                    c_model.rl_model.learn(buf=self.agent_list[i].rl_model.replay_buffer)
                self.global_state = next_state

                a = np.zeros(self.agent_num)
                a_logprob = np.zeros(self.agent_num)
                for i in range(self.agent_num):
                    #a[i], a_logprob[i] = c_model.rl_model.choose_action(self.global_state[i])  # next action
                    raw_action, a_logprob[i] = c_model.rl_model.choose_action(self.global_state[i])  # next action
                    if raw_action != self.last_action[i]:
                        if self.phase_duration[i] < self.min_green:
                            a[i] = self.last_action[i]
                            self.phase_duration[i] += self.slot  # 累加时间
                        else:
                            a[i] = raw_action
                            self.phase_duration[i] = self.slot  # 重置时间
                    else:
                        a[i] = raw_action
                        self.phase_duration[i] += self.slot  # 累加时间
                    self.last_action[i] = a[i]
                self.a = a
                self.a_logprob = a_logprob
            rewards = np.sum(rewards_array)
        else:
            raise NotImplementedError(f"Current code only support PS-PPO, but got {self.args.agent_type}")
        return rewards / len(self.agent_list), False

    def get_state(self):
        return self.state

    def get_phase(self):
        return self.action

    def get_agent_list(self):
        return self.agent_list

    def clear_buf(self):
        c_model = self.agent_list[0]
        c_model.rl_model.trainer.clear()

    def get_num(self):
        return self.agent_num

    def set_state(self, state):
        self.state = state
        return

    def set_phase(self, phase):
        self.action = phase
        return

    def load_model(self, name):
        if self.agent_type == "MAT":
            self.agent_list[0].rl_model.load_model(name)
            return
        if self.agent_type in {"MA2C"}:
            self.agent_list[0].rl_model.trainer.load_model(name)
            return
        # S/L global critic
        if self.agent_type == "QMIX":
            self.rl_model.load_model(name)
        for agent in self.agent_list:
            if isinstance(agent, tl.TLight):
                agent.rl_model.load_model(name)
        print(f'model loaded from {name}')

    def load_buffer(self, name):
        if self.args.agent_type == "MAT":
            self.agent_list[0].rl_model.load_buffer(name)
        else:
            for agent in self.agent_list:
                if isinstance(agent, tl.TLight):
                    agent.rl_model.load_buffer(name)

    def save_model(self, name):
        if self.agent_type == "MAT":
            self.agent_list[0].rl_model.save_model(name)
            return
        if self.agent_type in {"MA2C"}:
            self.agent_list[0].rl_model.trainer.save_model(name)
            return
        if self.agent_type == "QMIX":
            self.rl_model.save_model(name)
        for agent in self.agent_list:
            if isinstance(agent, tl.TLight):
                agent.rl_model.save_model(name)

    def save_buffer(self, name):
        if self.args.agent_type == "MAT":
            self.agent_list[0].rl_model.save_buffer(name)
        else:
            for agent in self.agent_list:
                if isinstance(agent, tl.TLight):
                    agent.rl_model.save_buffer(name)


# initialize Lights(L_1, L_2, ..., L_n)
def init(args, execute=False, path=None, seed=25):
    # start sumo-gui or sumo
    env.start(execute, path, str(seed))

    # get all agent in environment
    agent_list = create_agent_list(args, execution=execute)

    # neighbor fully observed, sort according to distance
    # for i in range(len(agent_list)):
    #     for j in range(len(agent_list)):
    #         if i != j:
    #             i_ = agent_list[i]
    #             j_ = agent_list[j]
    #             agent_list[i].add_neighbor(env.get_distance(i_, j_), j_)
    #     agent_list[i].neighbors.sort(key=lambda neighbor: neighbor[0])
    #     t = agent_list[i].neighbors[0][0]
    #     for a in agent_list[i].neighbors:
    #         a[0] = config.COP["DISCOUNT"] * t / a[0]

    # neighbor partial observed
    # for i in range(len(agent_list)):
    #     for j in range(len(agent_list)):
    #         if i != j and config.COP["ADJACENT"][i][j]:
    #             j_ = agent_list[j]
    #             agent_list[i].add_neighbor(j_)
    # create lights based on agent
    action_set = list()
    for i in range(len(agent_list)):
        action_set.append((config.INTERSECTION["GREEN"], 0, 'G'))  # Y or G, phase index, duration
    return Lights(agent_list, args, action_set, execute)


# initialize each light in agent list
def create_agent_list(args, execution=False):
    agent_list = list()
    tl_list = env.get_tl_list()
    for i in range(len(tl_list)):
        if i >= 0:
            agent_list.append(tl.TLight(tl_list[i], env.create_lane_to_det(), None,
                                        env.get_downstream(tl_list[i], args.reward, args.map), None,
                                        [], args,i, execution=execution))
        else:
            agent_list.append(tl.Light(tl_list[i], env.create_lane_to_det(), env.get_lane_map(tl_list[i]),
                                       env.get_downstream(tl_list[i], args.reward), env.get_upstream(tl_list[i]),
                                       [], args,i))
    # for agent in agent_list:
    #     agent._light = env.get_far_agent(agent.tl_id, agent_list)
    return agent_list
