from agent import CAVAgent
from net import DQN
import traci
import numpy as np
import torch
import torch.nn as nn
import env
import config


class NavNet(nn.Module):
    def __init__(self, state_dim, action_dim, args, agent_num, tl_id):
        super(NavNet, self).__init__()
        self.fc1 = nn.Linear(196, 512)
        self.relu = nn.ReLU()
        self.fc_value = nn.Linear(512, 128)
        self.fc_adv = nn.Linear(512, 128)

        self.value = nn.Linear(128, 1)
        self.adv = nn.Linear(128, action_dim)

    def forward(self, state, bl=False):
        y = self.relu(self.fc1(state))
        value = self.relu(self.fc_value(y))
        adv = self.relu(self.fc_adv(y))

        value = self.value(value)
        adv = self.adv(adv)

        advAverage = torch.mean(adv, dim=1, keepdim=True)
        Q = value + adv - advAverage
        return Q


class NavAgent(CAVAgent):
    def __init__(self, veh_id, router, adj_edge, args):
        super(NavAgent, self).__init__(veh_id, router, adj_edge, args)
        self.db = [0] * 3

    def get_router_state2(self, road_state=None, traffic_light_state=None, road_list=None, greens=None, light_count=None,
                          lane_dict=None):
        self.stage2()
        cur_edge = traci.vehicle.getRoadID(self.veh_id)
        self.done = (self.dest in self.adj_edge[cur_edge].values())
        road_set = set()
        for r in road_state.keys():
            road_set.add(r[:-2])

        road_set = list(road_set)
        states = np.zeros((len(road_set), 3))  # speed, inflow
        for i, (k, v) in enumerate(road_state.items()):
            states[road_set.index(k[:-2])][0] += v['number']
            states[road_set.index(k[:-2])][1] += v['velocity'] / 3
            states[road_set.index(k[:-2])][2] = v['length']

        # cur loc, des loc, trip length, cost ratio
        cur_x, cur_y = traci.vehicle.getPosition(self.veh_id)
        veh_states = [cur_x, cur_y, self.dest_x, self.dest_y]
        states = np.concatenate((np.array(states).flatten(), veh_states))
        return states  # 64*3+4 (, 196)

    def stage2(self):
        action_mask = self.get_avail_action()
        edge = traci.vehicle.getRoadID(self.veh_id)
        next_edges = self.adj_edge[edge]
        directions = list(next_edges.keys())
        mapping = {"r": 0, "s": 1, "l": 2}
        for i, direction in enumerate(mapping.keys()):
            if direction in directions:
                nx, ny = traci.lane.getShape(next_edges[direction] + '_0')[1]
                dn = ((nx - self.dest_x) ** 2 + (ny - self.dest_y) ** 2) ** 0.5
                self.db[i] = dn
                if action_mask[i] == 0:
                    self.db[i] = 1e5+7

    def get_reward(self, road_state=None):
        self.penalty = 0
        dt = traci.simulation.getTime() - self.act_time + self.penalty
        return -dt
