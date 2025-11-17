from agent import CAVAgent
from net import DQN
import traci
import numpy as np
import torch
import torch.nn as nn
import env
import config


class IQLNet(nn.Module):
    def __init__(self, state_dim, action_dim, args, agent_num, tl_id):
        super(IQLNet, self).__init__()
        self.fc1 = nn.Linear(134, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_dim)

    def forward(self, state, bl=False):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))

        return x


class IQLBAgent(CAVAgent):
    def __init__(self, veh_id, router, adj_edge, args):
        super(IQLBAgent, self).__init__(veh_id, router, adj_edge, args)
        self.veh_id = veh_id
        self.router = router
        self.adj_edge = adj_edge
        self.args = args
        self.max_zeta = 0

    def get_router_state2(self, road_state=None, traffic_light_state=None, road_list=None, greens=None, light_count=None,
                          lane_dict=None):
        road_set = set()
        for r in road_state.keys():
            road_set.add(r[:-2])

        road_set = list(road_set)
        states = np.zeros((len(road_set), 2))  # speed, inflow
        for i, (k, v) in enumerate(road_state.items()):
            states[road_set.index(k[:-2])][0] += v['velocity'] / 3

            veh_list = env.l_get_lane_vehicle_list(k)
            pos_list = []  # car position on current lane
            for veh in veh_list:
                pos_list.append(env.get_vehicle_position(veh))
            if k[-1] == "0":
                tar = self.adj_edge[k[:-2]]['r']
            elif k[-1] == "1":
                tar = self.adj_edge[k[:-2]]['r']
            elif k[-1] == "2":
                tar = self.adj_edge[k[:-2]]['r']
            if tar in road_set:
                states[road_set.index(tar)][1] += len(list(filter(lambda element: element <= 100, pos_list)))

        # cur loc, des loc, trip length, cost ratio
        cur_x, cur_y = traci.vehicle.getPosition(self.veh_id)
        veh_states = [cur_x, cur_y, self.dest_x, self.dest_y, traci.vehicle.getDistance(self.veh_id),
                      traci.vehicle.getDistance(self.veh_id) /
                      traci.simulation.getDistance2D(self.start_x, self.start_y, cur_x, cur_y, isDriving=True)]
        states = np.concatenate((np.array(states).flatten(), veh_states))
        return states  # 64*2+6

    def _available(self, edge, next_edges, direction):
        return True

    def get_avail_action(self):
        edge = traci.vehicle.getRoadID(self.veh_id)
        mapping = {"r": 0, "s": 1, "l": 2, "t": 3}
        if self.args.direction == 3:
            mapping = {"r": 0, "s": 1, "l": 2}
        action_mask = [0 for _ in range(4)]
        if self.args.direction == 3:
            action_mask = [0 for _ in range(3)]
        next_edges = self.adj_edge[edge]

        directions = list(next_edges.keys())
        # default_exp_time = sum([traci.edge.getTraveltime(k) for k in list(traci.simulation.findRoute(edge, self.dest).edges)])

        if self.dest in self.adj_edge[edge].values():
            _edge = {v: k for k, v in self.adj_edge[edge].items()}
            action_mask[mapping[_edge[self.dest]]] = 1
            return action_mask

        for direction in mapping.keys():
            if direction in directions and self._available(edge, next_edges, direction):
            # if direction in directions and self.sp_available(edge, next_edges, direction):
                action_mask[mapping[direction]] = 1
                self.avail_link.append(next_edges[direction])

        return action_mask

    def get_reward(self, road_state=None):
        rou_1, rou_2 = 100, 200
        beta_1, beta_2 = 0.5, 0.1
        r_uo = -(traci.simulation.getTime() - self.act_time)
        r_dif = 0
        r_fair = -rou_1
        road_set = set()
        for r in road_state.keys():
            road_set.add(r[:-2])
        road_set = list(road_set)
        regional = [0] * len(road_set)
        road_set = list(road_set)
        cur_lane = traci.vehicle.getLaneID(self.veh_id)
        G_s, G_s_ = 0, 0

        for i, (k, v) in enumerate(road_state.items()):
            regional[road_set.index(k[:-2])] += v['number']

        mfd = lambda k, k_j: 11.11 * (1 - k/k_j)
        for i in range(len(regional)):
            G_s += regional[i] * mfd(regional[i] * 7.5, traci.lane.getLength(road_set[i] + "_0"))
            if cur_lane == road_set[i]:
                G_s_ += (regional[i] - 1) * mfd(regional[i] * 7.5 - 7.5, traci.lane.getLength(road_set[i] + "_0"))
            else:
                G_s_ += regional[i] * mfd(regional[i] * 7.5, traci.lane.getLength(road_set[i] + "_0"))
        G_s = G_s / sum(regional)
        G_s_ = G_s_ / (sum(regional) - 1)
        r_dif = sum(regional) * (G_s - G_s_)

        r_stay = -rou_2 / np.var(regional)
        r_su = r_dif + r_stay  # system utility

        cur_x, cur_y = traci.vehicle.getPosition(self.veh_id)
        zeta = traci.vehicle.getDistance(self.veh_id) / \
               traci.simulation.getDistance2D(self.start_x, self.start_y, cur_x, cur_y, isDriving=True)

        if zeta > self.max_zeta:
            self.max_zeta = zeta
            r_su += r_fair

        return beta_1 * r_uo + beta_2 * (1 - beta_1) * r_su
