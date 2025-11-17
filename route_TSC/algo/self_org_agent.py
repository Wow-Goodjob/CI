from agent import CAVAgent
from net import DQN
import traci
import numpy as np
import torch
import torch.nn as nn
import env
import config


class DDQNNet(nn.Module):
    def __init__(self, state_dim, action_dim, args, agent_num, tl_id):
        super(DDQNNet, self).__init__()
        self.fc1 = nn.Linear(3840, 1024)
        self.relu = nn.ReLU()
        self.fc_value = nn.Linear(1024, 256)
        self.fc_adv = nn.Linear(1024, 256)

        self.value = nn.Linear(256, 1)
        self.adv = nn.Linear(256, action_dim)

    def forward(self, state, bl=False):
        y = self.relu(self.fc1(state))
        value = self.relu(self.fc_value(y))
        adv = self.relu(self.fc_adv(y))

        value = self.value(value)
        adv = self.adv(adv)

        advAverage = torch.mean(adv, dim=1, keepdim=True)
        Q = value + adv - advAverage
        return Q


class SelfOrgAgent(CAVAgent):
    def __init__(self, veh_id, router, adj_edge, args):
        super(SelfOrgAgent, self).__init__(veh_id, router, adj_edge, args)
        self.veh_id = veh_id
        self.router = router
        self.adj_edge = adj_edge
        self.args = args
        self.get_state_time = -1
        self.router_state = None

    def get_router_state2(self, road_state=None, traffic_light_state=None, road_list=None, greens=None, light_count=None,
                          lane_dict=None):
        cur_edge = traci.vehicle.getRoadID(self.veh_id)
        self.done = (self.dest in self.adj_edge[cur_edge].values())
        if traci.simulation.getTime() == self.get_state_time:
            return self.router_state
        a, b = 20, 2
        n_c = 3 / (self.args.rate * 11.11 + (1 - self.args.rate) * 22.22)
        th = b * n_c
        div = th / (a - 1)
        road_set = set()
        for r in road_state.keys():
            road_set.add(r[:-2])
        cell = np.zeros((len(road_set), 3))
        road_set = list(road_set)
        for i, (k, v) in enumerate(road_state.items()):
            veh_list = env.l_get_lane_vehicle_list(k)
            pos_list = []  # car position on current lane
            for veh in veh_list:
                pos_list.append(env.get_vehicle_position(veh))
            lane_len = env.get_lane_length(k)
            cut_point = np.linspace(0, lane_len, 4).tolist()  # divide lane into 3 pieces according to length
            cell_count = env.lane_position_mapper(pos_list, cut_point)
            cell[road_set.index(k[:-2])] += np.array(cell_count) / (lane_len / 3) / div
        cell = cell.reshape(-1, 1)
        rep_cell = np.zeros((len(cell), a))
        for i in range(len(cell)):
            if cell[i][0] >= th:
                rep_cell[i][-1] = 1
            else:
                rep_cell[i][int(cell[i][0])] = 1
        # cell = cell.flatten()
        # self.router_state = cell
        self.router_state = rep_cell.reshape(1, -1)[0]
        self.get_state_time = traci.simulation.getTime()
        return self.router_state

    def _available(self, edge, next_edges, direction):
        nx, ny = traci.lane.getShape(next_edges[direction] + '_0')[1]
        cx, cy = traci.lane.getShape(edge + '_0')[1]
        dn = ((nx - self.dest_x) ** 2 + (ny - self.dest_y) ** 2) ** 0.5
        dc = ((cx - self.dest_x) ** 2 + (cy - self.dest_y) ** 2) ** 0.5
        if dn <= dc or next_edges[direction] == self.dest:
            return True
        return False

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
        if sum(action_mask) == 0:
            shortest_dis = 3e5
            nearest_dir = 'n'
            for direction in directions:
                nx, ny = traci.lane.getShape(next_edges[direction] + '_0')[1]
                dn = ((nx - self.dest_x) ** 2 + (ny - self.dest_y) ** 2) ** 0.5
                if dn < shortest_dis:
                    nearest_dir = direction
            assert nearest_dir != 'n'
            self.avail_link.append(next_edges[nearest_dir])
            action_mask[mapping[nearest_dir]] = 1

        assert sum(action_mask) > 0
        return action_mask

    def get_reward(self, road_state=None):
        # scaled BRP: r_s
        cur_edge = traci.vehicle.getRoadID(self.veh_id)
        min_t = 99999999.99
        max_t = 0.0
        e, f, alpha = 1.0, 2.0, 0.2
        for lk in self.avail_link:
            _t = traci.lane.getLength(lk+'_0') / 11.11 * (1 + e * (traci.edge.getLastStepVehicleNumber(lk) /
                                                                   traci.lane.getLength(lk+'_0') * 7.5) ** f)
            min_t = min(min_t, _t)
            max_t = max(max_t, _t)

        self.avail_link = []
        _c = traci.lane.getLength(cur_edge+'_0') / 11.11 * (1 + e * (traci.edge.getLastStepVehicleNumber(cur_edge) /
                                                                     traci.lane.getLength(cur_edge+'_0') * 7.5) ** f)
        r_s = alpha * (max_t - _c) / (max_t - min_t) if max_t != min_t else alpha

        # r_ep
        p, q = -6, 3000 / (self.args.rate * self.args.veh_num)
        r_ep = 0

        if self.done:
            r_ep = p * (traci.simulation.getTime() - self.depart_time) / (self.args.rate * self.args.veh_num) + q

        return r_s + r_ep
