from agent import CAVAgent
from net import DQN
import traci
import numpy as np
import torch
import torch.nn as nn
import env
import config


class ADQNNet(nn.Module):
    def __init__(self, state_dim, action_dim, args, agent_num, tl_id):
        super(ADQNNet, self).__init__()
        self.fc1 = nn.Linear(8, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_dim)

    def forward(self, state, bl=False):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))

        return x


class AstarDQN(CAVAgent):
    def __init__(self, veh_id, router, adj_edge, args):
        super(AstarDQN, self).__init__(veh_id, router, adj_edge, args)
        self.induction = 0.2

    def get_router_state2(self, road_state=None, traffic_light_state=None, road_list=None, greens=None, light_count=None,
                          lane_dict=None):
        states = []
        cur_x, cur_y = traci.vehicle.getPosition(self.veh_id)
        cur_lane = traci.vehicle.getLaneID(self.veh_id)
        start, end = traci.lane.getShape(cur_lane)
        lane_x, lane_y = end
        veh_num = 0
        cur_road = cur_lane[:-2]

        cav_state = [abs(cur_x - self.dest_x), abs(cur_y - self.dest_y), 3, traci.lane.getLength(cur_lane),
                     traci.simulation.getDistance2D(self.start_x, self.start_y, self.dest_x, self.dest_y, isDriving=True),
                     traci.edge.getLastStepVehicleNumber(cur_road), traci.edge.getLastStepMeanSpeed(cur_road), 11.11]
        return np.array(cav_state)

    def _available(self, edge, next_edges, direction):
        pass

    def get_avail_action(self):
        edge = traci.vehicle.getRoadID(self.veh_id)

        if np.random.uniform() <= self.induction:
            sp_route = traci.simulation.findRoute(edge, self.dest).edges
            nxt_edge = sp_route[1]
            mapping = {"r": 0, "s": 1, "l": 2}
            action_mask = [0 for _ in range(3)]
            _edge = {v: k for k, v in self.adj_edge[edge].items()}
            action_mask[mapping[_edge[nxt_edge]]] = 1
            return action_mask

        else:
            next_edges = self.adj_edge[edge]
            action_mask = [1 for _ in range(3)]
            max_t, max_d = -1, -1
            for i, d in enumerate(["r", "s", "l"]):
                nxt_edge = next_edges[d]
                nxt_route = traci.simulation.findRoute(nxt_edge, self.dest).edges
                t = sum([traci.edge.getTraveltime(k) for k in nxt_route[:-1]])
                if t > max_t:
                    max_t = t
                    max_d = i
            action_mask[max_d] = 0
            return action_mask

    def get_reward(self, road_state=None):
        if self.done:
            return 0
        else:
            cur_edge = traci.vehicle.getRoadID(self.veh_id)
            edge_length = traci.lane.getLength(cur_edge + '_0')
            cur_speed = traci.edge.getLastStepMeanSpeed(cur_edge)
            return -(edge_length / cur_speed) if cur_speed != 0 else -edge_length
