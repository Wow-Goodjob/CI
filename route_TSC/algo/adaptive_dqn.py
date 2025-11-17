from agent import CAVAgent
from net import DQN
import traci
import numpy as np
import torch
import torch.nn as nn
import env
import config
import heapq


class AdaptiveQNet(nn.Module):
    def __init__(self, state_dim, action_dim, args, agent_num, tl_id):
        super(AdaptiveQNet, self).__init__()
        self.fc1 = nn.Linear(9, 64)
        self.relu = nn.PReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_dim)

    def forward(self, state, bl=False):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))

        return x


class AdaptiveDQN(CAVAgent):
    def __init__(self, veh_id, router, adj_edge, args):
        super(AdaptiveDQN, self).__init__(veh_id, router, adj_edge, args)
        self.edge_list = [e for e in traci.edge.getIDList() if e.startswith("road")]
        self.W = np.ones((len(self.edge_list), len(self.edge_list))) * 999
        self.edge_list_map = dict()
        for idx, e in enumerate(self.edge_list):
            self.edge_list_map[e] = idx
        self._edge_list_map = {v: k for k, v in self.edge_list_map.items()}
        for edge in self.edge_list:
            traci.edge.setEffort(edge, traci.lane.getLength(edge + '_0'))
            for _dir, _link in adj_edge[edge].items():
                if _dir in {'l', 's'}:
                    self.W[self.edge_list_map[edge]][self.edge_list_map[_link]] = 30/140
                else:
                    self.W[self.edge_list_map[edge]][self.edge_list_map[_link]] = 1e-9

    def get_route_with_lowest_impact(self):
        def dijkstra_with_path(graph, start, end):
            n = len(graph)
            distances = [float('inf')] * n
            distances[start] = 0
            previous = [-1] * n
            pq = [(0, start)]

            while pq:
                cur_dist, node = heapq.heappop(pq)
                if node == end:
                    path = []
                    while node != -1:
                        path.insert(0, node)
                        node = previous[node]
                    return distances[end], path
                if cur_dist > distances[node]:
                    continue
                for neighbor, weight in enumerate(graph[node]):
                    if weight > 0 and distances[node] + weight < distances[neighbor]:
                        distances[neighbor] = distances[node] + weight
                        previous[neighbor] = node
                        heapq.heappush(pq, (distances[neighbor], neighbor))
            return float('inf'), []  # If there's no path from start to end

        cur_edge = traci.vehicle.getRoadID(self.veh_id)
        start_node = self.edge_list_map[cur_edge]
        end_node = self.edge_list_map[self.dest]
        shortest_dis, shortest_path = dijkstra_with_path(self.W, start_node, end_node)
        route = [self._edge_list_map[e] for e in shortest_path]
        return route

    def get_router_state2(self, road_state=None, traffic_light_state=None, road_list=None, greens=None, light_count=None,
                          lane_dict=None):
        cur_edge = traci.vehicle.getRoadID(self.veh_id)
        self.done = (self.dest in self.adj_edge[cur_edge].values())
        traci.vehicle.rerouteEffort(self.veh_id)
        p1 = traci.vehicle.getRoute(self.veh_id)
        traci.vehicle.rerouteTraveltime(self.veh_id)
        p2 = traci.vehicle.getRoute(self.veh_id)
        p3 = self.get_route_with_lowest_impact()

        states = [0] * 9
        for i, _route in enumerate([p1, p2, p3]):
            num = sum([traci.edge.getLastStepVehicleNumber(_link) for _link in _route[1:]])
            length = sum([traci.edge.getLastStepHaltingNumber(_link) * 5 for _link in _route[1:]])
            exp_time = sum([min(traci.edge.getTraveltime(_link), traci.lane.getLength(_link+"_0")) for _link in _route[1:]])
            states[i*3+0] = num
            states[i*3+1] = length
            states[i*3+2] = exp_time
        for i in range(3):
            max_v = np.max(states[i::3])
            states[0+i] = 0 if max_v == 0 else states[0+i] / max_v
            states[3+i] = 0 if max_v == 0 else states[3+i] / max_v
            states[6+i] = 0 if max_v == 0 else states[6+i] / max_v
        # cur_x, cur_y = traci.vehicle.getPosition(self.veh_id)
        # states.extend([cur_x, cur_y, self.dest_x, self.dest_y])
        return states

    def get_next_edge(self, action, edge):
        choices = {0: "r", 1: "s", 2: "l", 3: "t"}
        if self.args.direction == 3:
            choices = {0: "r", 1: "s", 2: "l"}

        if edge == self.dest:
            assert RuntimeError
            new_route = [edge]
            traci.vehicle.setRoute(self.veh_id, new_route)
            return True, None, True
        # elif self.dest in self.adj_edge[edge].values():
        #     _edge = {v: k for k, v in self.adj_edge[edge].items()}
        #     _choices = {v: k for k, v in choices.items()}
        #     new_route = [edge, self.dest]
        #     traci.vehicle.setRoute(self.veh_id, new_route)
        #     return True, _choices[_edge[self.dest]], True
        else:
            if action == 0:
                traci.vehicle.rerouteEffort(self.veh_id)
            elif action == 1:
                traci.vehicle.rerouteTraveltime(self.veh_id)
            else:
                new_route = self.get_route_with_lowest_impact()
                traci.vehicle.setRoute(self.veh_id, new_route)
            return True, action, False

    def step(self, action):
        # traci.vehicle.highlight(self.veh_id, color=(255, 0, 0, 255))
        cur_edge = traci.vehicle.getRoadID(self.veh_id)
        # print(self.veh_id, cur_edge)
        self.act_his = {}
        if cur_edge not in self.act_his:
            self.act_his[cur_edge] = 1
        else:
            self.act_his[cur_edge] += 1
        self.act_his['cur'] = cur_edge
        self.act, action, done = self.get_next_edge(action, cur_edge)
        # if action is not None:
        self.action = action
        self.act_time = traci.simulation.getTime()
        return 1 if done else 0

    def _available(self, edge, next_edges, direction):
        pass

    def get_avail_action(self):
        return [1 for _ in range(3)]

    def get_reward(self, road_state=None):
        if self.done:
            return self.depart_time - traci.simulation.getTime()
        else:
            return 0
        # self.penalty = 0
        # dt = traci.simulation.getTime() - self.act_time + self.penalty
        # return -dt
