from abc import ABC
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import traci
import math
import copy
import time

# from pfrl import replay_buffers, explorers

import net
from algo.p_buffer import PrioritizedReplayBuffer
from algo.prioritized_memory import Memory
import config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Agent:
    def __init__(self, state_dim, action_dim, args, tl_id, net_type, n_steps=1, execution=False, net_config=None):
        super(Agent, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.args = args
        self.tl_id = tl_id

        self.buf = net.ReplayBuffer(args, net_config)
        if args.algo == "nav":
            self.buf = Memory(net_config["BUFFER_SIZE"])
        # self.pfrl_buf = None
        # self.pfrl_buf = replay_buffers.PrioritizedReplayBuffer(args.buffer_size, num_steps=n_steps)
        self.norm_e = 0.0
        self.norm_var = 0.0
        self.net_type = net_type
        self.training = False
        self.execution = execution
        self.em = nn.Embedding(4, 2).to(device)
        self.em_state = nn.Embedding(32, 2).to(device)

    def get_normalization(self):
        return
        # exp = var = 0.0
        # cnt = 0
        # for e in self.buf.memory_buf:
        #     state = e.state
        #     exp += np.mean(state)
        #     var += np.var(state)
        #     cnt += 1
        # self.norm_e = exp
        # self.norm_var = var

    def get_embedding_action(self, action):
        action = torch.tensor([action]).to(device)
        # print("ck2", action.device)
        em_action = self.em(action)
        # print("ck3", em_action)
        return em_action.tolist()

    def get_embedding_state(self, state):
        state = torch.tensor(state).to(device)
        # print("ck2", action.device)
        em_state = self.em_state(state)
        # print("ck3", em_action)
        return em_state.tolist()

    def normalization(self, x, train):
        if train:
            return (x - self.norm_e) / np.sqrt(self.norm_var)
        else:
            return x

    def act(self, state, qnet=None, avail_actions=None):
        raise NotImplementedError

    def observe(self, obs, reward, done, reset=False):
        raise NotImplementedError

    def load_model(self, name):
        raise NotImplementedError

    def save_model(self, name):
        raise NotImplementedError

    @staticmethod
    def store(state, reward, next_state, done):
        raise NotImplementedError

    @staticmethod
    def update_w(s_series, s_index, s):
        raise NotImplementedError

    @staticmethod
    def learn():
        raise NotImplementedError


class SelectItem(nn.Module):
    def __init__(self, index):
        super(SelectItem, self).__init__()
        self.index = index

    def forward(self, x):
        return x[self.index]


class CAVAgent:
    def __init__(self, veh_id, router, adj_edge, args):

        self.args = args

        traci.vehicle.rerouteTraveltime(veh_id)  # dijkstra rerouting
        traci.vehicle.setTau(veh_id, 1.0)
        self.default_route = traci.vehicle.getRoute(veh_id)
        self.dest = self.default_route[-1]
        self.dest_x = traci.lane.getShape(self.dest + '_0')[0][0]
        self.dest_y = traci.lane.getShape(self.dest + '_0')[0][1]

        # self.exp_travel_time = sum([traci.edge.getTraveltime(k) for k in self.default_route[:-1]])
        self.default_travel_distance = sum([traci.lane.getLength(k + '_0') for k in self.default_route[:-1]])
        self.depart_time = traci.simulation.getTime()

        self.start_x, self.start_y = traci.vehicle.getPosition(veh_id)

        self.veh_id = veh_id
        self.router = router
        self.travel_time = 0
        self.act_his = {}
        self.act = False
        self.cav_state = None
        self.action = -1
        self.adj_edge = adj_edge
        self.act_time = 0
        self.act_pos = traci.lane.getShape(traci.vehicle.getRoadID(self.veh_id) + '_0')[1]
        self.done = False

        self.ppo_a = None
        self.ppo_a_logprob = None

        self.cur_x = traci.vehicle.getPosition(veh_id)[0]
        self.cur_y = traci.vehicle.getPosition(veh_id)[1]
        self.cur_d = ((self.cur_x - self.dest_x) ** 2 + (self.cur_y - self.dest_y) ** 2) ** 0.5
        self.default_d = self.cur_d

        self.termination_reward = -1
        self.avail_link = []

        self.reward = []
        self.penalty = 0
        self.dc = ((self.cur_x - self.dest_x) ** 2 + (self.cur_y - self.dest_y) ** 2) ** 0.5

    def append_reward(self, r):
        self.reward.append(r)

    def get_cav_state(self, road_state=None, traffic_light_state=None):
        veh_id = self.veh_id
        state = [traci.vehicle.getSpeed(veh_id)]
        x, y = traci.vehicle.getPosition(veh_id)
        state.append(x)
        state.append(y)
        lane = traci.vehicle.getLaneID(veh_id)
        # _, _, _, _, sublane = lane.split('_')
        # state.append(int(sublane))
        s, e = traci.lane.getShape(self.dest + '_0')
        state.append((s[0] + s[1]) / 2)
        state.append((e[0] + e[1]) / 2)

        if road_state is not None:
            edge = traci.vehicle.getRoadID(veh_id)
            if edge in road_state:
                road_state[edge][-1] = 1
            state.extend([item for sublist in road_state.values() for item in sublist])

        if traffic_light_state is not None:
            state.extend(traffic_light_state)

        # self.cav_state = state
        return state

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
        nx, ny = traci.lane.getShape(next_edges[direction] + '_0')[1]
        cx, cy = traci.lane.getShape(edge + '_0')[1]
        dn = ((nx - self.dest_x) ** 2 + (ny - self.dest_y) ** 2) ** 0.5
        dc = ((cx - self.dest_x) ** 2 + (cy - self.dest_y) ** 2) ** 0.5
        if dn <= dc or next_edges[direction] == self.dest:
            return True
        else:
            _edge = next_edges[direction]
            _mapping = {"r": 0, "s": 1, "l": 2, "t": 3}
            if self.args.direction == 3:
                _mapping = {"r": 0, "s": 1, "l": 2}
            _next_edges = self.adj_edge[_edge]
            _directions = list(_next_edges.keys())
            for _direction in _mapping.keys():
                if _direction in _directions:
                    nx, ny = traci.lane.getShape(_next_edges[_direction] + '_0')[1]
                    # cx, cy = traci.lane.getShape(_edge + '_0')[1]
                    dn = ((nx - self.dest_x) ** 2 + (ny - self.dest_y) ** 2) ** 0.5
                    dc = ((cx - self.dest_x) ** 2 + (cy - self.dest_y) ** 2) ** 0.5
                    if dn + 80 <= dc or _next_edges[_direction] == self.dest:
                        return True
            return False

    def sp_available(self, edge, next_edges, direction):
        if next_edges[direction] == self.dest:
            return True
        else:
            _edge = next_edges[direction]
            sp_route = traci.simulation.findRoute(edge, self.dest).edges
            rl_route = traci.simulation.findRoute(_edge, self.dest).edges

            if len(rl_route) <= len(sp_route):
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
            action_mask[mapping[nearest_dir]] = 1

        assert sum(action_mask) > 0
        return action_mask

    def get_next_edge(self, action, edge):
        choices = {0: "r", 1: "s", 2: "l", 3: "t"}
        if self.args.direction == 3:
            choices = {0: "r", 1: "s", 2: "l"}

        if edge == self.dest:
            assert RuntimeError
            new_route = [edge]
            traci.vehicle.setRoute(self.veh_id, new_route)
            return True, None, True
        elif self.dest in self.adj_edge[edge].values():
            _edge = {v: k for k, v in self.adj_edge[edge].items()}
            _choices = {v: k for k, v in choices.items()}
            new_route = [edge, self.dest]
            traci.vehicle.setRoute(self.veh_id, new_route)
            return True, _choices[_edge[self.dest]], True
        else:
            direction = choices[action]
            if direction in list(self.adj_edge[edge].keys()):
                next_edge = self.adj_edge[edge][direction]
            else:
                assert RuntimeError
                # next_edge = self.adj_edge[edge][list(self.adj_edge[edge].keys())[0]]
            # state = traci.simulation.findRoute(next_edge, self.dest)
            # new_route = [edge] + list(state.edges)
            new_route = [edge, next_edge]
            traci.vehicle.setRoute(self.veh_id, new_route)
            return True, action, False

    def get_reward(self, road_state=None):
        if self.done:
            # dt = traci.simulation.getTime() - self.act_time
            # return -dt
            # assert self.termination_reward > 0
            return self.termination_reward
        else:
            # cur_edge = traci.vehicle.getRoadID(self.veh_id)

            # distance
            # if cur_edge in self.act_his and self.act_his[cur_edge] > 1:
            #     raise RuntimeError
            #     # return -self.act_his[cur_edge] * 100
            # cur_x = traci.vehicle.getPosition(self.veh_id)[0]
            # cur_y = traci.vehicle.getPosition(self.veh_id)[1]
            # cur_d = ((cur_x - self.dest_x) ** 2 + (cur_y - self.dest_y) ** 2) ** 0.5
            # dd = cur_d - self.cur_d
            # self.cur_d = cur_d

            # time
            # edge = traci.vehicle.getRoadID(self.veh_id)
            # cx, cy = traci.lane.getShape(edge + '_0')[1]
            # dc = ((cx - self.dest_x) ** 2 + (cy - self.dest_y) ** 2) ** 0.5
            # self.penalty = min(0, self.dc - dc) / 11.11
            # self.dc = dc
            self.penalty = 0
            dt = traci.simulation.getTime() - self.act_time + self.penalty
            return -dt

            # scaled BRP
            # min_t = 99999999.99
            # max_t = 0.0
            # e, f, alpha = 1.0, 2.0, 0.2
            # for lk in self.avail_link:
            #     _t = traci.lane.getLength(lk+'_0') / 11.11 * (1 + e * (traci.edge.getLastStepVehicleNumber(lk) /
            #                                                            traci.lane.getLength(lk+'_0') * 7.5) ** f)
            #     min_t = min(min_t, _t)
            #     max_t = max(max_t, _t)

            # self.avail_link = []
            # _c = traci.lane.getLength(cur_edge+'_0') / 11.11 * (1 + e * (traci.edge.getLastStepVehicleNumber(cur_edge) /
            #                                                              traci.lane.getLength(cur_edge+'_0') * 7.5) ** f)
            # return alpha * (max_t - _c) / (max_t - min_t) if max_t != min_t else alpha

    def is_valid(self):
        road = traci.vehicle.getRoadID(self.veh_id)
        if len(road) == 0 or road[0] == ':':
            return False
        vx, vy = traci.vehicle.getPosition(self.veh_id)
        lx, ly = traci.lane.getShape(traci.vehicle.getLaneID(self.veh_id))[1]
        dis = ((vx - lx) ** 2 + (vy - ly) ** 2) ** 0.5
        if 100 <= dis <= 150 and ('cur' not in self.act_his or self.act_his['cur'] != road) and \
                ('t' in self.adj_edge[road]):
            next_road = self.adj_edge[road]['t']
            new_route = [road, next_road]
            traci.vehicle.setRoute(self.veh_id, new_route)
            self.act = True
            self.act_his = {}
            if road not in self.act_his:
                self.act_his[road] = 1
            else:
                self.act_his[road] += 1
            self.act_his['cur'] = road
        return 100 <= dis <= 150 and ('cur' not in self.act_his or self.act_his['cur'] != road) and \
            ('t' not in self.adj_edge[road])
        # return len(road) > 0 and road[0] != ':' and road not in self.act_his and 100 <= dis <= 150

    def arrived(self):
        road = traci.vehicle.getRoadID(self.veh_id)
        return self.dest == road

    def get_router_state2(self, road_state=None, traffic_light_state=None, road_list=None, greens=None, light_count=None,
                          lane_dict=None):

        # list stores attribute v_{avr}
        edge_average_speed = []
        # list stores attribute d_{veh}
        edge_density = []
        # list stores attribute t_{avr}
        edge_travel_time = []
        # list stores attribute dis_{aim}
        edge_end_distance = []
        # list stores attribute position encoding P
        position_encoding = []
        # list stores 0/1 (destination: 1  else: 0)
        destination_mask = []
        # current lane 0/1
        current_mask = []
        # edge_angle
        edge_angle = []

        phase = []
        green = []

        # obtain the current position of the rl vehicle
        current_x, current_y = traci.vehicle.getPosition(self.veh_id)
        cur_lane = traci.vehicle.getLaneID(self.veh_id)

        _road_state = copy.deepcopy(road_state)

        # for k, v in _road_state.items():
        #     nx, ny = _road_state[k]['ex'], _road_state[k]['ey']
        #     cx, cy = current_x, current_y
        #     dn = ((nx - self.dest_x) ** 2 + (ny - self.dest_y) ** 2) ** 0.5
        #     dc = ((cx - self.dest_x) ** 2 + (cy - self.dest_y) ** 2) ** 0.5
        #     if dn > dc and self.dest not in k:
        #         del road_state[k]

        # obtain the position encoding of the edges
        start_positions = dict()
        end_positions = dict()
        distance_approximation = dict()

        for d in road_state:
            start_positions[d] = ((road_state[d]["sx"] - current_x) ** 2 + (
                        road_state[d]["sy"] - current_y) ** 2) ** 0.5
            end_positions[d] = ((road_state[d]["ex"] - self.dest_x) ** 2 + (
                        road_state[d]["ey"] - self.dest_y) ** 2) ** 0.5
            distance_approximation[d] = start_positions[d] + end_positions[d] + road_state[d]["length"]

        # sort the attributes of all edges according to the distance between the rl vehicle and the destination
        # sorted_positions = sorted(distance_approximation.items(), key=lambda x: x[1], reverse=False)

        sorted_road_list = [sorted(road_list[i:i + 12]) for i in range(0, len(road_list), 12)]
        flattened_sorted_list = [item for sublist in sorted_road_list for item in sublist]

        sx_list = []
        sy_list = []

        for k in flattened_sorted_list:
            # if k not in road_state:
            #     position_encoding.append(0)
            #     edge_average_speed.append(0)
            #     edge_density.append(0)
            #     edge_travel_time.append(0)
            #     edge_angle.append(0)
            #     edge_end_distance.append(0)
            #     current_mask.append(0)
            #     destination_mask.append(0)
            #     phase.append(0)
            #     green.append(0)
            #     sx_list.append(0)
            #     sy_list.append(0)
            #     continue

            position_encoding.append(distance_approximation[k])
            edge_average_speed.append(road_state[k]["velocity"])
            edge_density.append(road_state[k]["density"])
            edge_travel_time.append(road_state[k]["time"])
            cx = road_state[k]["sx"]
            cy = road_state[k]["sy"]

            sx_list.append(cx)
            sy_list.append(cy)

            if cx - current_x == 0:
                edge_angle.append(np.pi / 2)
            else:
                edge_angle.append(np.arctan((cy - current_y) / (cx - current_x)))

            distance = ((cx - self.dest_x) ** 2 + (cy - self.dest_y) ** 2) ** 0.5
            edge_end_distance.append(distance)
            if lane_dict[k] == lane_dict[cur_lane]:
                current_mask.append(1)
            else:
                current_mask.append(0)
            if lane_dict[k] == lane_dict[self.dest + '_0']:
                destination_mask.append(1)
            else:
                destination_mask.append(0)
            phase.append(traci.trafficlight.getPhase(lane_dict[k]))
            green.append(35 - int(traci.simulation.getTime()) % 35)

        cur_edge = traci.vehicle.getRoadID(self.veh_id)

        cur_velocity = traci.vehicle.getSpeed(self.veh_id)  # V
        cur_numbers = [traci.lane.getLastStepVehicleNumber(f'{cur_edge}_{i}') for i in range(3)]  # N_i
        cur_lengths = [traci.lane.getLength(f'{cur_edge}_{i}') for i in range(3)]  # L_i
        cur_densities = [a / b for a, b in zip(cur_numbers, cur_lengths)]  # D_i
        cur_time = [traci.lane.getTraveltime(f'{cur_edge}_{i}') for i in range(3)]  # ET_i
        current_x, current_y = traci.vehicle.getPosition(self.veh_id)  # c_x, c_y
        s, e = traci.lane.getShape(self.dest + '_1')  # d_x, d_y
        des_sx = s[0]
        des_sy = s[1]
        veh_dis = ((current_x - des_sx) ** 2 + (current_y - des_sy) ** 2) ** 0.5
        if des_sx - current_x == 0:
            cur_edge_angle = np.pi / 2
        else:
            cur_edge_angle = np.arctan((des_sy - current_y) / (des_sx - current_x))

        # normalize all the input attributes
        if max(edge_average_speed) == 0:
            edge_average_speed_result = edge_average_speed
        else:
            edge_average_speed_result = [speed / max(edge_average_speed)
                                         for speed in edge_average_speed]
            cur_velocity = cur_velocity / max(edge_average_speed)

        if max(edge_density) == 0:
            edge_density_result = edge_density
        else:
            edge_density_result = [density / max(edge_density)
                                   for density in edge_density]
            cur_densities = [density / max(edge_density)
                             for density in cur_densities]

        if max(edge_end_distance) == 0:
            final_edge_end_distance = edge_end_distance
        else:
            final_edge_end_distance = [distance / max(edge_end_distance)
                                       for distance in edge_end_distance]
            veh_dis = veh_dis / max(edge_end_distance)

        if max(edge_travel_time) == 0:
            final_edge_travel_time = edge_travel_time
        else:
            final_edge_travel_time = [time / max(edge_travel_time)
                                      for time in edge_travel_time]
            cur_time = [time / max(edge_travel_time)
                        for time in cur_time]

        max_angle = 0
        for i in range(len(edge_angle)):
            if abs(edge_angle[i]) > max_angle:
                max_angle = abs(edge_angle[i])
        edge_angle = [angle / max_angle for angle in edge_angle]
        cur_edge_angle = cur_edge_angle / max_angle

        final_position_encoding = [item / max(position_encoding) for item in position_encoding]

        final_sx_list = [x / max(sx_list) for x in sx_list]
        final_sy_list = [y / max(sy_list) for y in sy_list]

        # action = []

        # if len(actions) < 1:
        #     actions = [0] * 16
        # for i, l, g, a in zip(light_count, traffic_light_state, greens, actions):
        #     phase.extend([l] * i)
        #     green.extend([g] * i)
        #     action.extend([a] * i)

        _state = list(zip(edge_average_speed_result, edge_density_result, final_edge_end_distance,
                          current_mask, destination_mask, final_edge_travel_time, edge_angle, final_position_encoding,
                          phase, green, final_sx_list, final_sy_list))
        state = [i for _s in _state for i in _s]
        # l = len(state)
        # state = state + [0] * (self.args.agent_num * self.args.road_feature * 4 * 3 - l)

        veh_feature = [cur_velocity] + cur_densities + [veh_dis] + [cur_edge_angle] + cur_time + \
                      [current_x / max(sx_list), current_y / max(sy_list)]
        state.extend(veh_feature)

        # self.cav_state = state
        self.done = (self.dest in self.adj_edge[cur_edge].values())
        if self.done:
            # cost = traci.simulation.getTime() - self.depart_time
            # self.termination_reward = (math.exp(self.default_travel_distance / 11.11 / cost) - math.exp(0)) * 300
            cur_travel_distance = sum([traci.lane.getLength(k + '_0') for k in traci.vehicle.getRoute(self.veh_id)])
            self.termination_reward = self.default_travel_distance / cur_travel_distance * 100

        # self.done = False

        return state

    def get_router_state(self, road_state=None, traffic_light_state=None, road_list=None, greens=None, light_count=None,
                         actions=None, lane_dict=None):
        """
        obtain the average speeds and vehicle density of each edge in the network
        :return: the state numpy array
        """
        # list stores attribute v_{avr}
        edge_average_speed = []
        # list stores attribute d_{veh}
        edge_density = []
        # list stores attribute t_{avr}
        edge_travel_time = []
        # list stores attribute dis_{aim}
        edge_end_distance = []
        # list stores attribute position encoding P
        position_encoding = []
        # list stores 0/1 (destination: 1  else: 0)
        destination_mask = []
        # current lane 0/1
        current_mask = []
        # edge_angle
        edge_angle = []

        # obtain the current position of the rl vehicle
        current_x, current_y = traci.vehicle.getPosition(self.veh_id)

        # obtain the position encoding of the edges
        start_positions = dict()
        end_positions = dict()
        distance_approximation = dict()

        for d in road_state:
            start_positions[d] = ((road_state[d]["sx"] - current_x) ** 2 + (
                        road_state[d]["sy"] - current_y) ** 2) ** 0.5
            end_positions[d] = ((road_state[d]["ex"] - self.dest_x) ** 2 + (
                        road_state[d]["ey"] - self.dest_y) ** 2) ** 0.5
            distance_approximation[d] = start_positions[d] + end_positions[d] + road_state[d]["length"]

        # sort the attributes of all edges according to the distance between the rl vehicle and the destination
        # sorted_positions = sorted(distance_approximation.items(), key=lambda x: x[1], reverse=False)

        cur_lane = traci.vehicle.getLaneID(self.veh_id)

        for k in road_list:
            position_encoding.append(distance_approximation[k])
            edge_average_speed.append(road_state[k]["velocity"])
            edge_density.append(road_state[k]["density"])
            edge_travel_time.append(road_state[k]["time"])
            cx = road_state[k]["sx"]
            cy = road_state[k]["sy"]

            if cx - current_x == 0:
                edge_angle.append(np.pi / 2)
            else:
                edge_angle.append(np.arctan((cy - current_y) / (cx - current_x)))

            distance = ((cx - self.dest_x) ** 2 + (cy - self.dest_y) ** 2) ** 0.5
            edge_end_distance.append(distance)
            if lane_dict[k] == lane_dict[cur_lane]:
                current_mask.append(1)
            else:
                current_mask.append(0)
            if lane_dict[k] == lane_dict[self.dest + '_0']:
                destination_mask.append(1)
            else:
                destination_mask.append(0)

        cur_edge = traci.vehicle.getRoadID(self.veh_id)

        cur_velocity = traci.vehicle.getSpeed(self.veh_id)  # V
        cur_numbers = [traci.lane.getLastStepVehicleNumber(f'{cur_edge}_{i}') for i in range(3)]  # N_i
        cur_lengths = [traci.lane.getLength(f'{cur_edge}_{i}') for i in range(3)]  # L_i
        cur_densities = [a / b for a, b in zip(cur_numbers, cur_lengths)]  # D_i
        cur_time = [traci.lane.getTraveltime(f'{cur_edge}_{i}') for i in range(3)]  # ET_i
        current_x, current_y = traci.vehicle.getPosition(self.veh_id)  # c_x, c_y
        s, e = traci.lane.getShape(self.dest + '_1')  # d_x, d_y
        des_sx = s[0]
        des_sy = s[1]
        veh_dis = ((current_x - des_sx) ** 2 + (current_y - des_sy) ** 2) ** 0.5
        if des_sx - current_x == 0:
            cur_edge_angle = np.pi / 2
        else:
            cur_edge_angle = np.arctan((des_sy - current_y) / (des_sx - current_x))

        # normalize all the input attributes
        if max(edge_average_speed) == 0:
            edge_average_speed_result = edge_average_speed
        else:
            edge_average_speed_result = [speed / max(edge_average_speed)
                                         for speed in edge_average_speed]
            cur_velocity = cur_velocity / max(edge_average_speed)

        if max(edge_density) == 0:
            edge_density_result = edge_density
        else:
            edge_density_result = [density / max(edge_density)
                                   for density in edge_density]
            cur_densities = [density / max(edge_density)
                             for density in cur_densities]

        if max(edge_end_distance) == 0:
            final_edge_end_distance = edge_end_distance
        else:
            final_edge_end_distance = [distance / max(edge_end_distance)
                                       for distance in edge_end_distance]
            veh_dis = veh_dis / max(edge_end_distance)

        if max(edge_travel_time) == 0:
            final_edge_travel_time = edge_travel_time
        else:
            final_edge_travel_time = [time / max(edge_travel_time)
                                      for time in edge_travel_time]
            cur_time = [time / max(edge_travel_time)
                        for time in cur_time]

        max_angle = 0
        for i in range(len(edge_angle)):
            if abs(edge_angle[i]) > max_angle:
                max_angle = abs(edge_angle[i])
        edge_angle = [angle / max_angle for angle in edge_angle]
        cur_edge_angle = cur_edge_angle / max_angle

        final_position_encoding = [item / max(position_encoding) for item in position_encoding]

        phase = []
        green = []
        action = []

        if len(actions) < 1:
            actions = [0] * 16
        for i, l, g, a in zip(light_count, traffic_light_state, greens, actions):
            phase.extend([l] * i)
            green.extend([g] * i)
            action.extend([a] * i)

        _state = list(zip(edge_average_speed_result, edge_density_result, final_edge_end_distance,
                          current_mask, destination_mask, final_edge_travel_time, edge_angle, final_position_encoding,
                          phase, green, action))
        state = [i for _s in _state for i in _s]

        veh_feature = [cur_velocity] + cur_densities + [veh_dis] + [cur_edge_angle] + cur_time
        state.extend(veh_feature)

        # self.cav_state = state
        self.done = traci.vehicle.getRoadID(self.veh_id) == self.dest
        if self.done:
            cost = traci.simulation.getTime() - self.depart_time
            self.termination_reward = (math.exp(self.default_travel_distance / 11.11 / cost) - math.exp(0)) * 100
            self.termination_reward = -5 * (traci.simulation.getTime() - self.depart_time) + 2820
        # self.done = False

        return state
