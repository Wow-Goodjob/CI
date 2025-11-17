from agent import CAVAgent
from net import DQN
import traci
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import env
import config
import copy
import networkx as nx
from networkx.algorithms.simple_paths import shortest_simple_paths
from net import DQN


k_map = dict()


class DSOQNet(DQN):
    def __init__(self, state_dim, action_dim, args, agent_num, tl_id):
        super(DSOQNet, self).__init__(state_dim, action_dim, args, agent_num, tl_id)

        #self.n_road=72
        #self.n_intersection=16

        self.emb_node = nn.Embedding(192, 64)
        #self.fc1 = nn.Linear(2315, 128)
        #self.fc2 = nn.Linear(128, 64)
        #self.fc3 = nn.Linear(64, action_dim)

        #self.fc1 = nn.Linear(2315, 32)
        #self.fc2 = nn.Linear(32, 32)
        #self.fc3 = nn.Linear(32, action_dim)

        #self.bn1 = torch.nn.BatchNorm1d(64)
        #self.bn2 = torch.nn.BatchNorm1d(64)

    def forward(self, state, bl=False):

        B, LD = state.shape
        road = self.road_emb(
            state[:, :-1].reshape(B, self.n_intersection, self.n_road))  # B*N*F -> B*N*emb
        cav = self.emb_node(state[:, -1].long()).unsqueeze(dim=1)  # B*1*F' -> B*1*Emb
        att = self.att(road, road, cav).squeeze(dim=1)  # K, V, Q
        x0 = self.out_layer(att)

        x1 = F.relu(self.fc1(x0))
        x2 = self.fc2(x1)
        # if x2.requires_grad:
        #     g = make_dot(x2)
        #     g.view()
        return x2

        # global_state = state[:, :-1]
        # node_info = state[:, -1].int()
        #
        # x = F.relu(self.fc1(global_state))
        # x = F.relu(self.fc2(x))
        #
        # y = self.emb(node_info)
        #
        # x = self.fc3(torch.cat((y, x), dim=1))

        # x = F.relu(self.fc1(state))
        # x = F.relu((self.fc2(x)))
        # x = self.fc3(x)
        #
        # return x


def build_graph():
    G = nx.DiGraph()
    for edge_id in traci.edge.getIDList():
        if edge_id[0] == ':':
            continue
        from_node = traci.edge.getFromJunction(edge_id)
        to_node = traci.edge.getToJunction(edge_id)
        length = traci.lane.getLength(edge_id+'_0')
        G.add_edge(from_node, to_node, edge=edge_id, weight=length)  # 以距离为权重
    return G


def yen_k_shortest_paths(G, source, target, K=5):
    paths = list(shortest_simple_paths(G, source, target, weight="weight"))
    return paths[:K]  # 取前 K 条最短路径


class DSOAgent(CAVAgent):
    def __init__(self, veh_id, router, adj_edge, args):
        super(DSOAgent, self).__init__(veh_id, router, adj_edge, args)
        # self.graph = build_graph()

    # def get_avail_action(self):
    #
    #     edge = traci.vehicle.getRoadID(self.veh_id)
    #     mapping = {"r": 0, "s": 1, "l": 2, "t": 3}
    #     if self.args.direction == 3:
    #         mapping = {"r": 0, "s": 1, "l": 2}
    #     action_mask = [0 for _ in range(4)]
    #     if self.args.direction == 3:
    #         action_mask = [0 for _ in range(3)]
    #     next_edges = self.adj_edge[edge]
    #
    #     directions = list(next_edges.keys())
    #     # default_exp_time = sum([traci.edge.getTraveltime(k) for k in list(traci.simulation.findRoute(edge, self.dest).edges)])
    #
    #     if self.dest in self.adj_edge[edge].values():
    #         _edge = {v: k for k, v in self.adj_edge[edge].items()}
    #         action_mask[mapping[_edge[self.dest]]] = 1
    #         return action_mask
    #
    #     source = traci.edge.getFromJunction(traci.vehicle.getRoadID(self.veh_id))
    #     target = traci.edge.getFromJunction(self.dest)
    #
    #     if f'{source}_{target}' in k_map:
    #         K = k_map[f'{source}_{target}']
    #     else:
    #         K = yen_k_shortest_paths(self.graph, source=source, target=target, K=3)
    #         k_map[f'{source}_{target}'] = K
    #
    #     for direction in mapping.keys():
    #         for _k in K:
    #             if len(_k) >= 3 and _k[2] == traci.edge.getToJunction(next_edges[direction]):
    #                 action_mask[mapping[direction]] = 1
    #     if sum(action_mask) == 0:
    #         next_edge = traci.simulation.findRoute(edge, self.dest).edges[1]
    #         _edge = {v: k for k, v in self.adj_edge[edge].items()}
    #         action_mask[mapping[_edge[next_edge]]] = 1
    #
    #     assert sum(action_mask) > 0
    #     return action_mask

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
                          current_mask, destination_mask, final_edge_travel_time))
        state = [i for _s in _state for i in _s]
        # l = len(state)
        # state = state + [0] * (self.args.agent_num * self.args.road_feature * 4 * 3 - l)

        # veh_feature = [cur_velocity] + cur_densities + [veh_dis] + [cur_edge_angle] + cur_time + \
        #               [current_x / max(sx_list), current_y / max(sy_list)]
        # state.extend(veh_feature)

        global_state = [current_mask.index(1) if sum(current_mask) > 0 else 0]
        state.extend(global_state)
        # l = len(state)
        # state = state + [0] * (self.args.agent_num * self.args.road_feature * 4 * 3 - l)

        # veh_feature = [cur_velocity] + cur_densities + [veh_dis] + [cur_edge_angle] + cur_time + \
        #               [current_x / max(sx_list), current_y / max(sy_list)]
        # state.extend(veh_feature)

        # self.cav_state = state
        self.done = (self.dest in self.adj_edge[cur_edge].values())
        if self.done:
            # cost = traci.simulation.getTime() - self.depart_time
            # self.termination_reward = (math.exp(self.default_travel_distance / 11.11 / cost) - math.exp(0)) * 300
            cur_travel_distance = sum([traci.lane.getLength(k + '_0') for k in traci.vehicle.getRoute(self.veh_id)])
            self.termination_reward = self.default_travel_distance / cur_travel_distance * 100

        # self.done = False
        #state_array=np.array(state,dtype=np.float32)
        #print(f"状态维度:{state_array.shape},状态长度:{len(state)}")
        expected_length=2305
        if len(state)<expected_length:
            padding_length=expected_length-len(state)
            state.extend([0.0]*padding_length)
        return state

    # def get_router_state2(self, road_state=None, traffic_light_state=None, road_list=None, greens=None, light_count=None,
    #                       lane_dict=None):
    #     # list stores attribute v_{avr}
    #     edge_average_speed = []
    #     # list stores attribute d_{veh}
    #     edge_density = []
    #     # list stores attribute t_{avr}
    #     edge_travel_time = []
    #     # list stores attribute dis_{aim}
    #     edge_end_distance = []
    #     # list stores attribute position encoding P
    #     position_encoding = []
    #     # list stores 0/1 (destination: 1  else: 0)
    #     destination_mask = []
    #     # current lane 0/1
    #     current_mask = []
    #     # edge_angle
    #     edge_angle = []
    #
    #     waiting_vehicles = []
    #     vehicle_count = []
    #     phase = []
    #     green = []
    #
    #     # obtain the current position of the rl vehicle
    #     current_x, current_y = traci.vehicle.getPosition(self.veh_id)
    #     cur_lane = traci.vehicle.getLaneID(self.veh_id)
    #
    #     _road_state = copy.deepcopy(road_state)
    #
    #     # for k, v in _road_state.items():
    #     #     nx, ny = _road_state[k]['ex'], _road_state[k]['ey']
    #     #     cx, cy = current_x, current_y
    #     #     dn = ((nx - self.dest_x) ** 2 + (ny - self.dest_y) ** 2) ** 0.5
    #     #     dc = ((cx - self.dest_x) ** 2 + (cy - self.dest_y) ** 2) ** 0.5
    #     #     if dn > dc and self.dest not in k:
    #     #         del road_state[k]
    #
    #     # obtain the position encoding of the edges
    #     start_positions = dict()
    #     end_positions = dict()
    #     distance_approximation = dict()
    #
    #     for d in road_state:
    #         start_positions[d] = ((road_state[d]["sx"] - current_x) ** 2 + (
    #                     road_state[d]["sy"] - current_y) ** 2) ** 0.5
    #         end_positions[d] = ((road_state[d]["ex"] - self.dest_x) ** 2 + (
    #                     road_state[d]["ey"] - self.dest_y) ** 2) ** 0.5
    #         distance_approximation[d] = start_positions[d] + end_positions[d] + road_state[d]["length"]
    #
    #     # sort the attributes of all edges according to the distance between the rl vehicle and the destination
    #     # sorted_positions = sorted(distance_approximation.items(), key=lambda x: x[1], reverse=False)
    #
    #     sorted_road_list = [sorted(road_list[i:i + 12]) for i in range(0, len(road_list), 12)]
    #     flattened_sorted_list = [item for sublist in sorted_road_list for item in sublist]
    #
    #     sx_list = []
    #     sy_list = []
    #
    #     for k in flattened_sorted_list:
    #         # if k not in road_state:
    #         #     position_encoding.append(0)
    #         #     edge_average_speed.append(0)
    #         #     edge_density.append(0)
    #         #     edge_travel_time.append(0)
    #         #     edge_angle.append(0)
    #         #     edge_end_distance.append(0)
    #         #     current_mask.append(0)
    #         #     destination_mask.append(0)
    #         #     phase.append(0)
    #         #     green.append(0)
    #         #     sx_list.append(0)
    #         #     sy_list.append(0)
    #         #     continue
    #
    #         position_encoding.append(distance_approximation[k])
    #         edge_average_speed.append(road_state[k]["velocity"])
    #         edge_density.append(road_state[k]["density"])
    #         edge_travel_time.append(road_state[k]["time"])
    #         cx = road_state[k]["sx"]
    #         cy = road_state[k]["sy"]
    #
    #         waiting_vehicles.append(traci.lane.getLastStepHaltingNumber(k))
    #         vehicle_count.append(road_state[k]['number'])
    #
    #         sx_list.append(cx)
    #         sy_list.append(cy)
    #
    #         if cx - current_x == 0:
    #             edge_angle.append(np.pi / 2)
    #         else:
    #             edge_angle.append(np.arctan((cy - current_y) / (cx - current_x)))
    #
    #         distance = ((cx - self.dest_x) ** 2 + (cy - self.dest_y) ** 2) ** 0.5
    #         edge_end_distance.append(distance)
    #         if lane_dict[k] == lane_dict[cur_lane]:
    #             current_mask.append(1)
    #         else:
    #             current_mask.append(0)
    #         if lane_dict[k] == lane_dict[self.dest + '_0']:
    #             destination_mask.append(1)
    #         else:
    #             destination_mask.append(0)
    #         phase.append(traci.trafficlight.getPhase(lane_dict[k]))
    #         green.append(35 - int(traci.simulation.getTime()) % 35)
    #
    #     cur_edge = traci.vehicle.getRoadID(self.veh_id)
    #
    #     cur_velocity = traci.vehicle.getSpeed(self.veh_id)  # V
    #     cur_numbers = [traci.lane.getLastStepVehicleNumber(f'{cur_edge}_{i}') for i in range(3)]  # N_i
    #     cur_lengths = [traci.lane.getLength(f'{cur_edge}_{i}') for i in range(3)]  # L_i
    #     cur_densities = [a / b for a, b in zip(cur_numbers, cur_lengths)]  # D_i
    #     cur_time = [traci.lane.getTraveltime(f'{cur_edge}_{i}') for i in range(3)]  # ET_i
    #     current_x, current_y = traci.vehicle.getPosition(self.veh_id)  # c_x, c_y
    #     s, e = traci.lane.getShape(self.dest + '_1')  # d_x, d_y
    #     des_sx = s[0]
    #     des_sy = s[1]
    #     veh_dis = ((current_x - des_sx) ** 2 + (current_y - des_sy) ** 2) ** 0.5
    #     if des_sx - current_x == 0:
    #         cur_edge_angle = np.pi / 2
    #     else:
    #         cur_edge_angle = np.arctan((des_sy - current_y) / (des_sx - current_x))
    #
    #     # normalize all the input attributes
    #     if max(edge_average_speed) == 0:
    #         edge_average_speed_result = edge_average_speed
    #     else:
    #         edge_average_speed_result = [speed / max(edge_average_speed)
    #                                      for speed in edge_average_speed]
    #         cur_velocity = cur_velocity / max(edge_average_speed)
    #
    #     if max(edge_density) == 0:
    #         edge_density_result = edge_density
    #     else:
    #         edge_density_result = [density / max(edge_density)
    #                                for density in edge_density]
    #         cur_densities = [density / max(edge_density)
    #                          for density in cur_densities]
    #
    #     if max(edge_end_distance) == 0:
    #         final_edge_end_distance = edge_end_distance
    #     else:
    #         final_edge_end_distance = [distance / max(edge_end_distance)
    #                                    for distance in edge_end_distance]
    #         veh_dis = veh_dis / max(edge_end_distance)
    #
    #     if max(edge_travel_time) == 0:
    #         final_edge_travel_time = edge_travel_time
    #     else:
    #         final_edge_travel_time = [time / max(edge_travel_time)
    #                                   for time in edge_travel_time]
    #         cur_time = [time / max(edge_travel_time)
    #                     for time in cur_time]
    #
    #     if max(vehicle_count) == 0:
    #         final_vehicle_count = vehicle_count
    #     else:
    #         final_vehicle_count = [number / max(vehicle_count) for number in vehicle_count]
    #
    #     if max(waiting_vehicles) == 0:
    #         final_waiting_vehicles = waiting_vehicles
    #     else:
    #         final_waiting_vehicles = [number / max(waiting_vehicles) for number in waiting_vehicles]
    #
    #     max_angle = 0
    #     for i in range(len(edge_angle)):
    #         if abs(edge_angle[i]) > max_angle:
    #             max_angle = abs(edge_angle[i])
    #     edge_angle = [angle / max_angle for angle in edge_angle]
    #     cur_edge_angle = cur_edge_angle / max_angle
    #
    #     final_position_encoding = [item / max(position_encoding) for item in position_encoding]
    #
    #     final_sx_list = [x / max(sx_list) for x in sx_list]
    #     final_sy_list = [y / max(sy_list) for y in sy_list]
    #
    #     # action = []
    #
    #     # if len(actions) < 1:
    #     #     actions = [0] * 16
    #     # for i, l, g, a in zip(light_count, traffic_light_state, greens, actions):
    #     #     phase.extend([l] * i)
    #     #     green.extend([g] * i)
    #     #     action.extend([a] * i)
    #
    #     _state = list(zip(final_edge_travel_time, edge_density_result, final_waiting_vehicles, final_vehicle_count))
    #     global_state = [i for _s in _state for i in _s]
    #     state = [current_mask.index(1) if sum(current_mask) > 0 else 0]
    #     state.extend(global_state)
    #     # l = len(state)
    #     # state = state + [0] * (self.args.agent_num * self.args.road_feature * 4 * 3 - l)
    #
    #     # veh_feature = [cur_velocity] + cur_densities + [veh_dis] + [cur_edge_angle] + cur_time + \
    #     #               [current_x / max(sx_list), current_y / max(sy_list)]
    #     # state.extend(veh_feature)
    #
    #     # self.cav_state = state
    #     # self.done = (self.dest in self.adj_edge[cur_edge].values())
    #     # if self.done:
    #     #     # cost = traci.simulation.getTime() - self.depart_time
    #     #     # self.termination_reward = (math.exp(self.default_travel_distance / 11.11 / cost) - math.exp(0)) * 300
    #     #     cur_travel_distance = sum([traci.lane.getLength(k + '_0') for k in traci.vehicle.getRoute(self.veh_id)])
    #     #     self.termination_reward = self.default_travel_distance / cur_travel_distance * 100
    #
    #     # self.done = False
    #
    #     return state

    # def get_reward(self, road_state=None):
    #     self.penalty = 0
    #     dt = traci.simulation.getTime() - self.act_time + self.penalty
    #     return -dt
