import math
import os
import sys
import numpy as np
import time
import torch
import torch.nn as nn
import traci

import traci as tc
from sumolib import checkBinary

import config
import traffic_light as tl

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_context(agent_list):
    for agent in agent_list:
        tl_id = agent.tl_id
        tc.junction.subscribeContext(tl_id, tc.constants.CMD_GET_VEHICLE_VARIABLE, 300,
                                     [tc.constants.VAR_LANEPOSITION, tc.constants.VAR_SPEED, tc.constants.VAR_LANE_ID])
    lane_list = tc.lane.getIDList()
    for lane in lane_list:
        tc.lane.subscribe(lane, [0x14, 0x10, 0x12])  # halting counting carList
    # last step halting number (0x14)
    # last step vehicle number (0x10)


def set_phase(agent, index):
    tl_id = agent.get_id()
    tc.trafficlight.setPhase(tl_id, index)


def step(sec=0):
    tc.simulationStep(step=sec)


def close():
    tc.close()


def start(execution=False, path=None, seed='0', gui=None):  # 1444 0 42 0 0
    # environment variable
    # if 'SUMO_HOME' in os.environ:
    #     tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    #     sys.path.append(tools)
    # else:
    #     sys.exit("please declare environment variable 'SUMO_HOME'")

    binary = checkBinary(config.SIMULATION["RUN"] if gui is None else "sumo-gui")
    # th_pool = []
    # for th in range(1, 10):
    assert path is not None
    _th = tc.start([binary, "-c", path, "--tripinfo-output", "./tripinfos.xml",
                    "--seed", str(seed), "--duration-log.statistics", "--threads", "10",
                    "--routing-algorithm", "astar",
                    "--device.emissions.probability", "1.0",
                    "--queue-output", "./queues.xml",
                    "--queue-output.period", "3.0"])
    #     th_pool.append(_th)
    # return th_pool


def create_lane_to_det():
    dic = dict()
    det_list = tc.lanearea.getIDList()
    for i in range(len(det_list)):
        dic[tc.lanearea.getLaneID(det_list[i])] = det_list[i]
    lane_list = tc.lane.getIDList()
    for lane in lane_list:
        if lane not in dic:
            dic[lane] = "e2det_" + lane
    return dic


# divide lane into certain segments(bins)
# counting vehicle number on each segment
# return list(size=K)
def lane_position_mapper(lane_pos, bins):
    lane_pos_np = np.array(lane_pos)
    digitized = np.digitize(lane_pos_np, bins)
    position_counter = [len(lane_pos_np[digitized == i]) for i in range(1, len(bins))]
    return position_counter


# PressLight: P = x(l)/x_max(l)
def get_agent_relative_vehicles(agent, div=False):
    dic = agent.get_dic()
    lanes = get_control_lanes(agent)
    vehicles = [0] * len(lanes)
    for i in range(len(lanes)):
        vehicles[i] = l_get_number_of_waiting_vehicles(dic[lanes[i]], agent.tl_id, div=div) \
                      / get_lane_length(lanes[i]) * 7.5
    return vehicles, lanes


def get_agent_vehicles(agent, div=True):
    dic = agent.get_dic()
    lanes = get_control_lanes(agent)
    vehicles = [0] * len(lanes)
    for i in range(len(lanes)):
        vehicles[i] = l_get_number_of_waiting_vehicles(dic[lanes[i]], agent.tl_id, div=div)
        # times = times + get_waiting_time(dic[lanes[i]])
    # print(vehicles)
    return vehicles, lanes


def get_agent_segment_vehicles(agent, div=True):
    lanes = list(set(get_control_lanes(agent)))
    veh_count = []
    for i in range(len(lanes)):
        veh_lists = l_get_lane_vehicle_list(lanes[i], agent.tl_id, div=div)
        pos_list = []  # car position on current lane
        for veh in veh_lists:
            pos_list.append(get_vehicle_position(veh))
        lane_len = get_lane_length(lanes[i])
        cut_point = np.linspace(0, lane_len, 4).tolist()  # divide lane into 3 pieces according to length
        veh_count.extend(lane_position_mapper(pos_list, cut_point))
    # veh_count = agent.state_emb(veh_count)
    # veh_emb_list = []
    # for i in range(0, len(veh_count)):
    #     veh_emb_list.append(veh_count[i][0])
    #     veh_emb_list.append(veh_count[i][1])
    return veh_count  # N*3*1


def get_global_map_state(agent_list):
    res = []
    for agent in agent_list:
        res.append(get_map_state(agent))
    return res


AREA_LENGTH = 1024
GRID_WIDTH = 8


def get_map_state(agent):
    lanes = get_control_lanes(agent)
    vehicle_id_list = []
    for lane in lanes:
        vehicle_id_list.extend(l_get_lane_vehicle_list(lane, agent.tl_id))

    area_length = AREA_LENGTH
    grid_width = GRID_WIDTH
    length_num_grids = int(area_length / grid_width)
    mapOfCars = np.zeros((length_num_grids, length_num_grids))
    junction_position = tc.junction.getPosition(agent.tl_id)

    for vehicle_id in vehicle_id_list:
        vehicle_position = tc.vehicle.getPosition(vehicle_id)  # (double,double),tuple
        norm_pos = (vehicle_position[0] - junction_position[0] + area_length / 2,
                    vehicle_position[1] - junction_position[1] + area_length / 2)
        transform_tuple = vehicle_location_mapper(norm_pos)  # call the function
        mapOfCars[transform_tuple[0], transform_tuple[1]] += 1
        # print(transform_tuple, vehicle_position, junction_position)
        assert transform_tuple[0] >= 0 and transform_tuple[1] >= 0
    return mapOfCars


def vehicle_location_mapper(coordinate, area_length=AREA_LENGTH, area_width=600):
    grid_width = GRID_WIDTH
    transformX = math.floor(coordinate[0] / grid_width)
    transformY = math.floor((area_length - coordinate[1]) / grid_width)
    length_num_grids = int(area_length / grid_width)
    transformY = length_num_grids - 1 if transformY == length_num_grids else transformY
    transformX = length_num_grids - 1 if transformX == length_num_grids else transformX
    tempTransformTuple = (transformY, transformX)
    return tempTransformTuple


def get_length_state(agent, green=-1):
    lanes = list(set(tc.trafficlight.getControlledLanes(agent.tl_id)))
    lanes.sort()
    veh_count = []
    for lane in lanes:
        veh_lists = tc.lane.getLastStepVehicleIDs(lane)
        pos_list = []  # car position on current lane
        for veh in veh_lists:
            pos_list.append(get_vehicle_position(veh))
        lane_len = tc.lane.getLength(lane)
        cut_point = np.linspace(0, lane_len, 4).tolist()  # divide lane into 3 pieces according to length
        veh_count.extend(lane_position_mapper(pos_list, cut_point))
    if agent.args.state_contain_action:
        veh_count.append(agent.rl_model.action)
    if agent.args.state_contain_phase:
        veh_count.append(tc.trafficlight.getPhase(agent.tl_id) // 2)
    if agent.args.state_contain_phase_duration:
        veh_count.append(green)
    return veh_count  # N*3*1 + 1


def get_state(agent, state_type="independent", agent_list=None):
    if state_type == "global":
        ret = []
        for agent in agent_list:
            ret.extend(get_agent_segment_vehicles(agent, div=True))
            ret.extend(agent.get_action()[0])
        return ret
    # counting vehicles 1. total lane; 2. segment piece
    # ret, _ = get_agent_vehicles(agent, div=True)
    ret = get_agent_segment_vehicles(agent, div=True)
    if agent.args.state_contain_action:
        ret.extend(agent.get_action()[0])
    # local state
    if state_type == "local" or state_type == "global":
        # for each neighbor intersection
        for i in range(config.COP[agent.get_id()]):
            # t, _ = get_agent_vehicles(agent.neighbors[i], div=True)
            t = get_agent_segment_vehicles(agent.neighbors[i], div=True)
            ret.extend(t)
            if agent.args.state_contain_action:
                ret.extend(agent.neighbors[i].get_action()[0])
    if state_type == "global":
        ret += get_other_state(agent)
    ret = np.array(ret)
    if config.NORM['STATE_NORM_CLIP']:
        ret = norm_clip_state(ret)
    return ret


def norm_clip_state(state):
    state /= config.NORM['NORM_STATE']
    return state if config.NORM['CLIP_STATE'] < 0 else np.clip(state, 0, config.NORM['CLIP_STATE'])


def get_other_state(agent):
    ret = []
    for agent in agent._light:
        t, _ = get_agent_vehicles(agent)
        ret.extend(t)
        if agent.args.state_contain_action:
            ret.extend(agent.get_action()[0])
    return ret


# for Q-mix
def get_global_state(agent_list):
    ret = []
    for i, agent in enumerate(agent_list):
        ret.extend(get_agent_segment_vehicles(agent, div=True))
        if agent.args.state_contain_action:
            ret.extend(agent.get_action()[0])
        if agent.args.agent_type in {"MAPPO", "CenPPO", "PS-PPO", "MAT", "MA2C"} and agent.args.state_contain_agent_id:
            ret.extend([i + 1])
    return ret


def get_global_reward(agent_list):
    c = 0
    r = 0
    for agent in agent_list:
        _c, _r = get_reward(agent)
        c += _c
        r += _r
    return c, r


def get_length_reward(agent):
    length = 0
    upstream = list(set(tc.trafficlight.getControlledLanes(agent.tl_id)))
    for lane in upstream:
        length = length + tc.lane.getLastStepVehicleNumber(lane)
    return -length


def get_pressure_reward(agent):
    pressure = 0
    upstream = list(set(tc.trafficlight.getControlledLanes(agent.tl_id)))
    for lane in upstream:
        pressure = pressure + tc.lane.getLastStepVehicleNumber(lane) / tc.lane.getLength(lane) * 7.5
    for lane in agent.dic_down:
        pressure = pressure - tc.lane.getLastStepVehicleNumber(lane) / tc.lane.getLength(lane) * 7.5
    return -pressure


# return waiting number and reward
def get_reward(agent):
    # get sum length
    if agent.args.reward == "pressure":
        v, _ = get_agent_relative_vehicles(agent, div=False)
    else:
        v, _ = get_agent_vehicles(agent, div=True)
    length = 0
    for i in range(len(v)):
        length = length + v[i]

    agent.set_waiting(length)

    # get waiting
    if agent.args.reward == "waiting":
        waiting = get_intersection_waiting(agent)
        return length, -waiting

    elif agent.args.reward == "first_waiting":
        waiting = get_intersection_first_waiting(agent)
        return length, -waiting

        # get pressure
    pressure = length
    if "pressure" in agent.args.reward:
        for lane in agent.dic_down:
            pressure = pressure - l_get_number_of_waiting_vehicles(agent.dic[lane], agent.tl_id, div=False) \
                       / get_lane_length(agent.dic[lane][6:]) * 7.5

    # lane group reward
    lane_length = 0
    lane_pressure = 0
    op_lane_pressure = 0.0
    threshold = 46
    if agent.args.reward == "lane_pressure" or agent.args.reward == "lane_length":
        for d in agent.dic_lane:
            entering = l_get_number_of_waiting_vehicles(agent.dic[d], agent.tl_id)
            exiting = l_get_number_of_waiting_vehicles(agent.dic[agent.dic_lane[d]], agent.tl_id)
            # get lane length
            lane_length = lane_length + entering
            # get lane pressure
            lane_pressure = lane_pressure + entering - exiting
            # parameterized pressure
            op_lane_pressure = op_lane_pressure + (entering - exiting) / threshold

    if agent.args.reward == "length":
        if config.NORM['REWARD_NORM_CLIP']:
            if config.NORM['CLIP_REWARD'] > 0:
                pass
            if config.NORM['NORM_REWARD'] > 0:
                return length, [-l / config.NORM['NORM_REWARD'] for l in length]
        return length, -length
    if "pressure" in agent.args.reward:
        return length, -pressure
    if agent.args.reward == "lane_length":
        return length, -lane_length
    if agent.args.reward == "lane_pressure":
        return length, -op_lane_pressure


def norm_clip_reward(reward):
    return reward


def get_distance(agent1, agent2):
    tl_1 = agent1.get_id()
    tl_2 = agent2.get_id()
    p_1 = get_tl_pos(tl_1)
    p_2 = get_tl_pos(tl_2)
    return math.sqrt((p_1[0] - p_2[0]) ** 2 + (p_1[1] - p_2[1]) ** 2)


def get_intersection_waiting(agent):
    lanes = get_control_lanes(agent)
    waiting = 0
    for lane in lanes:
        waiting += get_waiting_time(lane)
    return np.float32(waiting)


def get_waiting_time(lane_id):
    t = 0
    vehicles = get_lane_waiting_list(lane_id)
    for v in vehicles:
        t = t + get_vehicle_waiting_time(v)
    return t


def get_first_waiting_time(lane_id):
    t = 0
    vehicles = get_lane_waiting_list(lane_id)
    max_pos = 0
    for v in vehicles:
        car_pos = tc.vehicle.getLanePosition(v)
        if car_pos > max_pos:
            max_pos = car_pos
            t = tc.vehicle.getWaitingTime(v)
    return t


def get_downstream(tl_id, spec=None, city_map="xuancheng"):
    lanes = []
    if city_map == "xuancheng":
        for i in range(3):
            for j in config.NEIGHBOR[tl_id]:
                if spec is not None and "segment_pressure" in spec and len(j) < 2:
                    lane = (tl_id[4] + '_' + j + '_o_' + str(i))
                else:
                    lane = (tl_id[4] + '_' + j + '_' + str(i))
                lanes.append(lane)
    elif city_map in {"hangzhou", "jinan"}:
        _, col, row = tl_id.split('_')
        for i in range(4):
            for j in range(3):
                lanes.append(f'road_{col}_{row}_{i}_{j}')
    return lanes


def get_upstream(tl_id, spec=""):
    lanes = []
    for i in range(3):
        for j in config.NEIGHBOR[tl_id]:
            if "segment_pressure" in spec and len(j) < 2:
                lane = (j + '_' + tl_id[4] + '_i_' + str(i))
            else:
                lane = (j + '_' + tl_id[4] + '_' + str(i))
            lanes.append(lane)
    return lanes


def get_lane_map(tl_id, spec=""):
    dic = dict()
    for i in range(len(config.NEIGHBOR[tl_id])):
        for j in range(3):
            if "segment_pressure" in spec:
                dic[(config.NEIGHBOR[tl_id][i] + '_' + tl_id[4] + '_' + str(j))] = \
                    (tl_id[4] + '_' + config.NEIGHBOR[tl_id][(i + j + 1) % 4] + '_' + str(j))
            else:
                dic[(config.NEIGHBOR[tl_id][i] + '_' + tl_id[4] + '_' + str(j))] = \
                    (tl_id[4] + '_' + config.NEIGHBOR[tl_id][(i + j + 1) % 4] + '_' + str(j))
    return dic


def get_far_agent(tl_id, agent_list):
    ret = []
    cur = int(tl_id[4]) - 1
    for i in range(len(agent_list)):
        if config.COP["ADJACENT"][cur][i] == 1:
            continue
        ret.append(agent_list[i])
    return ret


def get_vehicle_waiting_time(veh_id):
    return tc.vehicle.getWaitingTime(veh_id)


def get_lane_waiting_list(lane_id):
    return tc.lane.getLastStepVehicleIDs(lane_id)


def get_waiting_list(det_id):
    return tc.lanearea.getLastStepVehicleIDs(det_id)


def get_tl_pos(tl_id):
    return tc.junction.getPosition(tl_id)


def l_get_lane_vehicle_list(lane_id, tl_id=None, div=True):
    lane_dc = tc.lane.getSubscriptionResults(lane_id)
    lane_len = tc.lane.getLength(lane_id)
    if len(lane_dc) == 0:
        return []
    else:
        return lane_dc[0x12]  # 0x12 vehicle list on the lane


def get_intersection_waiting_vehicles(agent):
    length = 0
    for lane in get_control_lanes(agent):
        length = l_get_number_of_waiting_vehicles("e2det_" + lane, div=False)
    return length


# get vehicle number with lane subscription
def l_get_number_of_waiting_vehicles(det_id, tl_id=None, div=True):
    lane_id = det_id[6:]
    lane_dc = tc.lane.getSubscriptionResults(lane_id)
    lane_len = tc.lane.getLength(lane_id)
    if len(lane_dc) == 0:
        return np.float32(0)
    else:
        if div:
            return np.float32(lane_dc[0x10]) / lane_len  # 0x10 vehicle 0x14 halting vehicle
        else:
            return np.float32(lane_dc[0x10])


def get_number_of_queued_vehicles(lane_id, div=False):
    return tc.lane.getLastStepHaltingNumber(lane_id)


# c_function: get traffic metrics with subscribed CONTEXT
def c_get_number_of_waiting_vehicles(det_id, tl_id):
    lane_id = det_id[6:]
    cnt = 0
    jun_dc = tc.junction.getContextSubscriptionResults(tl_id)
    if len(jun_dc) == 0:
        return np.float32(0)
    else:
        for v in jun_dc:
            if jun_dc[v][tc.constants.VAR_LANE_ID] == lane_id:
                cnt += 1
    return np.float32(cnt)


def l_get_number_of_halting_vehicles(lane_id, div=True):
    lane_dc = tc.lane.getSubscriptionResults(lane_id)
    lane_len = tc.lane.getLength(lane_id)
    if len(lane_dc) == 0:
        return np.float32(0)
    else:
        if div:
            return np.float32(lane_dc[0x14]) / lane_len  # 0x10 vehicle 0x14 halting vehicle
        else:
            return np.float32(lane_dc[0x14])


def get_lane_length(lane_id):
    return tc.lane.getLength(lane_id)


def get_vehicle_position(veh_id):
    return tc.vehicle.getLanePosition(veh_id)


def get_number_of_waiting_vehicles(det_id, tl_id=None):
    return np.float32(tc.lanearea.getLastStepVehicleNumber(det_id))


def get_control_lanes(agent):
    tl_id = agent.get_id()
    return tc.trafficlight.getControlledLanes(tl_id)


def get_tl_list():
    tl_list = []
    for light in tc.trafficlight.getIDList():
        tl_list.append(light)
    return tl_list


def get_arrived():
    return tc.simulation.getArrivedNumber()


def get_loaded():
    return tc.simulation.getLoadedNumber()


def get_time():
    return tc.simulation.getTime()


def get_phase(tl_id):
    return tc.trafficlight.getPhase(tl_id) // 2


def get_pressure(d1, d2, agent):
    return l_get_number_of_waiting_vehicles(d1, agent.tl_id, div=False) - l_get_number_of_waiting_vehicles(d2,
                                                                                                           agent.tl_id,
                                                                                                           div=False)
    # return c_get_number_of_waiting_vehicles(d1, agent.tl_id) - c_get_number_of_waiting_vehicles(d2, agent.tl_id)


def get_ma2c_state(agent_list):
    arr = []
    for i, agent in enumerate(agent_list):
        # queue
        t = get_agent_segment_vehicles(agent, div=True)

        # waiting
        # for ild in agent.dic_up:
        #     max_pos = 0
        #     car_wait = 0
        #     cur_cars = tc.lane.getLastStepVehicleIDs(ild)
        #     for vid in cur_cars:
        #         car_pos = tc.vehicle.getLanePosition(vid)
        #         if car_pos > max_pos:
        #             max_pos = car_pos
        #             car_wait = tc.vehicle.getWaitingTime(vid)
        #     t.extend([car_wait])

        # action
        if agent.args.state_contain_action:
            t.extend(agent.get_action()[0])

        # id
        if agent.args.agent_type in {"MAPPO", "CenPPO", "PS-PPO", "MAT", "MA2C"} and agent.args.state_contain_agent_id:
            t.extend([i + 1])

        arr.append(t)
    ret = np.array(arr)
    return ret


def get_intersection_first_waiting(agent):
    lanes = get_control_lanes(agent)
    waiting = 0
    for lane in lanes:
        waiting += get_first_waiting_time(lane)
    return np.float32(waiting)


def get_ma2c_reward(agent_list):
    rewards = []
    for agent in agent_list:
        _, r = get_reward(agent)
        rewards.append(r)
        # q = 0
        # for ild in agent.dic_up:
        #     q += l_get_number_of_halting_vehicles(ild)
        # waiting = get_intersection_first_waiting(agent)
        # rewards.append(-q-waiting)

    return np.array(rewards)


def get_all_actions(agent_list):
    actions = []
    for agent in agent_list:
        a = [0] * 4
        a[int(agent.action)] = 1
        actions.extend(a)
    return actions


def get_ccgn_state(agent_list):
    states = []
    for agent in agent_list:
        state = []
        # 1 ont-hot phase
        a = [0] * 4
        a[int(agent.action)] = 1
        state.extend(a)  # 4

        # lane length
        lane_length = 0
        for lane in get_control_lanes(agent):
            lane_length += get_lane_length(lane)
        lane_length /= len(get_control_lanes(agent))
        state.extend([lane_length])  # 1

        # distance from prev
        dis = lane_length
        state.extend([dis])  # 1

        # 3 real time speed
        real_time_speed = 0
        for lane in get_control_lanes(agent):
            real_time_speed += tc.lane.getLastStepMeanSpeed(lane)
        real_time_speed /= len(get_control_lanes(agent))
        state.extend([real_time_speed])  # 1
        agent.his_speed.append(real_time_speed)

        # speed
        avg_speed = agent.his_speed[-1] + agent.his_speed[-1] - agent.his_speed[-2] \
            if len(agent.his_speed) > 1 else agent.his_speed[-1]
        state.extend([avg_speed])  # 1

        # 2 flow pos/location P_in
        state.extend([lane_length - 15 * avg_speed])  # 1

        # vehicle count
        vehicle_count = 0
        for lane in get_control_lanes(agent):
            vehicle_count += l_get_number_of_waiting_vehicles("e2det_" + lane)
        state.extend([vehicle_count])  # 1

        # waiting time
        wait = get_intersection_waiting(agent)
        state.extend([wait])  # 1

        states.append(state)

    return states


def get_ccgn_global_reward(agent_list):
    rewards = 0
    for agent in agent_list:
        rewards += get_ccgn_local_reward(agent, agent_list)
    return rewards


def get_ccgn_local_reward(agent, agent_list):
    reward = 0
    for i in range(len(agent_list)):
        if config.COP["ADJACENT"][int(agent.get_id()[4]) - 1][i] == "1":
            reward -= get_intersection_waiting_vehicles(agent_list[i])
            reward -= get_intersection_waiting(agent_list[i])
    return reward


def find_adj_edge(map_dir):
    import sumolib
    attr_connection = {'connection': ['from', 'to', 'fromLane', 'toLane', 'via', 'tl', 'linkIndex', 'dir', 'state']}
    attr_edge = {'edge': ['id', 'from', 'to', 'priority']}

    # obtain all the edges in the network
    turn_info = dict()
    edges = list(sumolib.xml.parse(map_dir, 'net', attr_edge))[0]
    for edge in edges.edge:
        next_edges = dict()
        turn_info[edge.id] = next_edges

    # obtain the adjacent edges of each edge
    connections = list(sumolib.xml.parse(map_dir, 'net', attr_connection))[0]
    for connection in connections.connection:
        edge = connection.attr_from
        direction = connection.dir
        if direction == "L" or direction == "R":
            direction = "s"
        next_edge = connection.to
        turn_info[edge][direction] = next_edge

    return turn_info


def get_edge_state(agent_list, lanes):
    lane_state = dict()
    for lane in lanes:
        _state = dict()
        _state["velocity"] = tc.lane.getLastStepMeanSpeed(lane)
        _state["number"] = tc.lane.getLastStepVehicleNumber(lane)
        _state["length"] = tc.lane.getLength(lane)
        _state["density"] = _state["number"] / _state["length"]
        _state["time"] = tc.lane.getTraveltime(lane)
        s, e = tc.lane.getShape(lane)
        _state["sx"] = s[0]
        _state["sy"] = s[1]
        _state["ex"] = e[0]
        _state["ey"] = e[1]
        _state["is_dest"] = 0
        # _state.append(tc.lane.getLastStepMeanSpeed(lane))  # Velocity
        # _state.append(tc.lane.getLastStepVehicleNumber(lane))  # Number
        # _state.append(tc.lane.getLength(lane))  # Length
        # _state.append(float(tc.lane.getLastStepVehicleNumber(lane)) / float(tc.lane.getLength(lane)))  # Density
        # _state.append(tc.lane.getTraveltime(lane))  # Time
        # s, e = tc.lane.getShape(lane)
        # _state.append(s[0])
        # _state.append(s[1])
        # _state.append(e[0])
        # _state.append(e[1])
        # _state.append(0)  # indentify whether is the dest lane
        lane_state[lane] = _state
    return lane_state


def get_light_state(agent_list):
    light_state = []
    for agent in agent_list:
        light = tc.trafficlight.getPhase(agent.tl_id) // 2
        light_state.append(light)
    return light_state


def get_map_lanes(agent_list, all_lanes):
    edges = []
    count_light = []
    light_dict = {}
    lane_dict = {}
    for agent in agent_list:
        lanes = traci.trafficlight.getControlledLanes(agent.tl_id)
        light_dict[agent.tl_id] = []
        for lane in lanes:
            light_dict[agent.tl_id].append(lane)
            lane_dict[lane] = agent.tl_id
        light_dict[agent.tl_id] = list(set(light_dict[agent.tl_id]))
        for lane in light_dict[agent.tl_id]:
            all_lanes.remove(lane)
    for agent in agent_list:
        count = 12
        for lane in all_lanes:
            if agent.tl_id[-3:] in lane[:-4]:
                lane_dict[lane] = agent.tl_id
        edges.extend(light_dict[agent.tl_id])
        count_light.append(count)
    return edges, count_light, lane_dict


if __name__ == '__main__':
    start(execution=False, path='./res/hangzhou/train.sumocfg', seed='25', gui=None)
    for i in range(600):
        step()

    tc.close()
