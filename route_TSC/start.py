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


def actuated(path=None):
    """
    使用感应控制（相位可按检测器数据延长）运行仿真并记录统计结果。
    参数:
    - path: SUMO 配置文件 .sumocfg 路径
    功能:
    - 初始化路口控制器，依据感应器车头时距与排队密度在绿灯窗口内延长相位
    - 仿真 12600 步，累计各路口排队长度，计算均值并输出到 CSV
    - 打印各相位后半段的平均绿灯时长
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase_dim", type=int, default=4)
    ap.add_argument("--action_dim", type=int, default=4)
    args = ap.parse_args()
    agent_list = env.get_tl_list()
    tsc_list = []
    env.start(True, path)
    for agent in agent_list:
        tsc_list.append(traffic_light.MaxPressureTSC(agent, env.create_lane_to_det(),
                                                     env.get_lane_map(agent),
                                                     env.get_downstream(agent, "pressure"),
                                                     env.get_upstream(agent), [], args))
    env.init_context(tsc_list)
    greens = [0] * 9
    for i in range(12600):
        env.step()
        for j, agent in enumerate(tsc_list):
            simulation_current_time = traci.simulation.getTime()
            simulation_start_time = time.time() - simulation_current_time

            phase_number = traci.trafficlight.getPhase(agent.tl_id)
            next_phase_time = traci.trafficlight.getNextSwitch(agent.tl_id)

            phase_remain_time = next_phase_time - simulation_current_time
            phase_id = int(traci.trafficlight.getPhaseName(agent.tl_id))

            if phase_id != agent.last_phase_id:
                phase_headway = -1
                vehicle_list = []

            for detector_id in [agent.p[phase_id][0][0], agent.p[phase_id][1][0]]:
                detector_id = 'e1det_' + detector_id
                vehicleInfos = traci.inductionloop.getVehicleData(detector_id)
                if len(vehicleInfos) > 0:
                    vehicle_list.append(vehicleInfos)
            if len(vehicle_list) > 0:
                phase_headway = 0
            else:
                if phase_headway >= 0:
                    phase_headway += 1
            lane_max_vehicle = max(env.l_get_number_of_waiting_vehicles('e2det_' + agent.p[phase_id][0][0], div=False),
                                   env.l_get_number_of_waiting_vehicles('e2det_' + agent.p[phase_id][1][0], div=False))

            if 3 <= phase_remain_time < 16:
                if lane_max_vehicle >= 5:
                    if phase_remain_time < lane_max_vehicle * 2.5:
                        traci.trafficlight.setPhaseDuration(agent.tl_id, phase_remain_time + 1)
                        # print('当前相位剩余绿灯时间不满足车辆放行,相位延长1s')
                    elif 0 <= phase_headway <= 2:
                        # print('车头时距为%s 密度%s 因此延长' % (phase_headway, lane_max_vehicle))
                        traci.trafficlight.setPhaseDuration(agent.tl_id, phase_remain_time + 1)
                        # print('相位延长1s')

                    # print(phase_remain_time)
                else:
                    pass
                    # print('车头时距为%s 密度%s 因此不延长' % (phase_headway, lane_max_vehicle))
            elif phase_remain_time <= 12 and phase_number % 3 == 0:
                pass
                # print('倒计时触发，不响应过车 %s' % phase_remain_time)
            else:
                pass
                # print('未在延长时间窗%s' % phase_remain_time)
            greens[j] += 1
            if phase_remain_time == 1 and next_phase_time >= 5:
                if greens[j] >= 5:
                    agent.greens[phase_id].append(greens[j])
                greens[j] = 0
            agent.last_phase_id = phase_id

        for agent in tsc_list:
            queue_length = 0.0
            for lane in env.get_control_lanes(agent):
                queue_length += env.get_number_of_queued_vehicles(lane) * 7.5
            agent.write_length(queue_length)
    avg_queue_length = 0.0
    queues = np.zeros((10, 3))
    for i in range(3):
        t = 0
        l = 1800 + 3600 * i
        r = l + 3600
        for j, agent in enumerate(tsc_list):
            queues[j][i] = sum(agent.get_length()[l:r]) / len(agent.get_length()[l:r]) / 8
            t += sum(agent.get_length()[l:r]) / len(agent.get_length()[l:r]) / 8
        queues[9][i] = t / 9
        avg_queue_length += t / 9
    print("average queue length: ", avg_queue_length / 3)
    waiting, duration = tripinfo.get_segment_info()
    queues_list = [sum(x) / len(x) for x in
                   zip(tsc_list[0].get_length(), tsc_list[1].get_length(), tsc_list[2].get_length(),
                       tsc_list[3].get_length(), tsc_list[4].get_length(), tsc_list[5].get_length(),
                       tsc_list[6].get_length(), tsc_list[7].get_length(), tsc_list[8].get_length())]
    # queue length visualization
    # o = Visualization()
    # ax = o.create_png(xlabel='Times', ylabel='Veh')
    # queues = [sum(x)/len(x) for x in zip(tsc_list[0].get_length(), tsc_list[1].get_length(), tsc_list[2].get_length(),
    #                                      tsc_list[3].get_length(), tsc_list[4].get_length(), tsc_list[5].get_length(),
    #                                      tsc_list[6].get_length(), tsc_list[7].get_length(), tsc_list[8].get_length())]
    os.makedirs('./simudata/Actuated/testFlow_t2', exist_ok=True)
    o = Visualization()
    o.csv_avs(queues, file="Actuated/testFlow_t2", name="queues" + path[22:-8])
    o.csv_av(waiting, file="Actuated/testFlow_t2", name="waiting" + path[22:-8])
    o.csv_av(duration, file="Actuated/testFlow_t2", name="duration" + path[22:-8])
    o.csv_av(queues_list[1800:], file="Actuated/testFlow_t2", name="queues_list" + path[22:-8])
    for j, agent in enumerate(tsc_list):
        for i in range(4):
            print(j, i, np.mean(agent.greens[i][int(len(agent.greens[i])) // 2:]))
        # o.csv_av(agent.green, file='Actuated', name="green"+str(j))
    env.close()


def static(execution=True, path=None, seed=25):
    gen_cav = []
    csv_cav = 2983 if '2983' in path else 5694
    with open(f'./res/hangzhou/cav_{csv_cav}.csv', newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # 使用列表推导式去除空值并将数据转换为整数（如果适用）
            processed_row = [int(cell) for cell in row if cell.strip()]
            gen_cav.extend(processed_row)

    print(gen_cav)
    leaving_cav_set = set()
    loaded_cav_set = set()
    total_arrived = 0

    env.start(execution, path)
    sumo_duration = 3600

    for i in range(sumo_duration):
        loaded_cav_set = loaded_cav_set | set(traci.simulation.getDepartedIDList())
        env.step()
        leaving_cav_set = leaving_cav_set | set(traci.simulation.getArrivedIDList())
        for cav in gen_cav:
            if str(cav) in loaded_cav_set and str(cav) not in leaving_cav_set \
                    and traci.vehicle.getRoadID(str(cav)) == traci.vehicle.getRoadID(str(cav))[-1]:
                leaving_cav_set.add(str(cav))
                total_arrived += 1

    env.close()
    waiting, duration = tripinfo.get_segment_info()
    os.makedirs('./simudata/Fixedtime/', exist_ok=True)
    o = Visualization()
    o.csv_av(waiting, file="Fixedtime/", name="" + str(seed) + "_waiting_" + path[-12:-8])
    o.csv_av(duration, file="Fixedtime/", name="" + str(seed) + "_duration_" + path[-12:-8])
    throughput = [tripinfo.get_tripinfo("duration", True)]
    routelength = [tripinfo.get_tripinfo("routeLength", False)]
    arrivalrate = [total_arrived / len(gen_cav)]
    o.csv_av(throughput, file="Fixedtime/", name="" + str(seed) + "_throughput_" + path[-12:-8])
    o.csv_av(routelength, file="Fixedtime/", name="" + str(seed) + "_routelength_" + path[-12:-8])
    o.csv_av(arrivalrate, file="Fixedtime/", name="" + str(seed) + "_arrivalrate_" + path[-12:-8])


def max_pressure(path=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase_dim", type=int, default=4)
    ap.add_argument("--action_dim", type=int, default=4)
    args = ap.parse_args()
    agent_list = env.get_tl_list()
    tsc_list = []
    env.start(True, path)
    for agent in agent_list:
        tsc_list.append(traffic_light.MaxPressureTSC(agent, env.create_lane_to_det(),
                                                     env.get_lane_map(agent),
                                                     env.get_downstream(agent, "pressure"),
                                                     env.get_upstream(agent), [], args))
    env.init_context(tsc_list)
    for i in range(12600):
        for agent in tsc_list:
            # print(agent.tl_id, agent.phase, agent.duration)
            if agent.duration != 0:
                agent.duration = agent.duration - 1
                continue
            # yellow ends, switch pre-cal green
            if agent.signal == 'Y':
                agent.set_phase()
            # green ends, cal next phase
            else:
                pressure = []
                for p in range(4):
                    pressure.append(
                        (p, env.get_pressure(agent.dic[agent.p[p][0][0]], agent.dic[agent.p[p][0][1]], agent) \
                         + env.get_pressure(agent.dic[agent.p[p][1][0]], agent.dic[agent.p[p][1][1]], agent)))
                pressure = sorted(pressure, key=lambda p: p[1], reverse=True)
                next_phase = pressure[0][0]
                # pressure = [p for p in pressure if p[1] == pressure[0][1]]
                # if len(pressure) == 1:
                #     next_phase = pressure[0][0]
                # else:
                #     next_phase = random.choice(pressure)[0]
                agent.set_phase(next_phase)
        for agent in tsc_list:
            queue_length = 0.0
            for lane in env.get_control_lanes(agent):
                queue_length += env.get_number_of_queued_vehicles(lane) * 7.5
            agent.write_length(queue_length)
        env.step()
    avg_queue_length = 0.0
    queues = np.zeros((10, 3))
    for i in range(3):
        t = 0
        l = 1800 + 3600 * i
        r = l + 3600
        for j, agent in enumerate(tsc_list):
            queues[j][i] = sum(agent.get_length()[l:r]) / len(agent.get_length()[l:r]) / 8
            t += sum(agent.get_length()[l:r]) / len(agent.get_length()[l:r]) / 8
        queues[9][i] = t / 9
        avg_queue_length += t / 9
    print("average queue length: ", avg_queue_length / 3)
    waiting, duration = tripinfo.get_segment_info()
    queues_list = [sum(x) / len(x) for x in
                   zip(tsc_list[0].get_length(), tsc_list[1].get_length(), tsc_list[2].get_length(),
                       tsc_list[3].get_length(), tsc_list[4].get_length(), tsc_list[5].get_length(),
                       tsc_list[6].get_length(), tsc_list[7].get_length(), tsc_list[8].get_length())]

    os.makedirs('./simudata/MaxPressure/testFlow_counting', exist_ok=True)
    # queue length visualization
    # o = Visualization()
    # ax = o.create_png(xlabel='Times', ylabel='Veh')
    # queues = [sum(x)/len(x) for x in zip(tsc_list[0].get_length(), tsc_list[1].get_length(), tsc_list[2].get_length(),
    #                                      tsc_list[3].get_length(), tsc_list[4].get_length(), tsc_list[5].get_length(),
    #                                      tsc_list[6].get_length(), tsc_list[7].get_length(), tsc_list[8].get_length())]
    o = Visualization()
    o.csv_avs(queues, file="MaxPressure/testFlow_counting", name="queues" + path[22:-8])
    o.csv_av(waiting, file="MaxPressure/testFlow_counting", name="waiting" + path[22:-8])
    o.csv_av(duration, file="MaxPressure/testFlow_counting", name="duration" + path[22:-8])
    o.csv_av(queues_list[1800:], file="MaxPressure/testFlow_counting", name="queues_list" + path[22:-8])
    env.close()


def process():
    """
    处理流量输入：读取并绘制 OD 流量图，生成转向关系文件。
    功能:
    - 使用 CsvInterpreter 读取 "xuancheng.csv" 并生成 OD 对
    - 计算 OD 配对统计
    - 使用 XmlGenerator 依据 OD 对生成 turndefs XML
    """
    # plot od-map
    flow = CsvInterpreter("xuancheng.csv")
    flow.flow_xuancheng()
    flow.generator_od_pair(0, 288)
    flow.calculate_od_pair()

    # generate traffic flow
    turn = XmlGenerator('3_3_1.turndefs.xml')
    turn.generator_turn_def(flow.od_pair)


def main(exe=False, name=None, idx=0, comment="", seed='25', light_path=None, ratio=0.5):
    """
    训练/执行交通信号控制（可选车辆路由 RL）主入口。
    参数:
    - exe: 执行模式（True 为执行/评估，False 为训练/预训练）
    - name: 运行名（为空时自动生成）
    - idx: 起始 episode 索引（>0 时加载部分模型/结果）
    - comment: 名称附加备注
    - seed: 仿真随机种子
    - light_path: 可选信号控制器模型加载路径
    - ratio: 车辆启用路由 RL 的占比（CAV 比例）
    功能:
    - 解析通用与算法超参数，初始化多路口智能体与 SUMO 环境
    - 按需创建 RouterNet（DQN/PPO）并在每个 episode 中与信号控制协同运行
    - 收集并保存时长、等待、速度、延误等指标至 simudata/name 目录
    - 条件保存最优/周期模型权重至 model/name
    - 返回运行名
    """
    torch.set_num_threads(1)

    ap = get_common_args()

    args = ap.parse_args()
    if config.ROUTER_RL["ALGO"] == "PPO":
        ap = get_ppo_arguments()
    if args.agent_type in {"PPO", "CenPPO", "PS-PPO"}:
        ap = get_ppo_arguments()
    elif args.agent_type == "MAT":
        ap = get_mat_arguments()
    # elif args.agent_type == "SAC":
    #     ap = get_sac_arguments()
    elif args.agent_type == "MAPPO":
        ap = get_mappo_arguments()
    elif args.agent_type == "MA2C":
        ap = get_ma2c_arguments()
    args = ap.parse_args()
    config.SPATIAL["TYPE"] = args.spatial

    args.rate = ratio

    # if args.state_contain_action:

    # simulation data
    awt = []
    awc = []
    aws = []
    awl = []
    aww = []
    awd = []
    rewards = np.array([])
    aw_length = np.array([])

    # png, csv
    o = Visualization()

    execute = exe
    # initialization the multi-agent
    if name is not None:
        agents = lights.init(args, execute, path=f"./res/{args.map}/train.sumocfg")  # init Lights object
    else:
        agents = lights.init(args, execute, path=f"./res/{args.map}/train.sumocfg")  # init Lights object

    # if idx > 0:
    #     o.load_reward(agents, name)

    if name is None:
        tm = str(time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime()))
        name = f'{args.agent_type}-{args.spatial}-{args.temporal}-{tm}-{args.map}' \
               f'-UseLighter_{args.org if args.lighter else False}-UseRouter_{args.rate if args.router else 0.0}' \
               f'-Algo_{args.algo}-{comment}'
    # load saved model, load saved replay buffer(list)
    else:
        # agents.load_model(name)
        pass
        # if args.agent_type != 'ppo':
        #     agents.load_buffer(name)
    name = f'{ratio}_{args.algo}_0315'
    print('run:', name)
    ap.add_argument("--dir", type=str, default=name)
    if execute:
        pass
        # agents.load_model(name)

    RouterNet = None
    adj_edge = None
    if args.router:
        from dqnagent import DQNAgent, CDQNAgent
        from ppoagent import PPOAgent
        adj_edge = env.find_adj_edge(f'./res/{args.map}/SUMO_roadnet_4_4_4_phase_Right_Green_turn.net.xml')
        if args.direction == 3:
            adj_edge = env.find_adj_edge(f'./res/{args.map}/4_Phase.net.xml')
        state_dim = args.agent_num * args.road_feature * 4 * 3 + args.cav_feature
        if config.ROUTER_RL["ALGO"] == "DQN":
            RouterNet = DQNAgent(state_dim=state_dim, action_dim=args.direction, args=args, tl_id="veh", net_type="",
                                 net_config=config.ROUTER_RL)
        elif config.ROUTER_RL["ALGO"] == "PPO":
            RouterNet = PPOAgent(state_dim=state_dim, action_dim=args.direction, args=args, tl_id="veh", net_type="",
                                 net_config=config.ROUTER_RL)

    max_t = 999
    loss = []

    if idx > 0:
        RouterNet.load_model(name+'/ep50')
        print("########### LOADED ###########")

    for i in range(idx, args.episode):
        print("ep:", i)
        cnt = 0
        agent_list = agents.get_agent_list()
        if args.agent_type == "MAT":
            agents.clear_buf()
        for agent in agent_list:
            if agent.rl_model.execution:
                cnt += 1
            agent.his_speed = []

        if cnt == len(agent_list):
            execute = True
        config.SIMULATION['EXECUTE'] = execute

        # pre-trian
        if i > idx and i >= 0 and (not execute):
            env.start(execute, path=f"./res/{args.map}/train.sumocfg", seed=seed)
        elif i > idx:
            # agents.load_model(name)
            env.start(execute, seed=seed)
        if args.l2v:
            assert light_path is not None
            # agents.load_model(light_path)
        env.init_context(agent_list)

        os.makedirs('./simudata/' + name, exist_ok=True)
        os.makedirs('./model/' + name, exist_ok=True)

        gen_cav = []
        n, k = 4200, int(4200 * args.rate)
        # gen_cav = random.sample(range(n), k)
        # print(gen_cav)
        args.veh_num = n

        # gen_cav = []
        # csv_cav = 2983
        # with open(f'./res/hangzhou/cav_{csv_cav}_{args.rate}.csv', newline='', encoding='utf-8-sig') as csvfile:
        #     reader = csv.reader(csvfile)
        #     for row in reader:
        #         # 使用列表推导式去除空值并将数据转换为整数（如果适用）
        #         processed_row = [int(cell) for cell in row if cell.strip()]
        #         gen_cav.extend(processed_row)

        # print(gen_cav)

        traci.vehicletype.setTau('DEFAULT_VEHTYPE', 2.0)

        reward, t, c, s, l, w, d, rt, _loss = episode.run(i, agents, o, args, execute, name, RouterNet, adj_edge, gen_cav)
        awt.append(t)  # duration
        awc.append(c)  # waiting count
        aws.append(s)  # speed
        awl.append(l)  # time loss
        aww.append(w)  # waiting time
        awd.append(d)  # depart delay
        loss.extend(_loss)
        # if i == 0:
        #     rewards = reward
        # else:
        #     rewards = np.concatenate((rewards, reward))

        # calculate average queue length
        avg_queue_length = 0.0
        # for agent in agents.agent_list:
        #     avg_queue_length += sum(agent.get_length()[-3600:]) / len(agent.get_length()[-3600:]) / 9 / 8
        aw_length = np.append(aw_length, avg_queue_length)
        o.csv_queue(aw_length, name)
        print("average queue length", avg_queue_length)

        print("ep:", i, "average travel time:", t)

        if t < max_t:
            os.makedirs('./model/' + name + '/best_model', exist_ok=True)
            # agents.save_model(name + '/best_model')
            if args.router:
                RouterNet.save_model(name + '/best_model')
            max_t = t

        # agents.save_model(name)
        if args.router:
            RouterNet.save_model(name)
        # with open('o.txt', 'w') as file:
        #     file.write(f', name={name}, idx={i}')
        # agents.save_buffer(name)
        if not execute:
            # o.png_step_reward(agents)
            o.csv_reward(agents, name)
            # o.csv_loss(agents, name)
        # checkpoint
        if i >= 5 and i % 5 == 0:
            os.makedirs('./model/' + name + '/' + "ep" + str(i), exist_ok=True)
            # agents.save_model(name + '/' + "ep" + str(i))
            if args.router:
                RouterNet.save_model(name + '/' + "ep" + str(i))
            # agents.save_buffer(name+'/'+"ep"+str(i))
        # if t <= 145.0:
        #     os.makedirs('./model/'+name+'/'+"ck"+str(k), exist_ok=True)
        #     agents.save_model(name+'/'+"ck"+str(k))
        #     agents.save_buffer(name+'/'+"ck"+str(k))
        #     k = k + 1

    o.csv_av(loss, name, "loss")

    o.csv_av(awt, name, "train-duration" if not execute else "execution-duration")
    o.csv_av(awc, name, "train-count" if not execute else "execution-count")
    o.csv_av(aws, name, "train-speed" if not execute else "execution-speed")
    o.csv_av(awl, name, "train-loss" if not execute else "execution-loss")
    o.csv_av(aww, name, "train-waiting" if not execute else "execution-waiting")
    o.csv_av(awd, name, "train-delay" if not execute else "execution-delay")
    # o.png_loss(agents)
    # o.calculate_reward(name)
    # o.calculate_loss(name)
    return name


def myplot(name=None):
    """
    可视化辅助：绘制总回报曲线。
    参数:
    - name: 运行名或集合，默认 {"test"}
    功能:
    - 调用 Visualization 生成总回报图（可扩展：局部回报/损失）
    """
    if name is None:
        name = {"test"}
    o = Visualization()
    o.png_total_reward(name)
    # o.png_local_reward(name)
    # o.png_total_loss(name)

def exe_flow(exe=True, name=None, path=None, idx=0.0, seed='25', ratio=0.5, cav_seed=25, algo='astar_dqn'):
    """
    执行阶段评估：加载已训练路由 RL 模型并在指定场景运行，导出评估指标。
    参数:
    - exe: 执行模式标志（通常 True）
    - name: 已训练模型目录名或检查点（将从中加载 RouterNet）
    - path: SUMO 执行配置 .sumocfg 路径
    - idx: 预留参数（未直接使用）
    - seed: 仿真随机种子
    - ratio: CAV 比例（用于选择 CSV）
    - cav_seed: 生成/选择 CAV 集的种子（影响 CSV 文件名）
    功能:
    - 初始化多路口智能体与环境，加载路由网络 RouterNet
    - 从 CSV 读取 CAV 集，运行若干 episode，记录并保存各类指标
    - 输出总体/仅 CAV 的行程时长、等待时间、时耗、路径长度等列表到 simudata/name/testFlow
    - 额外导出吞吐、路程、到达率、排队长度以及 CO2 排放等统计
    """
    torch.set_num_threads(1)

    ap = get_common_args(EXEC=True)

    args = ap.parse_args()
    sys.argv= ["arguments.py","--algo",algo]
    args=ap.parse_args()

    if args.agent_type in {"PPO", "CenPPO"}:
        ap = get_ppo_arguments(EXEC=True)
    elif args.agent_type == "MAT":
        ap = get_mat_arguments(EXEC=True)
    # elif args.agent_type == "SAC":
    #     ap = get_sac_arguments(EXEC=True)
    elif args.agent_type == "MAPPO":
        ap = get_mappo_arguments(EXEC=True)
    elif args.agent_type == "MA2C":
        ap = get_ma2c_arguments(EXEC=True)
    args = ap.parse_args()
    config.SPATIAL["TYPE"] = args.spatial
    # simulation data
    awt = []
    awc = []
    aws = []
    awl = []
    aww = []
    awd = []
    aw_throughput = []
    aw_routelength = []
    aw_arrivalrate = []
    aw_length = []
    ap.add_argument("--dir", type=str, default=name)
    execute = exe

    # initialization the multi-agent
    agents = lights.init(args, execute, path, seed=seed)  # Lights object

    args.rate = ratio

    # clear attentions list
    # for agent in agents.agent_list:
    #    agent.rl_model.q_network.gat.attns = {}

    if execute:
        pass
        # agents.load_model(name)
    # png, csv
    o = Visualization()

    RouterNet = None
    adj_edge = None
    if args.router:
        from dqnagent import DQNAgent, CDQNAgent
        from ppoagent import PPOAgent
        adj_edge = env.find_adj_edge(f'./res/{args.map}/SUMO_roadnet_4_4_4_phase_Right_Green_turn.net.xml')
        if args.direction == 3:
            adj_edge = env.find_adj_edge(f'./res/{args.map}/4_Phase.net.xml')
        state_dim = args.agent_num * args.road_feature * 4 * 3 + args.cav_feature
        if config.ROUTER_RL["ALGO"] == "DQN":
            RouterNet = DQNAgent(state_dim=state_dim, action_dim=args.direction, args=args, tl_id="veh", net_type="",
                                 net_config=config.ROUTER_RL)
        elif config.ROUTER_RL["ALGO"] == "PPO":
            RouterNet = PPOAgent(state_dim=state_dim, action_dim=args.direction, args=args, tl_id="veh", net_type="",
                                 net_config=config.ROUTER_RL)

    # 此处不应该直接load_model,因为第一次跑不应该有model
    RouterNet.load_model(name)

    gen_cav = []
    csv_cav = 2983 if '2983' in path else 6984
    with open(f'./res/hangzhou/{cav_seed}_cav_{csv_cav}_{args.rate}.csv', newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # 使用列表推导式去除空值并将数据转换为整数（如果适用）
            processed_row = [int(cell) for cell in row if cell.strip()]
            gen_cav.extend(processed_row)
    args.veh_num = csv_cav
    print(gen_cav)

    for i in range(args.episode):
        config.SIMULATION['EXECUTE'] = execute
        # agents.load_model(name)
        if i > 0:
            env.start(execute, path, seed=seed)
        env.init_context(agents.get_agent_list())

        traci.vehicletype.setTau('DEFAULT_VEHTYPE', 2.0)

        reward, t, c, s, l, w, d, rt, _ = episode.run(i, agents, o, args, execute, name, RouterNet, adj_edge, gen_cav)
        awt.append(t)  # duration
        awc.append(c)  # waiting count
        aws.append(s)  # speed
        awl.append(l)  # time loss
        aww.append(w)  # waiting time
        awd.append(d)  # depart delay

        aw_throughput.append(tripinfo.get_tripinfo("duration", True))
        aw_routelength.append(tripinfo.get_tripinfo("routeLength", False))
        aw_arrivalrate.append(rt)

        # calculate average queue length
        avg_queue_length = 0.0
        # for agent in agents.agent_list:
        #     avg_queue_length += sum(agent.get_length()[:12600]) / len(agent.get_length()[:12600]) / 9 / 8
        aw_length.append(avg_queue_length)
        print("average queue length", avg_queue_length)

        print("ep:", i, "average waiting time:", t)
        # env.close()
        os.makedirs('./simudata/' + name + "/testFlow", exist_ok=True)
        # agents.save_model(name)

    # tsc_list = agents.agent_list
    # # ax = o.create_png(xlabel='Times', ylabel='Veh')
    # queues = np.zeros((10, 3))
    # avg_queue_length = 0.0
    # for i in range(3):
    #     t = 0
    #     l = 1800 + 3600 * i
    #     r = l + 3600
    #     for j, agent in enumerate(tsc_list):
    #         queues[j][i] = sum(agent.get_length()[l:r]) / len(agent.get_length()[l:r])
    #         t += sum(agent.get_length()[l:r]) / len(agent.get_length()[l:r])
    #     queues[9][i] = t / 9
    #     avg_queue_length += t / 9
    # print("average queue length: ", avg_queue_length / 3)
    waiting, duration = tripinfo.get_segment_info()
    print("vehicle waiting: ",waiting, "average duration: ",duration)
    waiting=np.array([waiting])
    duration=np.array([duration])
    # queues_list = [sum(x) / len(x) for x in
    #                zip(tsc_list[0].get_length(), tsc_list[1].get_length(), tsc_list[2].get_length(),
    #                    tsc_list[3].get_length(), tsc_list[4].get_length(), tsc_list[5].get_length(),
    #                    tsc_list[6].get_length(), tsc_list[7].get_length(), tsc_list[8].get_length())]
    # o.csv_av(queues_list[1800:], file=name+"/testFlow1/", name="queues_list_3" + path[22:-8], column=str(seed))

    # queue length visualization
    # o = Visualization()
    # ax = o.create_png(xlabel='Times', ylabel='Veh')
    # queues = [sum(x)/len(x) for x in zip(tsc_list[0].get_length(), tsc_list[1].get_length(), tsc_list[2].get_length(),
    #                                      tsc_list[3].get_length(), tsc_list[4].get_length(), tsc_list[5].get_length(),
    #                                      tsc_list[6].get_length(), tsc_list[7].get_length(), tsc_list[8].get_length())]
    # o.csv_avs(queues, file=name+"/testFlow1/", name="queues"+path[22:-8])
    o.csv_av(waiting, file=name + "/testFlow/", name=str(ratio) + "_" + str(cav_seed) + "_loss_" + path[-12:-8] + str(seed))
    o.csv_av(duration, file=name + "/testFlow/", name=str(ratio) + "_" + str(cav_seed) + "_duration_" + path[-12:-8] + str(seed))

    route_length_list = tripinfo.get_tripinfo_list('routeLength', cav_set=set(gen_cav))
    duration_list = tripinfo.get_tripinfo_list(cav_set=set(gen_cav))
    delay_list = tripinfo.get_tripinfo_list('timeLoss', cav_set=set(gen_cav))
    waiting_list = tripinfo.get_tripinfo_list('waitingTime', cav_set=set(gen_cav))
    o.csv_av(route_length_list, file=name + "/testFlow/", name=str(ratio) + "_" + str(cav_seed) + "_cav_route_length_list_" + path[-12:-8] + str(seed))
    o.csv_av(duration_list, file=name + "/testFlow/", name=str(ratio) + "_" + str(cav_seed) + "_cav_duration_list_" + path[-12:-8] + str(seed))
    o.csv_av(delay_list, file=name + "/testFlow/", name=str(ratio) + "_" + str(cav_seed) + "_cav_delay_list_" + path[-12:-8] + str(seed))
    o.csv_av(waiting_list, file=name + "/testFlow/", name=str(ratio) + "_" + str(cav_seed) + "_cav_waiting_list_" + path[-12:-8] + str(seed))

    route_length_list = tripinfo.get_tripinfo_list('routeLength')
    duration_list = tripinfo.get_tripinfo_list()
    delay_list = tripinfo.get_tripinfo_list('timeLoss')
    waiting_list = tripinfo.get_tripinfo_list('waitingTime')
    o.csv_av(route_length_list, file=name + "/testFlow/", name=str(ratio) + "_" + str(cav_seed) + "_route_length_list_" + path[-12:-8] + str(seed))
    o.csv_av(duration_list, file=name + "/testFlow/", name=str(ratio) + "_" + str(cav_seed) + "_duration_list_" + path[-12:-8] + str(seed))
    o.csv_av(delay_list, file=name + "/testFlow/", name=str(ratio) + "_" + str(cav_seed) + "_delay_list_" + path[-12:-8] + str(seed))
    o.csv_av(waiting_list, file=name + "/testFlow/", name=str(ratio) + "_" + str(cav_seed) + "_waiting_list_" + path[-12:-8] + str(seed))

    # o.csv_av(queues_list[1800:], file=name + "/testFlow/", name="" + str(cav_seed) + "_queues_list_3" + path[22:-8],
    #          column=str(cav_seed))
    # o.csv_av(queues, file=name + "/testFlow/", name="queues" + path[22:-8])
    # ax.plot([i for i in range(len(queues))], queues)
    # plt.show()

    # trip_waiting = tripinfo.get_trip_waiting()
    # o.csv_av(trip_waiting, file=name+"/testFlow1/", name="waiting"+path[22:-8])

    o.csv_av(aw_throughput, name, "throughput",
             "./simudata/" + name + "/testFlow/" + str(ratio) + "_" + str(cav_seed) + "_throughput_" + path[-12:-8] + str(seed) + ".csv")
    o.csv_av(aw_routelength, name, "routelength",
             "./simudata/" + name + "/testFlow/" + str(ratio) + "_" + str(cav_seed) + "_routelength_" + path[-12:-8] + str(seed) + ".csv")
    o.csv_av(aw_arrivalrate, name, "arrivalrate",
             "./simudata/" + name + "/testFlow/" + str(ratio) + "_" + str(cav_seed) + "_arrivalrate_" + path[-12:-8] + str(seed) + ".csv")

    # aw_length = [tripinfo.get_avg_qlength()]

    o.csv_av(aw_length, name, "length",
             "./simudata/" + name + "/testFlow/" + str(ratio) + "_" + str(cav_seed) + "_qlength_" + path[-12:-8] + str(seed) + ".csv")

    o.csv_av(awt, name, "train-duration" if not execute else "execution-duration",
             "./simudata/" + name + "/testFlow/" + str(ratio) + "_" + str(cav_seed) + "_duration_" + path[-12:-8] + str(seed) + ".csv")
    o.csv_av(awc, name, "train-count" if not execute else "execution-count",
             "./simudata/" + name + "/testFlow/" + str(ratio) + "_" + str(cav_seed) + "_count_" + path[-12:-8] + str(seed) + ".csv")
    o.csv_av(aws, name, "train-speed" if not execute else "execution-speed",
             "./simudata/" + name + "/testFlow/" + str(ratio) + "_" + str(cav_seed) + "_speed_" + path[-12:-8] + str(seed) + ".csv")
    o.csv_av(awl, name, "train-loss" if not execute else "execution-loss",
             "./simudata/" + name + "/testFlow/" + str(ratio) + "_" + str(cav_seed) + "_loss_" + path[-12:-8] + str(seed) + ".csv")
    o.csv_av(aww, name, "train-waiting" if not execute else "execution-waiting",
             "./simudata/" + name + "/testFlow/" + str(ratio) + "_" + str(cav_seed) + "_waiting_" + path[-12:-8] + str(seed) + ".csv")
    o.csv_av(awd, name, "train-delay" if not execute else "execution-delay",
             "./simudata/" + name + "/testFlow/" + str(ratio) + "_" + str(cav_seed) + "_delay_" + path[-12:-8] + str(seed) + ".csv")
    # tripinfo.get_mathot("./simudata/" + name + "/testFlow/" + str(cav_seed) + "_mathot_tripduration_" + path[-12:-8] + str(seed),
    #                     "duration")
    # tripinfo.get_mathot("./simudata/" + name + "/testFlow/" + str(cav_seed) + "_mathot_CO2_" + path[-12:-8] + str(seed),
    #                     "CO2_abs")

    emission_CO2 = [tripinfo.get_emission_info()]
    o.csv_av(emission_CO2, name, "emission",
             "./simudata/" + name + "/testFlow/" + str(ratio) + "_" + str(cav_seed) + "_CO2_" + path[-12:-8] + str(seed) + ".csv")

    # tripinfo.get_inter_queue(f'./simudata/{name}/queue_mathot_{path[-12:-8] + str(seed)}')

    cav_duration = [tripinfo.get_cav_info(set(gen_cav), 'duration')]
    cav_waiting = [tripinfo.get_cav_info(set(gen_cav), 'waitingTime')]
    cav_loss = [tripinfo.get_cav_info(set(gen_cav), 'timeLoss')]
    cav_route_length = [tripinfo.get_cav_info(set(gen_cav), 'routeLength')]

    o.csv_av(cav_duration, file=f"{name}/testFlow/", name="" + str(ratio) + "_" + str(cav_seed) + "_cav_duration_" + path[-12:-8] + str(seed))
    o.csv_av(cav_waiting, file=f"{name}/testFlow/", name="" + str(ratio) + "_" + str(cav_seed) + "_cav_waiting_" + path[-12:-8] + str(seed))
    o.csv_av(cav_loss, file=f"{name}/testFlow/", name="" + str(ratio) + "_" + str(cav_seed) + "_cav_loss_" + path[-12:-8] + str(seed))
    o.csv_av(cav_route_length, file=f"{name}/testFlow/", name="" + str(ratio) + "_" + str(cav_seed) + "_cav_routelength_" + path[-12:-8] + str(seed))

    # save file: agent.rl_model.gat.atts(list)
    # if args.spatial == 'GAT':
    #     df = pd.DataFrame(tsc_list[4].rl_model.q_network.gat.atts,
    #                       columns=['Inter 5 to Inter 5', 'Inter 2 to Inter 5', 'Inter 4 to Inter 5',
    #                                'Inter 6 to Inter 5', 'Inter 8 to Inter 5'])
    #     df.to_csv('./simudata/' + name + '/' + args.spatial + '-' + args.temporal + '-atts.csv', index=False, sep=',')

    # o.png_loss(agents)
    # o.calculate_reward(name)
    # o.calculate_loss(name)
    # return name


def greedy_strategy(path=None, seed=8, rate=0.5, algo="Astar", m_seed='0'):
    """
    基线策略评估：以贪心/启发式路由策略对 CAV 进行重路由并统计指标。
    参数:
    - path: SUMO 配置 .sumocfg 路径
    - seed: 选择 CAV 集的种子（对应 CSV 文件）
    - rate: CAV 比例
    - algo: 输出目录名中使用的标识（如 Astar）
    - m_seed: SUMO 仿真内部种子（或混合种子）
    功能:
    - 读取 CAV 集，仿真运行 3600 步，基于 CAVAgent 进行策略评估
    - 统计 CAV 与总体的路径长度、时长、时耗、等待等并输出到 simudata/algo/rate/
    - 同时输出吞吐、平均速度与 CO2 排放等聚合指标
    """

    csv_cav = 2983 if '2983' in path else 6984

    # gen_cav = random.sample(range(csv_cav), int(csv_cav * rate))
    # gen_cav = list(gen_cav)
    # with open(f'./res/hangzhou/{seed}_cav_{csv_cav}_{rate}.csv', 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(gen_cav)

    gen_cav = []
    with open(f'./res/hangzhou/{seed}_cav_{csv_cav}_{rate}.csv', newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # 使用列表推导式去除空值并将数据转换为整数（如果适用）
            processed_row = [int(cell) for cell in row if cell.strip()]
            gen_cav.extend(processed_row)

    print(len(gen_cav), gen_cav)

    cav_list = []
    leaving_cav_set = set()

    env.start(True, path, m_seed)
    total_arrived = 0
    adj_edge = env.find_adj_edge(f'./res/hangzhou/4_Phase.net.xml')

    traci.vehicletype.setTau('DEFAULT_VEHTYPE', 2.0)

    from agent import CAVAgent
    for stp in range(3600):
        # for cav in cav_list:
        #     if cav.veh_id not in leaving_cav_set and not cav.arrived():
        #         road = traci.vehicle.getRoadID(cav.veh_id)
        #         if len(road) == 0 or road[0] == ':':
        #             continue
        #         vx, vy = traci.vehicle.getPosition(cav.veh_id)
        #         lx, ly = traci.lane.getShape(traci.vehicle.getLaneID(cav.veh_id))[1]
        #         dis = ((vx - lx) ** 2 + (vy - ly) ** 2) ** 0.5
        #         if 100 <= dis <= 150 and ('cur' not in cav.act_his or cav.act_his['cur'] != road):
        #             traci.vehicle.rerouteTraveltime(cav.veh_id)
        #             cav.act_his = {}
        #             if road not in cav.act_his:
        #                 cav.act_his[road] = 1
        #             else:
        #                 cav.act_his[road] += 1
        #             cav.act_his['cur'] = road

        loaded_list = list(traci.simulation.getDepartedIDList())
        if len(loaded_list) > 0:
            cav_list.extend([CAVAgent(cav, router=None, adj_edge=adj_edge, args=None) for cav in loaded_list if int(cav) in set(gen_cav)])
        env.step()

        leaving_cav_set = leaving_cav_set | set(traci.simulation.getArrivedIDList())

    for cav in gen_cav:
        if str(cav) in leaving_cav_set:
            total_arrived += 1

    print("Loaded:", len(cav_list))
    print("Arrived:", total_arrived)

    env.close()
    waiting, duration = tripinfo.get_segment_info()
    print("vehicle waiting: ", waiting, "average duration: ", duration)
    waiting = np.array([waiting])
    duration = np.array([duration])
    os.makedirs(f'./simudata/{algo}/{rate}/', exist_ok=True)
    o = Visualization()

    route_length_list = tripinfo.get_tripinfo_list('routeLength', cav_set=set(gen_cav))
    duration_list = tripinfo.get_tripinfo_list(cav_set=set(gen_cav))
    delay_list = tripinfo.get_tripinfo_list('timeLoss', cav_set=set(gen_cav))
    waiting_list = tripinfo.get_tripinfo_list('waitingTime', cav_set=set(gen_cav))
    o.csv_av(route_length_list, file=f"{algo}/{rate}/", name="" + str(seed) + "_cav_route_length_list_" + path[-12:-8])
    o.csv_av(duration_list, file=f"{algo}/{rate}/", name="" + str(seed) + "_cav_duration_list_" + path[-12:-8])
    o.csv_av(delay_list, file=f"{algo}/{rate}/", name="" + str(seed) + "_cav_delay_list_" + path[-12:-8])
    o.csv_av(waiting_list, file=f"{algo}/{rate}/", name="" + str(seed) + "_cav_waiting_list_" + path[-12:-8])

    route_length_list = tripinfo.get_tripinfo_list('routeLength')
    duration_list = tripinfo.get_tripinfo_list()
    delay_list = tripinfo.get_tripinfo_list('timeLoss')
    waiting_list = tripinfo.get_tripinfo_list('waitingTime')
    o.csv_av(route_length_list, file=f"{algo}/{rate}/", name="" + str(seed) + "_route_length_list_" + path[-12:-8])
    o.csv_av(duration_list, file=f"{algo}/{rate}/", name="" + str(seed) + "_duration_list_" + path[-12:-8])
    o.csv_av(delay_list, file=f"{algo}/{rate}/", name="" + str(seed) + "_delay_list_" + path[-12:-8])
    o.csv_av(waiting_list, file=f"{algo}/{rate}/", name="" + str(seed) + "_waiting_list_" + path[-12:-8])

    o.csv_av(waiting, file=f"{algo}/{rate}/", name="" + str(seed) + "_loss_" + path[-12:-8])
    o.csv_av(duration, file=f"{algo}/{rate}/", name="" + str(seed) + "_duration_" + path[-12:-8])
    # aw_length = [tripinfo.get_avg_qlength()]
    throughput = [tripinfo.get_tripinfo("duration", True)]
    routelength = [tripinfo.get_tripinfo("routeLength", False)]
    waitingcount = [tripinfo.get_tripinfo("waitingCount", False)]
    waitingtime = [tripinfo.get_tripinfo("waitingTime", False)]
    emission_CO2 = [tripinfo.get_emission_info()]
    averagespeed = [tripinfo.get_avg_speed()]
    # arrivalrate = [total_arrived / len(cav_list)]

    o.csv_av(throughput, file=f"{algo}/{rate}/", name="" + str(seed) + "_throughput_" + path[-12:-8])
    # o.csv_av(aw_length, file=f"{algo}/{rate}/", name="" + str(seed) + "_qlength_" + path[-12:-8])
    o.csv_av(routelength, file=f"{algo}/{rate}/", name="" + str(seed) + "_routelength_" + path[-12:-8])
    # o.csv_av(arrivalrate, file=f"{algo}/{rate}/", name="" + str(seed) + "_arrivalrate_" + path[-12:-8])
    o.csv_av(waitingcount, file=f"{algo}/{rate}/", name="" + str(seed) + "_waitingcount_" + path[-12:-8])
    o.csv_av(waitingtime, file=f"{algo}/{rate}/", name="" + str(seed) + "_waiting_" + path[-12:-8])
    o.csv_av(averagespeed, file=f"{algo}/{rate}/", name="" + str(seed) + "_avgspeed_" + path[-12:-8])

    o.csv_av(emission_CO2, file=f"{algo}/{rate}/", name="" + str(seed) + "_CO2_" + path[-12:-8])


def astar_dqn_base(rt=0.5,cav_seed=8):
    fname = "0.5_astar_dqn_0315/best_model"
    exe_flow(exe=False, name=fname, path='./res/hangzhou/exe_2983.sumocfg', ratio=rt, cav_seed=8)

def self_org_base(rt=0.5):
    fname = "0.5_self_org_0315/best_model"
    exe_flow(exe=False, name=fname, path='./res/hangzhou/exe_2983.sumocfg', ratio=rt, cav_seed=8,algo="self_org")

def nav_base(rt=0.5):
    fname = "0.5_nav_0315/best_model"
    exe_flow(exe=False, name=fname, path='./res/hangzhou/exe_2983.sumocfg', ratio=rt, cav_seed=8,algo="nav")
    
def dso_base(rt=0.5):
    fname = "0.5_dso_0315/best_model"
    exe_flow(exe=False, name=fname, path='./res/hangzhou/exe_2983.sumocfg', ratio=rt, cav_seed=8,algo="dso")

if __name__ == '__main__':
    torch.manual_seed(config.SIMULATION["SEED"])
    np.random.seed(config.SIMULATION["SEED"])
    #for rt in [0.5]:
     #for cav_seed in range(8, 108, 10):

    #    comment = ""  # final state/ exp decay
    #    name = main(exe=False, comment=comment, light_path=None, ratio=rt)
    #    print("****************************EXEC****************************")
    #test_result_waiting_2983 = []
    #test_result_waiting_6984 = []
    #test_result_waitingcount_2983 = []
    #test_result_waitingcount_6984 = []
    #test_result_avgspeed_2983 = []
    #test_result_avgspeed_6984 = []

    #fname = name+'/best_model'
    #print(fname)
    #exe_flow(exe=False, name=fname, path='./res/hangzhou/exe_2983.sumocfg', ratio=rt, cav_seed=8)
    #astar_dqn_base() #baseline1 A*
    #greedy_strategy(path='./res/hangzhou/exe_2983.sumocfg',algo="dijkstra")  # baseline2 dij
    #self_org_base() #baseline3 self_org
    #nav_base() #baseline4 NAV
    dso_base() #baseline5 DSO