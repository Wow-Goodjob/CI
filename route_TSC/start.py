import os
import shutil
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
import PPO
import Causal

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
    print(name)
    name = f'{ratio}_{args.algo}_joint_0315'
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
        agents.router_net=RouterNet
        agents.adj_edge=adj_edge
        agents.joint_training=args.joint_training

    max_t = 999
    loss = []

    if idx > 0:
        #RouterNet.load_model(name+'/ep50')
        RouterNet.load_model('./model/0.5_astar_dqn_0315')
        print("########### LOADED ###########")
    ppo_agent = PPO.PPOAgent(state_dim=44, action_dim=4, lr=0.00005, K_epochs=10)
    for i in range(idx, args.episode):
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

        reward, t, c, s, l, w, d, rt, _loss = episode.run(i, agents, o, args, execute, name, RouterNet, adj_edge, gen_cav,ppo_agent)
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
            if args.router or args.joint_training:
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

if __name__ == '__main__':
    torch.manual_seed(config.SIMULATION["SEED"])
    np.random.seed(config.SIMULATION["SEED"])


    main(exe=False,name=None,idx=0,seed='25',light_path=None,ratio=0.5)
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
    #dso_base() #baseline5 DSO