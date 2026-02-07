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

    args.joint_training=True
    args.router_train_freq=3

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
    name = f'{ratio}_{args.algo}_joint_0315'
    print('run:', name)
    ap.add_argument("--dir", type=str, default=name)
    if execute:
        pass
        # agents.load_model(name)

    RouterNet = None
    adj_edge = None
    if args.router or args.joint_training:
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

        if args.joint_training:
            reward, t, c, s, l, w, d, rt, _loss=episode.run_joint(i, agents, o, args, execute, name, RouterNet, adj_edge, gen_cav)
        else:
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

def run_joint(ep, agents, o, args, execute, name=None, router=None, adj_edge=None, gen_cav=[]):
    if args.algo=="self_org":
        from algo.self_org_agent import SelfOrgAgent as CAVAgent
    elif args.algo=="astar_dqn":
        from algo.astar_dqn import AstarDQN as CAVAgent
    else:
        from agent import CAVAgent as CAVAgent

    global writer_step, ep_writer_step
    config.SIMULATION["EP"]=ep

    config.RL["EPSILON"] = max(config.RL["EPSILON"] * pow(config.RL["EPSILON_DECAY_RATE"], config.SIMULATION["EP"]),
                               config.RL["MIN_EPSILON"])
    config.ROUTER_RL["EPSILON"] = max(config.ROUTER_RL["EPSILON"] * pow(config.ROUTER_RL["EPSILON_DECAY_RATE"],
                                                                        config.SIMULATION["EP"]),
                                      config.ROUTER_RL["MIN_EPSILON"])

    if execute:
        config.ROUTER_RL["EPSILON"] = 0
        config.RL["EPSILON"] = 0

    print(f'epsilon: {config.RL["EPSILON"]}, router_epsilon: {config.ROUTER_RL["EPSILON"]}')

    for i in range(len(agents.agent_list)):
        agent=agents.agent_list[i]
        state=env.get_state(agent,agent.obs_name,agent_list=agents.agent_list)
        agent.action=0
        agent.state=state
    sumo_duration = 12600 if execute else 3600
    '''total_arrived=0
    rewards=[]
    cav_episode_rewards=[]
    loss=[]'''
    rewards=[]
    total_arrived = 0
    actions = [0] * args.agent_num
    cur_actions = [0] * args.agent_num
    states = []
    next_states = []
    episode_rewards = []
    cav_episode_rewards = []
    episode_durations = []
    average_episode_rewards = []
    for agent in agents.agent_list:  # initial state
        states.append(env.get_length_state(agent, green=0))
        # states.append(env.get_state(agent, state_type=agent.obs_name, agent_list=agents.agent_list))

    rate = args.rate  # CAV penetration rate
    cav_list = []
    leaving_cav_set = set()
    lanes = traci.lane.getIDList()
    lanes = [lane_id for lane_id in lanes if ":" not in lane_id]
    road_list, light_count, lane_dict = env.get_map_lanes(agents.agent_list, lanes)
    veh_id_list = dict()
    loaded_veh = 0
    greens = [0] * args.agent_num
    cav_step_rewards = []
    loss = []

    for t in range(1800):
        reward,terminated=agents.d_step(tm=t)
        rewards.append(reward)

        if args.joint_training and router:
            road_state = env.get_edge_state(agents.agent_list,road_list)
            traffic_light_state = env.get_light_state(agents.agent_list)

            loaded_list = list(traci.simulation.getDepartedIDList())
            if len(loaded_list) > 0 and len(gen_cav) > 0:
                agents.cav_list.extend([CAVAgent(cav, router, adj_edge, args)
                                        for cav in loaded_list if int(cav) in set(gen_cav)])

            cav_step_rewards = []
            for cav in agents.cav_list:
                if (cav.veh_id not in agents.leaving_cav_set and not cav.arrived()
                        and not cav.done and cav.is_valid()):

                    cav_state = cav.get_router_state2(
                        copy.deepcopy(road_state), traffic_light_state,
                        agents.road_list, agents.greens, agents.light_count, agents.lane_dict
                    )
                    router_avail_action = cav.get_avail_action()

                    if config.ROUTER_RL["ALGO"] == "DQN":
                        action = cav.router.act(np.array(cav_state), avail_actions=router_avail_action)

                    if cav.act:
                        reward = cav.get_reward(road_state=road_state)
                        cav.append_reward(reward)
                        if cav.reward[0] < 0:
                            cav.router.store(cav.cav_state, reward, cav_state, done=cav.done, actions=cav.action)

                    cav.step(action)
                    cav.cav_state = cav_state

                if cav.done and len(cav.reward) > 0:
                    cav_step_rewards.append(sum(cav.reward) / len(cav.reward))

            cav_episode_rewards.extend(cav_step_rewards)

            if not execute and t % 3 == 0:
                _loss = router.learn()
                if _loss:
                    loss.append(_loss / config.ROUTER_RL["BATCH_SIZE"])

        env.step()

        if args.joint_training:
            agents.leaving_cav_set = agents.leaving_cav_set | set(traci.simulation.getArrivedIDList())

    for cav in agents.cav_list:
        if cav.veh_id in agents.leaving_cav_set:
            total_arrived += 1

    print("Loaded CAVs:", len(agents.cav_list))
    print("Arrived CAVs:", total_arrived)
    env.close()

    return  (np.array(rewards), tripinfo.get_tripinfo('duration'),
            tripinfo.get_tripinfo('waitingCount'), tripinfo.get_tripinfo('arrivalSpeed'),
            tripinfo.get_tripinfo('timeLoss'), tripinfo.get_tripinfo('waitingTime'),
            tripinfo.get_tripinfo('departDelay'), total_arrived, loss)

def run_route_signal_ppo(env,cav_list,traffic_lights,router_agents,signal_agents,config,tripinfo):
    SIM_TIME=1800
    SIGNAL_INTERVAL=10
    rewards=[]
    total_arrived=0

    last_signal_transition={}

    for t in range(SIM_TIME):
        env.step()
        if t%SIGNAL_INTERVAL==0:
            for tl_id in traffic_lights:
                state=env.get_light_state(tl_id)
                action,logp,value=signal_agents[tl_id].act(state)
                env.set_traffic_light(tl_id,action)
                last_signal_transition[tl_id]={"state":state,"action":action,"logp":logp,"value":value,"reward":None,"done":False}
            for cav in cav_list:
                if cav.done or cav.arrived():
                    continue
                if not cav.is_valid():
                    continue

'''
    def write(self, s, a, a_logprob, r, s_, dw, done, id=-1):
        # self.replay_buffer.store(s, self.a, self.a_logprob, r, s_, dw, done)
        if len(self.replay_buffers) <= self.total_steps:
            self.replay_buffers.append(PPOReplayBuffer(self.args, self.state_dim))
        self.replay_buffers[self.total_steps].store(s, a, a_logprob, r, s_, dw, done)
        self.total_steps += 1
        self.a = a
        self.a_logprob = a_logprob
'''