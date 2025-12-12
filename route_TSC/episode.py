import numpy as np
import time
import pandas as pd
import os
import traci
import copy

import config
import env
import tripinfo
import traffic_light as tl
import torch
from net import DQN
import random

# from torch.utils.tensorboard import SummaryWriter


# tm = str(time.strftime('%Y-%m-%d-%H-%M', time.localtime()))
# os.makedirs(f'./log/ENV/{tm}', exist_ok=True)
# writer = SummaryWriter(f'./log/ENV/{tm}')
writer_step = 0
ep_writer_step = 0


def run(ep, agents, o, args, execute, name=None, router=None, adj_edge=None, gen_cav=[]):
    if args.algo == "self_org":
        from algo.self_org_agent import SelfOrgAgent as CAVAgent
    elif args.algo == "iql_b":
        from algo.iql_b_agent import IQLBAgent as CAVAgent
    elif args.algo == "astar_dqn":
        from algo.astar_dqn import AstarDQN as CAVAgent
    elif args.algo == "adaptive":
        from algo.adaptive_dqn import AdaptiveDQN as CAVAgent
    elif args.algo == "nav":
        from algo.nav_agent import NavAgent as CAVAgent
    elif args.algo == "dso":
        from algo.dso_agent import DSOAgent as CAVAgent
    else:
        from agent import CAVAgent as CAVAgent

    global writer_step, ep_writer_step
    # print(writer_step, ep_writer_step)
    config.SIMULATION["EP"] = ep
    # epsilon decay
    config.RL["EPSILON"] = max(config.RL["EPSILON"] * pow(config.RL["EPSILON_DECAY_RATE"], config.SIMULATION["EP"]),
                               config.RL["MIN_EPSILON"])
    if args.l2v:
        config.RL["EPSILON"] = config.RL["MIN_EPSILON"]
    config.ROUTER_RL["EPSILON"] = max(config.ROUTER_RL["EPSILON"] * pow(config.ROUTER_RL["EPSILON_DECAY_RATE"],
                                      config.SIMULATION["EP"]), config.ROUTER_RL["MIN_EPSILON"])

    if execute:
        config.ROUTER_RL["EPSILON"] = 0
        config.RL["EPSILON"] = 0

    print(f'epsilon: {config.RL["EPSILON"]}, {config.ROUTER_RL["EPSILON"]}')
    rewards = []

    for i in range(len(agents.agent_list)):
        agent = agents.agent_list[i]
        state = env.get_state(agent, agent.obs_name, agent_list=agents.agent_list)
        # agent.action = agent.rl_model.act(np.array(state))
        agent.action = 0
        agent.state = state
    sumo_duration = 12600 if execute else 3600

    # for i in range(sumo_duration):
    #     reward, terminated = agents.d_step(tm=i)

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

    for t in range(3600):
        road_state = env.get_edge_state(agents.agent_list, road_list)
        traffic_light_state = env.get_light_state(agents.agent_list)

        for cav in cav_list:
            if cav.veh_id not in leaving_cav_set and not cav.arrived() and not cav.done and cav.is_valid():
                cav_state = cav.get_router_state2(copy.deepcopy(road_state), traffic_light_state, road_list, greens, light_count, lane_dict)
                router_avail_action = cav.get_avail_action()
                if config.ROUTER_RL["ALGO"] == "DQN":
                    if args.algo == "nav":
                        action = cav.router.act(np.array(cav_state), avail_actions=cav.db)
                    else:
                        action = cav.router.act(np.array(cav_state), avail_actions=router_avail_action)
                if cav.act:
                    reward = cav.get_reward(road_state=road_state)
                    cav.append_reward(reward)
                    if cav.reward[0] < 0 or args.algo != "hatt_router":
                        cav.router.store(cav.cav_state, reward, cav_state, done=cav.done, actions=cav.action)
                        if cav.router.buf_cnt == args.threshold:
                            print("***************************** Training! **********************************")
                    cav.act = False
                if cav.done:
                    if len(cav.reward) > 0:
                        cav_step_rewards.append(sum(cav.reward) / len(cav.reward))
                        # print(cav.veh_id, cav.reward, cav.default_route, traci.vehicle.getRoute(cav.veh_id))
                    # total_arrived += 1
                cav.step(action)
                cav.cav_state = cav_state

        for _t in range(1):
            if args.router and t > 0:
                loaded_list = list(traci.simulation.getDepartedIDList())
                if len(loaded_list) > 0 and len(gen_cav) > 0:
                    cav_list.extend([CAVAgent(cav, router, adj_edge, args) for cav in loaded_list if int(cav) in set(gen_cav)])
                elif len(loaded_list) > 0 and len(gen_cav) == 0:
                    cav_list.extend([CAVAgent(cav, router, adj_edge, args) for cav in loaded_list if random.random() < rate])
            env.step()
            if args.router and t > 0:
                leaving_cav_set = leaving_cav_set | set(traci.simulation.getArrivedIDList())

        if not execute:
            rewards.append(0)
        if t % 3 == 0 and not execute:
            _loss = router.learn()
            if _loss:
                loss.append(_loss / config.ROUTER_RL["BATCH_SIZE"])

    for cav in cav_list:
        if cav.veh_id not in leaving_cav_set and not cav.done and len(cav.reward) > 0:
            cav_step_rewards.append(sum(cav.reward) / len(cav.reward))
    cav_episode_rewards.extend(cav_step_rewards)

    if args.router and config.ROUTER_RL["ALGO"] == "PPO" and not execute:
        router.learn()
        router.replay_buffers = []

    # if args.router and config.ROUTER_RL["ALGO"] == "DQN" and not execute:
    #     for _ in range(1200):
    #         router.learn()

    if args.router and args.decay and router.training and not execute:
        if config.ROUTER_RL["ALGO"] == "DQN":
            router.scheduler.step()
            print("lr", router.optimizer.param_groups[0]['lr'])
        else:
            pass

    for cav in cav_list:
        if cav.veh_id in leaving_cav_set:
            total_arrived += 1

    print("Loaded:", len(cav_list))
    print("Arrived:", total_arrived)
    env.close()

    episode_durations.append(tripinfo.get_tripinfo('duration'))
    # writer.add_scalar(f'reward/ep', np.array(cav_episode_rewards).mean(axis=0)[0], ep_writer_step)
    ep_writer_step += 1

    if not execute:
        if ep == 0:
            # step reward
            # df = pd.DataFrame(np.array(episode_rewards))
            # df.to_csv(f'./simudata/{name}/episode_rewards.csv')
            # average reward
            # df = pd.DataFrame(np.array(episode_rewards).mean(axis=0)).T
            # df.to_csv(f'./simudata/{name}/average_episode_rewards.csv')

            # cav step reward
            df = pd.DataFrame(np.array(cav_episode_rewards))
            df.to_csv(f'./simudata/{name}/cav_episode_rewards.csv')
            # cav average reward
            df = pd.DataFrame([np.array(cav_episode_rewards).mean(axis=0)]).T
            df.to_csv(f'./simudata/{name}/cav_average_episode_rewards.csv')

            # duration
            df = pd.DataFrame(np.array(episode_durations))
            df.to_csv(f'./simudata/{name}/episode_durations.csv')
        else:
            # _episode_rewards = pd.read_csv(f'./simudata/{name}/episode_rewards.csv').values[:, 1:]
            # _episode_rewards = np.vstack((_episode_rewards, np.array(episode_rewards)))
            # df = pd.DataFrame(_episode_rewards)
            # df.to_csv(f'./simudata/{name}/episode_rewards.csv')

            # _average_episode_rewards = pd.read_csv(f'./simudata/{name}/average_episode_rewards.csv').values[:, 1:]
            # _average_episode_rewards = np.vstack((_average_episode_rewards, np.array(episode_rewards).mean(axis=0)))
            # df = pd.DataFrame(_average_episode_rewards)
            # df.to_csv(f'./simudata/{name}/average_episode_rewards.csv')

            _cav_episode_rewards = pd.read_csv(f'./simudata/{name}/cav_episode_rewards.csv').values[:, 1:]
            _cav_episode_rewards = np.vstack((_cav_episode_rewards, np.array([cav_episode_rewards]).transpose()))
            df = pd.DataFrame(_cav_episode_rewards)
            df.to_csv(f'./simudata/{name}/cav_episode_rewards.csv')

            _cav_average_episode_rewards = pd.read_csv(f'./simudata/{name}/cav_average_episode_rewards.csv').values[:, 1:]
            _cav_average_episode_rewards = np.vstack((_cav_average_episode_rewards, np.array(cav_episode_rewards).mean(axis=0)))
            df = pd.DataFrame(_cav_average_episode_rewards)
            df.to_csv(f'./simudata/{name}/cav_average_episode_rewards.csv')

            _episode_duration = pd.read_csv(f'./simudata/{name}/episode_durations.csv').values[:, 1:]
            _episode_duration = np.vstack((_episode_duration, np.array(episode_durations)))
            df = pd.DataFrame(_episode_duration)
            df.to_csv(f'./simudata/{name}/episode_durations.csv')

    return np.array(rewards), tripinfo.get_tripinfo('duration'), tripinfo.get_tripinfo('waitingCount'), \
           tripinfo.get_tripinfo('arrivalSpeed'), tripinfo.get_tripinfo('timeLoss'), \
           tripinfo.get_tripinfo('waitingTime'), tripinfo.get_tripinfo('departDelay'), total_arrived / len(cav_list), loss

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

    for t in range(3600):
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