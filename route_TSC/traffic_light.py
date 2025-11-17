import numpy as np
import torch
# from torch.utils.tensorboard import SummaryWriter

import config
import env
from dqnagent import DQNAgent
from ppoagent import PPOAgent


# writer = SummaryWriter()


class Light:
    def __init__(self, tl_id, dic, dic_lane, dic_down, dic_up, _light, args, execution=False):
        self.tl_id = tl_id
        self.dic = dic
        self.dic_lane = dic_lane
        self.dic_down = dic_down
        self.dic_up = dic_up
        self.neighbor = None
        self.neighbors = []
        self.args = args
        self._light = _light
        self.waiting = 0
        self.action_dim = args.action_dim
        self.tl_log = list()
        self.reward = list()
        self.length = list()
        self.phase = 0
        self.phase_dim = args.phase_dim
        self.action = 0
        self.next_action = 0

    def add_neighbor(self, tl_id):
        self.neighbors.append(tl_id)

    def step(self, action_set):
        _, reward = env.get_reward(self)
        self.reward.append(reward)
        self.write_log()
        return reward, action_set

    def write_log(self):
        self.tl_log.append(self.get_action())

    def get_dic(self):
        return self.dic

    def get_id(self):
        return self.tl_id

    def set_waiting(self, waiting):
        self.waiting = waiting

    def get_action(self):
        return self.action

    def get_next_action(self):
        return self.next_action

    def get_reward(self):
        return self.reward

    def get_length(self):
        return self.length

    def p2a(self, phase):
        a = [0] * self.phase_dim
        a[phase] = 1
        return np.float32(a)

    def write_reward(self, reward):
        self.reward.append(reward)

    def write_length(self, length):
        self.length.append(length)


# implement single agent
class TLight(Light):
    def __init__(self, tl_id, dic, dic_lane, dic_down, dic_up, _light, args, pfrl=False, execution=False):
        super(TLight, self).__init__(tl_id, dic, dic_lane, dic_down, dic_up, _light, args)
        self.pfrl = pfrl
        self.junction = config.INTERSECTION["JUNCTION"]
        self.agent_type = args.agent_type
        self.action_dim = args.action_dim
        self.state_dim = args.state_dim
        self.phase_dim = args.phase_dim
        self.i_state_dim = self.state_dim
        # self.n_state_dim = self.state_dim * (self.neighbor + 1)
        self.n_state_dim = self.state_dim
        self.g_state_dim = self.state_dim * self.junction
        self.state_type = {
            "independent": self.state_dim,
            # "local": self.state_dim * (self.neighbor + 1),
            "local": self.state_dim,
            "global": self.state_dim * self.junction
        }
        # if args.spatial == "GAT":
        # self.obs_name = "local"
        # self.sta_name = "local"
        # else:
        self.obs_name = args.cop
        self.sta_name = "independent"
        # if args.agent_type == "QMIX":
        #     self.obs_name = "global"
        if self.agent_type == "DQN":
            self.rl_model = DQNAgent(self.state_type[self.obs_name], self.action_dim, args, tl_id, "q_net", n_steps=1,
                                     execution=execution, net_config=config.RL)
        elif self.agent_type == "PPO":
            self.rl_model = PPOAgent(self.state_type[self.obs_name], self.action_dim, args, tl_id, "ac_net", n_steps=1,
                                     execution=execution)
        elif self.agent_type == "MAT":
            self.rl_model = MATAgent(self.state_type[self.obs_name], self.action_dim, args, tl_id, "ac_net", n_steps=1,
                                     execution=execution)
        elif self.agent_type == "MAPPO":
            self.rl_model = MAPPOAgent(self.state_type[self.obs_name], self.action_dim, args, tl_id, "ac_net",
                                       n_steps=1, execution=execution)
        elif self.agent_type == "MA2C":
            self.rl_model = MA2CAgent(self.state_type[self.obs_name], self.action_dim, args, tl_id, "ac_net", n_steps=1,
                                      execution=execution)
        elif self.agent_type == "CCGN":
            self.rl_model = CCGNAgent(self.state_type[self.obs_name], self.action_dim, args, tl_id, "ac_net", n_steps=1,
                                      execution=execution)
        elif self.agent_type == "CenPPO":
            self.rl_model = CenPPOAgent(self.state_type[self.obs_name], self.action_dim, args, tl_id, "ac_net",
                                        n_steps=1, execution=execution)
        elif self.agent_type == "PS-PPO":
            self.rl_model = PPOAgent(self.state_type[self.obs_name], self.action_dim, args, tl_id, "ac_net", n_steps=1,
                                     execution=execution)

        # traffic light module
        self.state = []
        self.next_state = []
        self.global_state = []
        self.local_state = []
        self.state_time = 0
        self.last_change_time = 0
        self.loss = list()
        self.history = args.history
        self.s_series = list()  # state series
        self.s_index = 0  # index of cur state
        self.his_flag = True  # history flag

        self.iter = 0
        self.cul_time = 10

        self.green_duration = 15
        self.green = 15
        self.yellow = 3
        self.unit = 15

        # for fixed circle
        self.phase = 0
        self.phase_length = 0  # set [0, 10]
        self.min_phase_length = 5

        # MAT
        self.ep_step = 0
        self.is_train = False

        # CCGN
        self.his_speed = []

        # self.changed = 0
        # self.unchanged = 0

    def step(self, action_set, agent_list=None, q_values=None, global_state=None, mat_action=-1):
        agent = self.rl_model
        duration, self.action, phase_type = action_set
        duration -= 1
        length, reward = 3, np.float64(0.0)
        done = 1 if length < 2 else 0
        # calculate queue length on each second
        if self.args.episode == 1:
            queue_length = 0.0
            for lane in env.get_control_lanes(self):
                queue_length += env.get_number_of_queued_vehicles(lane)
            self.write_length(queue_length)

        if duration == 0:
            if self.args.agent_type in {"DQN", "PPO"}:
                length, reward = env.get_reward(self)  # MAT cal global reward outside
            # elif self.args.agent_type in {"CCGN"}:
            #     reward = env.get_ccgn_global_reward(agent_list)
            if self.cul_time > config.INTERSECTION["THRESHOLD"] + 1e5:  # pass
                self.next_action = (self.next_action + 1) % self.phase_dim  # action == next action so time grow over
                phase_type = 'Y'
                duration = self.yellow
                self.cul_time = 0
            elif phase_type == 'G':
                self.is_train = True
                if self.args.agent_type in {"DQN", "PPO"}:
                    next_state = np.array(env.get_state(self, state_type=self.obs_name, agent_list=agent_list))
                # elif self.args.agent_type in {"CCGN"}:
                #     next_state = np.array(env.get_ccgn_state(agent_list))
                self.green_duration = self.green
                if self.agent_type in {"DQN"}:
                    self.next_action = agent.act(next_state)
                elif self.agent_type in {"CCGN"}:
                    self.next_action = agent.act(global_state, agent_list[0].rl_model.q_network)
                elif self.agent_type == "PPO":
                    a, a_logprob = agent.choose_action(next_state)
                    self.next_action = a
                elif self.agent_type in {"MAT", "MAPPO"}:
                    self.next_action = mat_action[0]
                elif self.agent_type in {"MA2C"}:
                    self.next_action = mat_action
                elif self.agent_type == "SAC":
                    self.next_action = agent.get_actions(next_state)
                elif self.agent_type in {"CenPPO", "PS-PPO"}:
                    self.next_action = mat_action
                    # self.ep_step += (self.ep_step + 1) % self.args.episode_length
                # default: switch phase, set Y
                phase_type = 'Y'
                duration = self.yellow
                # self.next_action = agent.act(next_state)
                if (self.args.control_type == 'adapt' or self.args.control_type == 'syn') \
                        and self.action == self.next_action:  # keep the current phase
                    phase_type = 'G'
                    duration = self.green
                    self.cul_time += self.green

                if self.action != self.next_action:
                    phase_type = 'Y'
                    duration = self.yellow
                else:
                    duration = self.unit
                    self.cul_time += self.unit

                # if env.get_time() >= 5400 and env.get_time() <= 9000:
                #     if self.action != self.next_action:
                #         self.changed += 1
                #     phase_type = 'Y'
                #     duration = self.yellow
                # else:
                #     self.unchanged += 1
                #     duration = self.unit
                #     self.cul_time += self.unit

                if not agent.execution:
                    if self.args.agent_type in {'DQN'}:
                        agent.store(self.state, reward, next_state, done)
                    elif self.args.agent_type == "PPO":
                        agent.write(self.state, a=a, a_logprob=a_logprob, r=reward, s_=next_state, dw=False, done=done)
                    elif self.args.agent_type == "MAT":  # on-policy
                        pass
                    elif self.args.agent_type == "SAC":
                        pass
                    elif self.args.agent_type == "MAPPO":
                        pass

                # self.s_index, self.his_flag = agent.update_w(self.s_series, self.s_index, next_state)

                if self.args.agent_type not in {"MAT", "MAPPO", "MA2C", "CCGN", "CenPPO", "PS-PPO"}:
                    self.state = next_state
                if not agent.execution and self.args.agent_type not in {"MAT", "MAPPO", "MA2C", "CCGN", "CenPPO",
                                                                        "PS-PPO"}:
                    self.write_log()
                    self.write_reward(reward)
                if not agent.execution and self.his_flag and self.agent_type not in {"MAT", "MAPPO", "MA2C", "CCGN",
                                                                                     "CenPPO", "PS-PPO"}:
                    # pass
                    agent.learn()
                    # self.loss.append(loss)
                    # writer.add_scalar(self.tl_id + "_critic_loss", loss, self.iter)
                    self.iter += 1
                    # writer.flush()
            elif phase_type == 'Y':
                # switch to G
                phase_type = 'G'
                duration = self.green_duration  # fixed green duration: 15s

                if self.args.control_type == "syn" and self.action != self.next_action:
                    duration = self.green_duration - self.yellow  # fixed decision slag: 15s(Y+G)

                self.action = self.next_action
                self.phase = (self.phase + 1) % 4
                self.cul_time = duration
            # fixed phase circle
            if self.args.control_type == "fixed":
                duration = self.yellow if phase_type == 'Y' else self.min_phase_length + self.action
        if self.args.control_type == "fixed":
            return reward, (duration, self.phase, phase_type)
        # print(self.tl_id, (duration, self.action, phase_type))
        return reward, (duration, self.action, phase_type)

    def get_action(self):
        return [[self.action]]
        # return self.rl_model.get_embedding_action(self.action)

    def state_emb(self, state):
        return self.rl_model.get_embedding_state(state)


class MaxPressureTSC(Light):
    def __init__(self, tl_id, dic, dic_lane, dic_down, dic_up, _light, args):
        super(MaxPressureTSC, self).__init__(tl_id, dic, dic_lane, dic_down, dic_up, _light, args)
        self.tl_id = tl_id
        self.dic = dic
        self.dic_lane = dic_lane
        self.dic_down = dic_down
        self.dic_up = dic_up
        self._light = _light
        self.p = self.phase_map_lanes()
        self.duration = 15
        self.phase = 0
        self.next_phase = 0
        self.green = 15
        self.yellow = 3
        self.signal = 'G'
        self.last_phase_id = 3
        self.greens = [[], [], [], []]

    def get_id(self):
        return self.tl_id

    def set_phase(self, phase=None):
        # execute cur phase
        if phase is None:
            env.set_phase(self, self.next_phase)
            self.signal = 'G'
            self.duration = self.green
            self.phase = self.next_phase
        # phase selection
        else:
            # same phase
            if phase == self.phase:
                self.duration = self.green
                env.set_phase(self, 2 * self.phase)
                self.signal = 'G'
            # switch phase
            else:
                self.duration = self.yellow
                env.set_phase(self, 2 * self.phase + 1)
                self.signal = 'Y'
                self.next_phase = phase

    def get_lanes(self):
        print(env.get_control_lanes(self.tl_id))

    def phase_map_lanes(self):
        nei = config.NEIGHBOR[self.tl_id]
        cur = self.tl_id[4]
        p = [[], [], [], []]
        # NS NSL EW EWL
        # if self.tl_id in ["node2", "node5", "node8"]:
        #   p[0].append([nei[1] + '_' + cur + ('_i_1' if len(nei[1]) < 2 else '_1'), cur + '_' + nei[3] + ('_o_1' if len(nei[3]) < 2 else '_1')])
        #   p[0].append([nei[3] + '_' + cur + ('_i_1' if len(nei[3]) < 2 else '_1'), cur + '_' + nei[1] + ('_o_1' if len(nei[1]) < 2 else '_1')])
        #   p[1].append([nei[1] + '_' + cur + ('_i_2' if len(nei[1]) < 2 else '_2'), cur + '_' + nei[3] + ('_o_1' if len(nei[3]) < 2 else '_2')])
        #   p[1].append([nei[3] + '_' + cur + ('_i_2' if len(nei[3]) < 2 else '_2'), cur + '_' + nei[1] + ('_o_1' if len(nei[1]) < 2 else '_2')])
        #   p[2].append([nei[0] + '_' + cur + ('_i_1' if len(nei[0]) < 2 else '_1'), cur + '_' + nei[2] + ('_o_1' if len(nei[2]) < 2 else '_1')])
        #   p[2].append([nei[2] + '_' + cur + ('_i_1' if len(nei[2]) < 2 else '_1'), cur + '_' + nei[0] + ('_o_1' if len(nei[0]) < 2 else '_1')])
        #   p[3].append([nei[0] + '_' + cur + ('_i_2' if len(nei[0]) < 2 else '_2'), cur + '_' + nei[2] + ('_o_1' if len(nei[2]) < 2 else '_2')])
        #   p[3].append([nei[2] + '_' + cur + ('_i_2' if len(nei[2]) < 2 else '_2'), cur + '_' + nei[0] + ('_o_1' if len(nei[0]) < 2 else '_2')])
        # # EW EWL NS NSL
        # else:
        #   p[2].append([nei[1] + '_' + cur + ('_i_1' if len(nei[1]) < 2 else '_1'), cur + '_' + nei[3] + ('_o_1' if len(nei[3]) < 2 else '_1')])
        #   p[2].append([nei[3] + '_' + cur + ('_i_1' if len(nei[3]) < 2 else '_1'), cur + '_' + nei[1] + ('_o_1' if len(nei[1]) < 2 else '_1')])
        #   p[3].append([nei[1] + '_' + cur + ('_i_2' if len(nei[1]) < 2 else '_2'), cur + '_' + nei[3] + ('_o_1' if len(nei[3]) < 2 else '_2')])
        #   p[3].append([nei[3] + '_' + cur + ('_i_2' if len(nei[3]) < 2 else '_2'), cur + '_' + nei[1] + ('_o_1' if len(nei[1]) < 2 else '_2')])
        #   p[0].append([nei[0] + '_' + cur + ('_i_1' if len(nei[0]) < 2 else '_1'), cur + '_' + nei[2] + ('_o_1' if len(nei[2]) < 2 else '_1')])
        #   p[0].append([nei[2] + '_' + cur + ('_i_1' if len(nei[2]) < 2 else '_1'), cur + '_' + nei[0] + ('_o_1' if len(nei[0]) < 2 else '_1')])
        #   p[1].append([nei[0] + '_' + cur + ('_i_2' if len(nei[0]) < 2 else '_2'), cur + '_' + nei[2] + ('_o_1' if len(nei[2]) < 2 else '_2')])
        #   p[1].append([nei[2] + '_' + cur + ('_i_2' if len(nei[2]) < 2 else '_2'), cur + '_' + nei[0] + ('_o_1' if len(nei[0]) < 2 else '_2')])
        if self.tl_id in ["node2", "node5", "node8"]:
            p[0].append([nei[1] + '_' + cur + '_1', cur + '_' + nei[3] + '_1'])
            p[0].append([nei[3] + '_' + cur + '_1', cur + '_' + nei[1] + '_1'])
            p[1].append([nei[1] + '_' + cur + '_2', cur + '_' + nei[0] + '_2'])
            p[1].append([nei[3] + '_' + cur + '_2', cur + '_' + nei[2] + '_2'])
            p[2].append([nei[0] + '_' + cur + '_1', cur + '_' + nei[2] + '_1'])
            p[2].append([nei[2] + '_' + cur + '_1', cur + '_' + nei[0] + '_1'])
            p[3].append([nei[0] + '_' + cur + '_2', cur + '_' + nei[3] + '_2'])
            p[3].append([nei[2] + '_' + cur + '_2', cur + '_' + nei[1] + '_2'])
        # EW EWL NS NSL
        else:
            p[2].append([nei[1] + '_' + cur + '_1', cur + '_' + nei[3] + '_1'])
            p[2].append([nei[3] + '_' + cur + '_1', cur + '_' + nei[1] + '_1'])
            p[3].append([nei[1] + '_' + cur + '_2', cur + '_' + nei[0] + '_2'])
            p[3].append([nei[3] + '_' + cur + '_2', cur + '_' + nei[2] + '_2'])
            p[0].append([nei[0] + '_' + cur + '_1', cur + '_' + nei[2] + '_1'])
            p[0].append([nei[2] + '_' + cur + '_1', cur + '_' + nei[0] + '_1'])
            p[1].append([nei[0] + '_' + cur + '_2', cur + '_' + nei[3] + '_2'])
            p[1].append([nei[2] + '_' + cur + '_2', cur + '_' + nei[1] + '_2'])

        return p
