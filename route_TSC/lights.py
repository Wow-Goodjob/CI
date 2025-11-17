import numpy as np

import config
import env
import traffic_light as tl


# implement global agent
class Lights:
    def __init__(self, agent_list, args, action_set, execute=False):
        self.agent_num = len(agent_list)
        self.agent_list = agent_list
        self.rl_model = None
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.agent_num = args.agent_num
        self.agent_type = args.agent_type
        self.args = args

        self.state = [0] * (self.state_dim * self.agent_num)
        self.action = [0] * (self.action_dim * self.agent_num)
        self.new_state = [0] * (self.state_dim * self.agent_num)
        self.global_state = None
        self.action_set = action_set

        # MAT info
        self.actions = None
        self.values = None
        self.action_log_probs = None
        self.rnn_states = None
        self.rnn_states_critic = None

        self.slot = 15
        self.rl_step = 0
        self.ob = None

        self.a = None
        self.a_logprob = None

    def step(self):
        rewards = 0

        # green
        for i in range(config.INTERSECTION["GREEN"]):
            for agent in self.agent_list:
                env.set_phase(agent, 2 * agent.action)
            env.step()
        # act
        for agent in self.agent_list:
            agent.next_action = agent.rl_model.act(np.array(agent.state))

        # yellow or green
        for i in range(config.INTERSECTION["YELLOW"]):
            for agent in self.agent_list:
                if agent.action != agent.next_action:
                    env.set_phase(agent, 2 * agent.action + 1)
                else:
                    env.set_phase(agent, 2 * agent.action)
            env.step()

        # learn
        for agent in self.agent_list:
            length, reward = env.get_reward(agent)
            rewards += reward
            agent.next_state = env.get_state(agent, agent.obs_name)
            agent.rl_model.store(agent.state, reward, agent.next_state, 1 if length < 2 else 0)
            if not agent.rl_model.execution:
                agent.rl_model.learn()
            agent.state = agent.next_state
            agent.action = agent.next_action
            agent.write_log()
            agent.write_reward(reward)

        return rewards / len(self.agent_list)

    def d_step(self, tm=-1):
        rewards = 0
        q_values = []
        is_terminated = False
        if self.args.agent_type == "MAT":  # CTCE
            c_model = self.agent_list[0]
            if tm == 0:  # init step
                self.values, self.actions, self.action_log_probs, self.rnn_states, self.rnn_states_critic = \
                    c_model.rl_model.trainer.collect(c_model.rl_model.trainer.buffer.step)
                for i, a in enumerate(self.action_set):
                    self.action_set[i] = (self.slot, self.actions[0][i][0], 'G')
            # step
            for i in range(len(self.agent_list)):
                res = self.agent_list[i].step(self.action_set[i], self.agent_list, mat_action=self.actions[0][i])
                reward = res[0]
                self.action_set[i] = res[1]
                duration, phase_index, phase_type = self.action_set[i]
                env.set_phase(self.agent_list[i], 2 * phase_index if phase_type == 'G' else 2 * phase_index + 1)
                rewards += reward
            env.step()
            if env.get_time() % self.slot == self.slot - 1:  # 15s-step, add transaction
                rewards = 0.0
                for agent in self.agent_list:
                    _, reward = env.get_reward(agent)
                    agent.write_log()
                    agent.write_reward(reward)
                    rewards += reward
                obs = np.array([env.get_global_state(self.agent_list)]).reshape(self.args.agent_num, -1)
                share_obs = np.array([env.get_global_state(self.agent_list)]).reshape(self.args.agent_num, -1)
                # obs = np.array([env.get_global_map_state(self.agent_list)]).reshape(self.args.agent_num, 128*128)
                # share_obs = np.array([env.get_global_map_state(self.agent_list)]).reshape(self.args.agent_num, 128*128)
                dones = np.array([[False] * 9])
                infos = ""
                available_actions = None
                data = obs, share_obs, rewards, dones, infos, available_actions, self.values, self.actions, \
                    self.action_log_probs, self.rnn_states, self.rnn_states_critic

                # insert data into buffer
                c_model.rl_model.trainer.insert(data)
                # c_model.is_train = False
                # if c_model.rl_model.trainer.buffer.step == self.args.episode_length - 1 and self.args.episode > 1:
                if c_model.rl_model.trainer.buffer.step == 0 and self.args.episode > 1:
                    print(f'Training: {env.get_time()}')
                    c_model.rl_model.trainer.compute()
                    c_model.rl_model.trainer.prep_training()
                    train_infos = c_model.rl_model.trainer.train(c_model.rl_model.trainer.buffer)
                    c_model.rl_model.buffer.after_update()
                    # c_model.rl_model.trainer.policy.scheduler.step()
                    print("lr", c_model.rl_model.trainer.policy.optimizer.param_groups[0]['lr'])
                    is_terminated = True
                self.values, self.actions, self.action_log_probs, self.rnn_states, self.rnn_states_critic = \
                    c_model.rl_model.trainer.collect(c_model.rl_model.trainer.buffer.step)  # next action

        elif self.args.agent_type == "HASAC":
            raise NotImplementedError(f'ERROR: {self.args.agent_type} not implemented!')

        elif self.args.agent_type == "HAPPO":
            raise NotImplementedError(f'ERROR: {self.args.agent_type} not implemented!')

        elif self.args.agent_type == "MA2C":
            c_model = self.agent_list[0]
            is_terminated = env.get_time() >= 3599

            if tm == 0:  # init step
                c_model.rl_model.trainer.reset()
                self.rl_step = 0
                # get obs
                self.ob = env.get_ma2c_state(self.agent_list)
                c_model.rl_model.ps = [np.ones(c_model.rl_model.n_a_ls[i]) / c_model.rl_model.n_a_ls[i] for i in
                                       range(self.args.agent_num)]
                # get policy and actions
                policy, self.actions = c_model.rl_model.get_policy(self.ob, is_terminated)
                c_model.rl_model.ps = policy
                # set phase
                for i, a in enumerate(self.action_set):
                    self.action_set[i] = (self.slot, self.actions[i], 'G')
            # step
            for i in range(len(self.agent_list)):
                reward, self.action_set[i] = self.agent_list[i].step(self.action_set[i], self.agent_list,
                                                                     mat_action=self.actions[i])
                duration, phase_index, phase_type = self.action_set[i]
                env.set_phase(self.agent_list[i], 2 * phase_index if phase_type == 'G' else 2 * phase_index + 1)
            env.step()
            if env.get_time() % self.slot == self.slot - 1:  # 15s-step, add transaction
                self.rl_step += 1
                # get reward
                reward = env.get_ma2c_reward(self.agent_list)
                # add transition
                is_terminated = env.get_time() >= 3599
                is_terminated = False
                self.values = c_model.rl_model.get_value(self.ob, is_terminated, self.actions)
                c_model.rl_model.trainer.add_transition(self.ob, c_model.rl_model.ps, self.actions,
                                                        reward, self.values, is_terminated)

                # next transition
                next_ob = env.get_ma2c_state(self.agent_list)
                self.ob = next_ob
                policy, self.actions = c_model.rl_model.get_policy(self.ob, is_terminated)
                c_model.rl_model.ps = policy

                # learn
                if self.args.episode > 1 and self.rl_step == self.args.buffer_size:
                    print(env.get_time())
                    dt = 240 - self.rl_step
                    if is_terminated:
                        R = np.zeros(self.args.agent_num)
                    else:
                        _, action = c_model.rl_model.get_policy(self.ob, is_terminated)
                        R = c_model.rl_model.get_value(self.ob, is_terminated, action)
                    c_model.rl_model.trainer.backward(R, dt)

        elif self.args.agent_type == "MAPPO":  # CTDE
            c_model = self.agent_list[0]
            if tm == 0:  # init step
                self.values, self.actions, self.action_log_probs, self.rnn_states, self.rnn_states_critic = \
                    c_model.rl_model.collect(c_model.rl_model.critic_buffer.step)
                for i, a in enumerate(self.action_set):
                    self.action_set[i] = (self.slot, self.actions[0][i][0], 'G')
            # step
            for i in range(len(self.agent_list)):
                res = self.agent_list[i].step(self.action_set[i], self.agent_list, mat_action=self.actions[0][i])
                reward = res[0]
                self.action_set[i] = res[1]
                duration, phase_index, phase_type = self.action_set[i]
                env.set_phase(self.agent_list[i], 2 * phase_index if phase_type == 'G' else 2 * phase_index + 1)
                rewards += reward
            env.step()
            if env.get_time() % self.slot == self.slot - 1:  # 15s-step, add transaction
                # print(env.get_time(), c_model.rl_model.critic_buffer.step)
                rewards = 0.0
                for agent in self.agent_list:
                    _, reward = env.get_reward(agent)
                    agent.write_log()
                    agent.write_reward(reward)
                    rewards += reward
                obs = np.array([env.get_global_state(self.agent_list)]).reshape(1, self.args.agent_num, -1)
                share_obs = np.array([env.get_global_state(self.agent_list)]).reshape(1, self.args.agent_num, -1)
                dones = np.array([[False] * 9])
                infos = [[{"bad_transition": False}]]
                available_actions = np.array([None] * 9)
                data = obs, share_obs, np.array(
                    [[rewards]]), dones, infos, available_actions, self.values, self.actions, \
                    self.action_log_probs, self.rnn_states, self.rnn_states_critic

                # insert data into buffer
                c_model.rl_model.insert(data)
                # c_model.is_train = False
                if c_model.rl_model.critic_buffer.step == 0 and self.args.episode > 1:
                    print(f'Training: {env.get_time()}, {c_model.rl_model.critic_buffer.step}')
                    c_model.rl_model.compute()
                    c_model.rl_model.prep_training()
                    train_infos = c_model.rl_model.train()
                    c_model.rl_model.after_update()
                    # c_model.rl_model.trainer.policy.scheduler.step()
                    is_terminated = True
                self.values, self.actions, self.action_log_probs, self.rnn_states, self.rnn_states_critic = \
                    c_model.rl_model.collect(c_model.rl_model.critic_buffer.step)  # next action

        elif self.args.agent_type == "CCGN":
            c_model = self.agent_list[0]
            action = np.zeros(self.args.agent_num)
            for i in range(len(self.agent_list)):
                reward, self.action_set[i] = self.agent_list[i].step(self.action_set[i], self.agent_list,
                                                                     global_state=self.global_state)
                duration, phase_index, phase_type = self.action_set[i]
                env.set_phase(self.agent_list[i], 2 * phase_index if phase_type == 'G' else 2 * phase_index + 1)
                action[i] = phase_index
            env.step()
            if env.get_time() % self.slot == self.slot - 1:
                rewards = env.get_ccgn_global_reward(self.agent_list)
                next_state = np.array(env.get_global_state(self.agent_list)).reshape(self.args.agent_num, -1)
                done = env.get_time() >= 3599

                c_model.rl_model.store(self.global_state, rewards, next_state, done, actions=action)
                c_model.rl_model.learn()
                self.global_state = next_state

        elif self.agent_type == "PS-PPO":
            c_model = self.agent_list[0]
            if tm == 0:  # init step
                a = np.zeros(9)
                a_logprob = np.zeros(9)
                for i in range(self.agent_num):
                    a[i], a_logprob[i] = c_model.rl_model.choose_action(self.global_state[i])

                self.a = a
                self.a_logprob = a_logprob
                c_model.rl_model.a = self.a
                c_model.rl_model.a_logprob = self.a_logprob

                for i, a in enumerate(self.action_set):
                    self.action_set[i] = (self.slot, self.a[i], 'G')

            rewards = np.zeros(self.args.agent_num)
            for i in range(len(self.agent_list)):
                reward, self.action_set[i] = \
                    self.agent_list[i].step(self.action_set[i], self.agent_list, global_state=self.global_state,
                                            mat_action=self.a[i])
                duration, phase_index, phase_type = self.action_set[i]
                env.set_phase(self.agent_list[i], 2 * phase_index if phase_type == 'G' else 2 * phase_index + 1)

            env.step()
            if env.get_time() % self.slot == self.slot - 1:
                next_state = np.array([env.get_global_state(self.agent_list)]).reshape(self.args.agent_num, -1)
                for i in range(len(self.agent_list)):
                    _, rewards[i] = env.get_reward(self.agent_list[i])
                done = env.get_time() >= 3599

                for i in range(self.agent_num):
                    self.agent_list[i].rl_model.write(self.global_state[i], a=self.a[i], a_logprob=self.a_logprob[i],
                                                      r=rewards[i], s_=next_state[i], dw=False, done=done)
                for i in range(self.agent_num):
                    c_model.rl_model.learn(buf=self.agent_list[i].rl_model.replay_buffer)
                self.global_state = next_state

                a = np.zeros(9)
                a_logprob = np.zeros(9)
                for i in range(self.agent_num):
                    a[i], a_logprob[i] = c_model.rl_model.choose_action(self.global_state[i])  # next action
                self.a = a
                self.a_logprob = a_logprob

        elif self.agent_type == "CenPPO":
            c_model = self.agent_list[0]
            if tm == 0:  # init step
                self.a, self.a_logprob = c_model.rl_model.choose_action(self.global_state)
                c_model.rl_model.a = self.a
                c_model.rl_model.a_logprob = self.a_logprob
                for i, a in enumerate(self.action_set):
                    self.action_set[i] = (self.slot, self.a[i], 'G')

            rewards = np.zeros(self.args.agent_num)
            for i in range(len(self.agent_list)):
                reward, self.action_set[i] = \
                    self.agent_list[i].step(self.action_set[i], self.agent_list, global_state=self.global_state,
                                            mat_action=self.a[i])
                duration, phase_index, phase_type = self.action_set[i]
                env.set_phase(self.agent_list[i], 2 * phase_index if phase_type == 'G' else 2 * phase_index + 1)
            # print(self.action_set)
            env.step()
            if env.get_time() % self.slot == self.slot - 1:
                next_state = np.array([env.get_global_state(self.agent_list)]).reshape(1, self.args.agent_num, -1)
                for i in range(len(self.agent_list)):
                    _, rewards[i] = env.get_reward(self.agent_list[i])
                done = env.get_time() >= 3599
                if self.args.episode > 1:
                    c_model.rl_model.write(self.global_state, a=self.a, a_logprob=self.a_logprob, r=rewards,
                                           s_=next_state, dw=False, done=done)
                    c_model.rl_model.learn()
                self.global_state = next_state
                self.a, self.a_logprob = c_model.rl_model.choose_action(self.global_state)  # next action

        else:  # DTDE
            for i in range(len(self.agent_list)):
                if self.agent_type == "QMIX":
                    reward, self.action_set[i], q_v = self.agent_list[i].step(self.action_set[i], self.agent_list,
                                                                              q_values)
                    q_values.append(q_v)
                else:
                    reward, self.action_set[i] = self.agent_list[i].step(self.action_set[i], self.agent_list)
                rewards += reward
                duration, phase_index, phase_type = self.action_set[i]
                env.set_phase(self.agent_list[i], 2 * phase_index if phase_type == 'G' else 2 * phase_index + 1)
                # if self.agent_list[i].tl_id == "node5":
                #     print(self.action_set[i], 2 * phase_index if phase_type == 'G' else 2 * phase_index + 1)
            env.step()
        return rewards / (1 if self.args.agent_type in {"MAT", "MAPPO", "SAC", "CCGN"} else len(self.agent_list)), False

    def get_state(self):
        return self.state

    def get_phase(self):
        return self.action

    def get_agent_list(self):
        return self.agent_list

    def clear_buf(self):
        c_model = self.agent_list[0]
        c_model.rl_model.trainer.clear()

    def get_num(self):
        return self.agent_num

    def set_state(self, state):
        self.state = state
        return

    def set_phase(self, phase):
        self.action = phase
        return

    def load_model(self, name):
        if self.agent_type == "MAT":
            self.agent_list[0].rl_model.load_model(name)
            return
        if self.agent_type in {"MA2C"}:
            self.agent_list[0].rl_model.trainer.load_model(name)
            return
        # S/L global critic
        if self.agent_type == "QMIX":
            self.rl_model.load_model(name)
        for agent in self.agent_list:
            if isinstance(agent, tl.TLight):
                agent.rl_model.load_model(name)
        print(f'model loaded from {name}')

    def load_buffer(self, name):
        if self.args.agent_type == "MAT":
            self.agent_list[0].rl_model.load_buffer(name)
        else:
            for agent in self.agent_list:
                if isinstance(agent, tl.TLight):
                    agent.rl_model.load_buffer(name)

    def save_model(self, name):
        if self.agent_type == "MAT":
            self.agent_list[0].rl_model.save_model(name)
            return
        if self.agent_type in {"MA2C"}:
            self.agent_list[0].rl_model.trainer.save_model(name)
            return
        if self.agent_type == "QMIX":
            self.rl_model.save_model(name)
        for agent in self.agent_list:
            if isinstance(agent, tl.TLight):
                agent.rl_model.save_model(name)

    def save_buffer(self, name):
        if self.args.agent_type == "MAT":
            self.agent_list[0].rl_model.save_buffer(name)
        else:
            for agent in self.agent_list:
                if isinstance(agent, tl.TLight):
                    agent.rl_model.save_buffer(name)


# initialize Lights(L_1, L_2, ..., L_n)
def init(args, execute=False, path=None, seed=25):
    # start sumo-gui or sumo
    env.start(execute, path, str(seed))

    # get all agent in environment
    agent_list = create_agent_list(args, execution=execute)

    # neighbor fully observed, sort according to distance
    # for i in range(len(agent_list)):
    #     for j in range(len(agent_list)):
    #         if i != j:
    #             i_ = agent_list[i]
    #             j_ = agent_list[j]
    #             agent_list[i].add_neighbor(env.get_distance(i_, j_), j_)
    #     agent_list[i].neighbors.sort(key=lambda neighbor: neighbor[0])
    #     t = agent_list[i].neighbors[0][0]
    #     for a in agent_list[i].neighbors:
    #         a[0] = config.COP["DISCOUNT"] * t / a[0]

    # neighbor partial observed
    # for i in range(len(agent_list)):
    #     for j in range(len(agent_list)):
    #         if i != j and config.COP["ADJACENT"][i][j]:
    #             j_ = agent_list[j]
    #             agent_list[i].add_neighbor(j_)
    # create lights based on agent
    action_set = list()
    for i in range(len(agent_list)):
        action_set.append((config.INTERSECTION["GREEN"], 0, 'G'))  # Y or G, phase index, duration
    return Lights(agent_list, args, action_set, execute)


# initialize each light in agent list
def create_agent_list(args, execution=False):
    agent_list = list()
    tl_list = env.get_tl_list()
    for i in range(len(tl_list)):
        if i >= 0:
            agent_list.append(tl.TLight(tl_list[i], env.create_lane_to_det(), None,
                                        env.get_downstream(tl_list[i], args.reward, args.map), None,
                                        [], args, execution=execution))
        else:
            agent_list.append(tl.Light(tl_list[i], env.create_lane_to_det(), env.get_lane_map(tl_list[i]),
                                       env.get_downstream(tl_list[i], args.reward), env.get_upstream(tl_list[i]),
                                       [], args))
    # for agent in agent_list:
    #     agent._light = env.get_far_agent(agent.tl_id, agent_list)
    return agent_list
