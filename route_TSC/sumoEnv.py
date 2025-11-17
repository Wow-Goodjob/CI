'''
Created on Jun 4, 2020

@author: Javier Arroyo

'''

import matplotlib.pyplot as plt
import random
import gymnasium as gym
import requests
import numpy as np
import pandas as pd
import inspect
import json
import os

from collections import OrderedDict
from scipy import interpolate
from pprint import pformat
from gymnasium import spaces
import traci as tc
from sumolib import checkBinary


class SumoGymEnv(gym.Env):

    def __init__(self, rate=0.3, city='hangzhou'):

        super(SumoGymEnv, self).__init__()

        # Define gym observation space
        self.observation_space = spaces.Box(low=np.array(self.lower_obs_bounds),
                                            high=np.array(self.upper_obs_bounds),
                                            dtype=np.float32)

        # Define gym action space
        self.action_space = spaces.Box(low=np.array(self.lower_act_bounds),
                                       high=np.array(self.upper_act_bounds),
                                       dtype=np.float32)

        self.city = city
        self.cav = set([str(i) if random.random() < rate else str(0) for i in range(1, 6000)])
        self.cav.add("9")
        self.left_cav_list = set()
        self.reset()

    def reset(self, seed=None, options=None):
        # restart SUMO
        binary = checkBinary("sumo-gui")
        tc.start([binary, "-c", f"./res/{self.city}/train.sumocfg",
                  "--seed", seed,
                  "--tripinfo-output", "./tripinfos.xml",
                  "--duration-log.statistics"])

    def step(self, cav_list, action):

        for cav, a in zip(cav_list, action):
            cav.step(a)

        for i in range(3):
            tc.simulation.step()

        for cav in cav_list:
            pass

        # Compute reward of this (state-action-state') tuple
        reward = self.get_reward()
        self.episode_rewards.append(reward)

        # Define whether a terminal state (as defined under the MDP of the task) is reached
        terminated = False

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        # Get observations at the end of this time step
        observations = self.get_observations()

        return observations, reward, terminated, info

    def close(self):
        tc.close()

    def get_reward(self):
        reward = -1
        return reward

    def get_observations(self, res):
        '''
        Get the observations, i.e. the conjunction of measurements,
        regressive and predictive variables if any. Also transforms
        the output to have the right format.

        Parameters
        ----------
        res: dictionary
            Dictionary mapping simulation variables and their value at the
            end of the last time step.

        Returns
        -------
        observations: numpy array
            Reformatted observations that include measurements and
            predictions (if any) at the end of last step.

        '''

        # Initialize observations
        observations = []

        # First check for time
        if 'time' in self.observations:
            # Time is always the first feature in observations
            observations.append(res['time'] % self.upper_obs_bounds[0])

        # Get measurements at the end of the simulation step
        for obs in self.measurement_vars:
            observations.append(res[obs])

        # Get regressions if this is a regressive agent
        if self.is_regressive:
            regr_index = res['time'] - self.step_period * np.arange(1, self.regr_n + 1)
            for var in self.regressive_vars:
                res_var = requests.put('{0}/results'.format(self.url),
                                       json={'point_names': [var],
                                             'start_time': int(regr_index[-1]),
                                             'final_time': int(regr_index[0])}).json()['payload']
                # fill_value='extrapolate' is needed for the very few cases when
                # res_var['time'] is not returned to be exactly between
                # regr_index[-1] and regr_index[0] but shorter. In these cases
                # we extrapolate linearly to reach the desired value at the extreme
                # of the regression period.
                f = interpolate.interp1d(res_var['time'],
                                         res_var[var], kind='linear', fill_value='extrapolate')
                res_var_reindexed = f(regr_index)
                observations.extend(list(res_var_reindexed))

        # Get predictions if this is a predictive agent.
        if self.is_predictive:
            predictions = requests.put('{0}/forecast'.format(self.url),
                                       json={'point_names': self.predictive_vars,
                                             'horizon': int(self.predictive_period),
                                             'interval': int(self.step_period)}).json()['payload']
            for var in self.predictive_vars:
                for i in range(self.pred_n):
                    observations.append(predictions[var][i])

        # Reformat observations
        observations = np.array(observations).astype(np.float32)

        return observations


if __name__ == "__main__":
    # Instantiate the env
    env = SumoGymEnv()

    env.reset()
    print('Observation space: {}'.format(env.observation_space))
    print('Action space: {}'.format(env.action_space))

