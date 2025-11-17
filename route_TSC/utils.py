import copy

import pandas as pd
import numpy as np
import random
import queue
import os
import csv
from scipy.interpolate import make_interp_spline
from scipy.stats import zscore, gaussian_kde
from matplotlib.patches import Rectangle
import config
from config import direction_map as d_map
from matplotlib import pyplot as plt
import xml.dom.minidom
from mpl_toolkits.mplot3d import Axes3D
import tripinfo
from matplotlib.font_manager import FontProperties
import xml.etree.ElementTree as ET
from matplotlib import cm
import matplotlib.colors as mcolors
from matplotlib import font_manager
from matplotlib.cm import get_cmap
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap

import seaborn as sns
# sns.set(style="darkgrid")
# sns.set(style="whitegrid")
# sns.set_style("white")
sns.set(style="whitegrid", font_scale=2)
import matplotlib.collections as clt
import ptitprince as pt


class CsvInterpreter:
    def __init__(self, name):
        self.name = name
        data = pd.read_csv("./data/" + name,
                           dtype={"SSID": "str", "BEGINTIME": "str", "ENDTIME": "str"})
        self.data = data
        self.flow = np.zeros((287, 9, 13))
        self.total_flow = np.zeros(287)
        self.od_pair = np.zeros((9, 9))
        self.general_od_pair = np.zeros((9, 9))
        self.out_od_pair = np.zeros((9, 12))
        self.in_od_pair = np.zeros((12, 9))
        self.full_od_pair = np.zeros((12, 12))
        self.start = -1
        self.end = -1

    def flow_xuancheng(self):
        for d in self.data.iterrows():
            series, junction, direction, num = self.process(d[1])
            self.flow[series][junction][direction] = num
            self.total_flow[series] = self.total_flow[series] + num

    def generator_od_pair(self, start, end):
        self.start = start
        self.end = end
        for d in self.data.iterrows():
            series, junction, direction, num = self.process(d[1])
            # print(junction, direction, d_map[junction])
            destination = d_map[junction][direction + 1]
            destination = destination.split()
            if start <= series <= end:
                d2 = -1
                d1 = int(destination[0])
                if len(destination) > 1:
                    d2 = int(destination[1])
                if 1 <= d1 <= 9:
                    self.od_pair[junction][d1 - 1] += num
                if 1 <= d2 <= 9:
                    self.od_pair[junction][d2 - 1] += num
                if 11 <= d1 <= 22:
                    if direction in config.out_flag[junction]:
                        self.in_od_pair[d1 - 11][junction] += num
                    else:
                        self.out_od_pair[junction][d1 - 11] += num
                if 11 <= d2 <= 22:
                    if direction in config.out_flag[junction]:
                        self.in_od_pair[d2 - 11][junction] += num
                    else:
                        self.out_od_pair[junction][d2 - 11] += num

    def calculate_od_pair(self):
        for i in range(9):
            cur = 0
            _cur = 0
            _cur2 = 0
            for j in range(9):
                cur += self.od_pair[i][j]
            for j in range(9):
                self.od_pair[i][j] = self.od_pair[i][j] * 1.0 / cur * 100
            for j in range(12):
                _cur += self.out_od_pair[i][j]
                _cur2 += self.in_od_pair[j][i]
            if _cur == 0:
                continue
            for j in range(12):
                self.out_od_pair[i][j] = self.out_od_pair[i][j] * 1.0 / _cur * 100
                self.in_od_pair[j][i] = self.in_od_pair[j][i] * 1.0 / _cur2 * 100
        # self.dfs()

    def dfs(self):
        for i in range(9):
            for j in range(9):
                self.od_pair[i][j] = self.od_pair[i][j] / 100
            for j in range(12):
                self.out_od_pair[i][j] = self.out_od_pair[i][j] / 100
                self.in_od_pair[j][i] = self.in_od_pair[j][i] / 100
        # print(self.od_pair, self.out_od_pair)
        for i in range(9):
            self.bfs(i)

        out_map = [0, 1, 2, 2, 5, 8, 8, 7, 6, 6, 3, 0]
        for i in range(12):
            for j in range(12):
                if i == j:
                    continue
                for k in range(9):
                    _i = out_map[i]
                    _j = out_map[j]
                    # if _i == _j:
                    #     self.full_od_pair[i][j] += \
                    #         self.out_od_pair[_i][i] * 1 * self.out_od_pair[_j][j]
                        # arm(i, j) = in(i) * od(i, i) * out(j)
                    if _i != k and _j != k:
                        self.full_od_pair[i][j] += \
                            self.in_od_pair[i][_i] * self.general_od_pair[_i][k] * self.general_od_pair[k][_j] * self.out_od_pair[_j][j]

                    # if k == 0:
                    #     self.full_od_pair[i][j] += \
                    #         self.in_od_pair[i][_i] * self.general_od_pair[_i][_j] * self.out_od_pair[_j][j]
                    # arm(i, j) = in(i) * od(i, k) * out(j)

    def bfs(self, idx):
        q = queue.Queue()
        q.put(idx)
        vis = {idx}
        mp = [[1, 3], [0, 2, 4], [1, 5], [0, 4, 6], [1, 3, 5, 7], [2, 4, 8], [3, 7], [4, 6, 8], [5, 7]]
        while not q.empty():
            t = q.get()
            t_mp = mp[t]
            for k in t_mp:
                if idx == k:
                    continue
                if idx == t or t == k:
                    self.general_od_pair[idx][k] = self.od_pair[idx][k]
                if idx != t and t != k:
                    self.general_od_pair[idx][k] = \
                        self.general_od_pair[idx][k] + self.general_od_pair[idx][t] * self.od_pair[t][k]
                if k not in vis:
                    q.put(k)
                    vis.add(k)

    def process(self, info):
        dic = {"HK-101": 1, "HK-104": 2, "HK-103": 3,
               "HK-96": 4, "HK-95": 5, "HK-94": 6,
               "HK-91": 7, "HK-84": 8, "HK-92": 9}
        time = info["BEGINTIME"].split()
        series = -1
        if len(time) > 1:
            series = self.date2time(time[1])

        series = series // 5 - 1
        # print(time, series)
        junction = dic[info["SSID"]] - 1
        direction = info["CDBH"] - 1
        num = info["FLOW"]

        return series, junction, direction, num

    @staticmethod
    def date2time(date):
        time = date.split(':')
        return int(time[0]) * 60 + int(time[1])


class Visualization:
    def __init__(self):
        self.id = config.SIMULATION["ID"]
        self.font_name = 'Times New Roman',
        self.label_font = {
            'family': 'Times New Roman',
            'size': 18
        }
        self.legend_font = {
            'family': 'Times New Roman',
            'size': 18
        }
        self.simc_label_font = {
            'family': 'SimSun',
            'size': 16
        }
        self.color = ['lightcoral', 'lightsalmon', 'royalblue', 'mediumpurple']
        self.label_size = 18
        self.w = 5.874  # figure width, 20cm
        self.h = 7.874  # figure height, 20cm = 7.874inch

    def csv_av(self, aw, file, name="awt", diy_path=None, column="Average Travel Time"):
        _id = self.id
        df = {
            column: list(aw)
        }
        # print(df)
        data = pd.DataFrame(df)
        csv_file = "./simudata/" + file + "/" + name + ".csv"
        if os.path.exists(csv_file) and len(column) < 10:
            data = pd.read_csv(csv_file)
            data[column] = list(aw)
            data.to_csv(csv_file, index=False)
        else:
            if diy_path is None:
                data.to_csv(csv_file, index=False, sep=',')
            else:
                data.to_csv(diy_path, index=False, sep=',')

    def csv_avs(self, aw, file, name="awt", diy_path=None):
        _id = self.id
        cnt = len(aw)
        df = {}
        for i in range(cnt):
            df[str(i)] = aw[i]
        # print(df)
        data = pd.DataFrame(df)
        if diy_path is None:
            data.to_csv("./simudata/" + file + "/" + name + ".csv", index=False, sep=',')
        else:
            data.to_csv(diy_path,  index=False, sep=',')

    def png_loss(self, agents):
        _id = self.id
        agent_list = agents.get_agent_list()
        for i in range(len(agent_list)):
            plt.plot([j for j in range(len(agent_list[i].loss))],
                     agent_list[i].loss, label=("Agent %d" % (i + 1)))
        plt.title("MSE Loss")
        plt.xlabel('Time')
        plt.ylabel('Loss')
        plt.legend(loc="low right")
        plt.show()

    def png_step_reward(self, agents):
        ax = self.create_png()
        agent_list = agents.get_agent_list()
        for agent in agent_list:
            ax.plot([j for j in range(len(agent.reward))], agent.reward, label=agent.tl_id)

        ax.set_xlabel('Time', fontdict=self.label_font)
        ax.set_ylabel('Reward', fontdict=self.label_font)
        ax.legend(prop=self.legend_font)

        plt.show()

    def load_reward(self, agents, name):
        path = "./simudata/" + name + "/reward.csv"
        _id = self.id
        assert os.path.isfile(path)
        data = pd.read_csv(path)
        agent_list = agents.get_agent_list()
        for i in range(len(agent_list)):
            t = "Agent %d" % (i + 1)
            agent_list[i].reward = data[t].values.tolist()

    def csv_reward(self, agents, name):
        _id = self.id
        agent_list = agents.get_agent_list()
        df = {}
        for i in range(len(agent_list)):
            t = "Agent %d" % (i + 1)
            df[t] = pd.Series(agent_list[i].reward)
        data = pd.DataFrame(df)
        data.to_csv("./simudata/" + name + "/reward.csv", index=False, sep=',')

    def csv_queue(self, queue, name):
        df = {}
        t = "queue"
        df[t] = pd.Series(queue)
        data = pd.DataFrame(df)
        data.to_csv("./simudata/" + name + "/queues.csv", index=False, sep=',')

    def csv_loss(self, agents, name):
        _id = self.id
        agent_list = agents.get_agent_list()
        for i in range(len(agent_list)):
            df = {} if i == 0 else pd.read_csv("./simudata/" + name + "/loss.csv")
            t = "Agent %d" % (i + 1)
            df[t] = agent_list[i].loss
            df[t] = pd.Series(agent_list[i].loss)
            data = pd.DataFrame(df)
            data.to_csv("./simudata/" + name + "/loss.csv", index=False, sep=',')

    def png_tl_log(self, agent):
        ax = self.create_png()
        ax.plot([i for i in range(len(agent.tl_log))], agent.tl_log, label=agent.tl_id)
        ax.legend(prop=self.legend_font)
        plt.show()

    def png_2flow(self, xlabel, ylabel):
        color_dict = [['royalblue', 'darkblue', 'blue'],
                      ['orange', 'orangered', 'darkorange'],
                      ['palegreen', 'lightcoral', 'khaki']]
        flow_dict = ['低流量', '高流量', '中等流量']
        data = pd.read_csv('./data/plotflow.csv')
        ax = self.create_png(xlabel, ylabel, h=15, w=8)
        rects = []
        lines = []
        for i in range(3):
            rect = Rectangle((i*60-1, -10), 60, 5500, color=color_dict[2][i], alpha=0.25, label=flow_dict[i])
            rects.append(rect)
            ax.add_patch(rect)
        for i in range(3):
            l = max(0, i*60-1)
            r = l+60
            x = data['0'][l: r]
            y = data['1'][l: r]
            z = data['2'][l: r]
            if i == 2:
                line1,  = ax.plot(x, y, color=color_dict[0][i], linewidth=5, label='真实交通流')
                line2,  = ax.plot(x, z, color=color_dict[1][i], linestyle='--', linewidth=5, label='测试交通流')
                lines = [line1, line2]
            else:
                ax.plot(x, y, color=color_dict[0][i], linewidth=5)
                ax.plot(x, z, color=color_dict[1][i], linestyle='--', linewidth=5)
        x = data['0']
        ax.set_xticks([x[i] for i in range(len(x)) if (i % 10 == 0)])
        ax.set_xticklabels([x[i] for i in range(len(x)) if (i % 10 == 0)],
                           ha='center', fontdict={
                                'family': 'Times New Roman',
                                'size': 20
                            }, rotation=45)
        ax.tick_params(axis='y', labelsize=20)
        ax.set_ylabel(ylabel, fontdict={
            'family': 'SimSun',
            'size': 22
        })
        ax.set_xlabel(xlabel, fontdict={
            'family': 'SimSun',
            'size': 22
        })
        legend1 = ax.legend(handles=lines, loc='lower right', prop={
            'family': 'SimSun',
            'size': 22
        })
        legend2 = ax.legend(handles=rects, bbox_to_anchor=(1, 0.3), loc='center right', prop={
            'family': 'SimSun',
            'size': 22
        })
        ax.add_artist(legend1)
        ax.add_artist(legend2)
        plt.show()

    def png_smooth_line(self, xlabel, ylabel, x, y):
        ax = self.create_png(xlabel, ylabel)
        ox = np.linspace(0, 1, len(x))
        poly_func = np.poly1d(np.polyfit(ox, y, 10))
        ax.plot(x, [poly_func(i) for i in ox])
        ax.set_xticks([x[i] for i in range(len(x)) if (i % 2 == 0)])
        ax.set_xticklabels([x[i] for i in range(len(x)) if (i % 2 == 0)], ha='center', fontdict=self.label_font, rotation=45)
        ax.set_ylabel(ylabel, fontdict=self.simc_label_font)
        ax.set_xlabel(xlabel, fontdict=self.simc_label_font)
        plt.show()

    def png_bar(self, xlabel, ylabel, x, y, models, env, ymin=400, path=None, x_ticks=[-1+i*1.1 for i in range(1, 21, 2)]):
        # x_ticks = [-1+i for i in range(1, 5)]
        color_dict = ['midnightblue', 'royalblue', 'mediumpurple', 'lightsalmon', 'indianred', 'grey']
        ax = self.create_png(xlabel, ylabel)
        for j in range(len(models)):
            ax.bar([i+0.3*(j-len(models)/2)+0.1 for i in x_ticks], y[j], width=0.3, align='center', label=models[j], color=color_dict[j])
        ax.set_xticklabels([])
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(env, ha='center', fontdict=self.label_font)
        ax.set_xlabel(xlabel, fontdict=self.label_font)
        ax.set_ylabel(ylabel, fontdict=self.label_font)
        ymin = y.min() - 50
        plt.ylim(ymin=ymin)
        plt.legend(loc='upper right', prop=self.label_font, ncol=2)
        if path is not None:
            plt.savefig(path, bbox_inches='tight')
            plt.show()
            plt.close()

    def scatter(self, xlabel, ylabel, x, y, models, env, ymin=400, path=None):
        num_models = 3
        num_groups = 10
        means = np.mean(y, axis=1)
        std_devs = np.std(y, axis=1)

        # 配置颜色和标签
        colors = ['midnightblue', 'royalblue', 'mediumpurple', 'lightsalmon', 'indianred', 'grey']
        labels = models

        # 创建图形
        fig, ax = plt.subplots(figsize=(7.874, 5.874))

        # 绘制每个模型的散点图和误差条
        for model_idx, (model_means, model_stds, color, label) in enumerate(zip(means, std_devs, colors, labels)):
            ax.scatter(x, model_means, label=label, color=color, alpha=0.7)
            ax.errorbar(x, model_means, yerr=model_stds, fmt='o', color=color, capsize=5, alpha=0.7)
            ax.plot(x, model_means, color=color, linestyle='-', alpha=0.7)  # 连线

        ax.set_xticklabels([])
        ax.set_xticks(x)
        font = font_manager.FontProperties(family='Times New Roman', size=18)
        plt.yticks(fontproperties=font)
        ax.set_xticklabels(env, ha='center', fontdict=self.label_font)
        ax.set_xlabel(xlabel, fontdict=self.label_font)
        ax.set_ylabel(ylabel, fontdict=self.label_font)
        plt.legend(loc='best', prop=self.label_font)
        plt.tight_layout()
        if path is not None:
            plt.savefig(path, bbox_inches='tight')
            plt.show()
            plt.close()

    def png_multiple_line(self, xlabel, ylabel, x, y, models, env, ymin=400, path=None):
        color_dict = ['royalblue', 'mediumpurple', 'lightsalmon', 'lightcoral', 'forestgreen']
        ax = self.create_png(xlabel, ylabel)
        for j in range(len(models)):
            ax.plot(x, y[j], label=models[j], color=color_dict[j], linewidth=2)
        ax.set_xticklabels([])
        ax.set_xticks(x)
        ax.set_xticklabels(env, ha='center', fontdict=self.label_font)
        ax.set_xlabel(xlabel, fontdict=self.label_font)
        ax.set_ylabel(ylabel, fontdict=self.label_font)
        plt.legend(loc='best', prop=self.label_font)
        if path is not None:
            plt.savefig(path)
            plt.show()
            plt.close()

    def png_twin_bar(self, xlabel, ylabel, x, y, models, bar1, bar2):
        color_dict = ['lightcoral', 'royalblue', 'mediumpurple', 'lightsalmon', 'forestgreen']
        ax = self.create_png(xlabel, ylabel)
        for k in range(len(x)):
            for j in range(len(models)):
                if k == 0:
                    ax.bar([i+0.4*(j-len(models)/2) for i in x[k]], y[j][k], width=0.4,
                           align='center', label=models[j], color=color_dict[j])
                else:
                    ax.bar([i + 0.4 * (j - len(models) / 2) for i in x[k]], y[j][k], width=0.4, align='center',
                           color=color_dict[j])
        ax.set_xticklabels([])
        ax.set_xticks(x[0]+x[1]+x[2])
        ax.set_xticklabels(bar1, ha='center', fontdict=self.simc_label_font, rotation=45)
        ax.set_xlabel('\n'.join(bar2), fontdict=self.simc_label_font)
        plt.legend(prop=self.legend_font, loc='upper left')
        plt.show()

    def png_bar_3d(self, xlabel, ylabel, zlabel, data):
        color_dict = ['red', 'blue', 'purple', 'orange', 'sienna']
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # 生成数据
        xpos = [0.1, 0.22, 0.34, 0.5, 0.62, 0.74, 0.9, 1.02, 1.14]
        ypos = [0.5, 0.56]
        zpos = np.zeros(18)
        dx = 0.1
        dy = 0.05
        dz = data
        xx, yy = np.meshgrid(xpos, ypos)  # 网格化坐标
        xpos, ypos = xx.ravel(), yy.ravel()
        # 绘制三维柱状图
        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=['dodgerblue']*9+['slateblue']*9)

        # 设置坐标轴名称
        # ax.set_xlabel(xlabel, fontdict={'family': 'SimSun', 'size': 8})
        # ax.set_ylabel(ylabel, fontdict={'family': 'Times New Roman', 'size': 8})
        # ax.set_zlabel(zlabel, fontdict={'family': 'SimSun', 'size': 8})
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.set_xlabel('X')
        # 调整视角
        ax.view_init(elev=10, azim=135)
        # 显示图形
        plt.show()

    def png_hist(self, xlabel, ylabel, data, models, thr):
        # 用 numpy.histogram 函数计算频率和区间的范围
        color_dict = ['lightcoral', 'royalblue', 'mediumpurple', 'lightsalmon', 'forestgreen']
        marker_dict = ['o', 'v']
        width = 1.0
        ax = self.create_png(xlabel, ylabel)
        for i, d in enumerate(data):
            bins = [1.0*i for i in range(0, thr)]
            bins_ = [1.0*i+0.5 for i in range(0, thr)]
            freq, _ = np.histogram(d['Average Travel Time'].values / 8, bins=bins)
            # 用 Matplotlib 的 bar 函数画出直方图
            ax.bar(bins[:-1], freq * 1.0 / len(d['Average Travel Time'].values),
                   width=width, align='edge', color=color_dict[i], edgecolor='grey', alpha=0.35)
            x = np.array(bins_[:-1])
            y = freq * 1.0 / len(d['Average Travel Time'].values / 8)
            new_x = np.linspace(x.min(), x.max(), 50)
            smooth_y = make_interp_spline(x, y)(new_x)
            plt.plot(new_x, np.maximum(smooth_y, 0),
                     color=color_dict[i], linewidth=2, linestyle='solid', label=models[i], marker=marker_dict[i])

        # 设定图形的轴标签和标题
        plt.xlabel(xlabel, fontdict=self.simc_label_font)
        plt.ylabel(ylabel, fontdict=self.simc_label_font)
        # plt.title('Frequency Distribution Histogram')
        plt.legend(prop=self.legend_font, loc='upper right')
        # 显示图形
        plt.show()

    def junction_queue(self, xlabel, ylabel, data):
        models = ['MaxPressure', 'IQL', 'PressLight', 'CoLight', 'GAT-GRU']
        colors = ['lightcoral', 'royalblue', 'mediumpurple', 'lightsalmon', 'sienna']
        labels = ['UL', 'UM', 'UR', 'ML', 'MM', 'MR', 'BL', 'BM', 'BR']
        ax = self.create_png(xlabel, ylabel)
        bar_width = 1.0
        x = np.arange(len(labels)) * 7
        for i, d in data.iterrows():
            if i == 0:
                continue
            ax.bar(x + i * bar_width, np.array(d.values, dtype=float), color=colors[(i-1) % len(colors)],
                   width=bar_width, label=models[i-1])
        ax.set_xticklabels([])
        ax.set_xticks([bar_width*3 + 7*i for i in range(0, 10)])
        ax.set_xticklabels(labels, fontdict=self.label_font)
        plt.ylabel(ylabel, fontdict=self.simc_label_font)
        plt.legend(prop=self.legend_font, loc='upper right')

        plt.show()

    def png_flow(self, xlabel, ylabel, q_list, var_list=None):
        models = ['MaxPressure', 'IQL', 'PressLight', 'CoLight', 'GAT-GRU']
        color_dict = ['lightcoral', 'royalblue', 'mediumpurple', 'darkorange', 'sienna']
        line_dict = ['>', 'd', 'o', 'v', 's']
        for q, v in zip(q_list, var_list):
            ax = self.create_png(xlabel, ylabel, h=11.25, w=6)
            for m in range(len(models)):
                data = q[models[m]]
                var = v[models[m]]

                ax.plot([i for i in range(len(data[:10800]))], data[:10800].rolling(500).mean() / 8,
                        label=models[m], color=color_dict[m], linewidth=(2.5 if models[m] == 'GAT-GRU' else 2.0),
                        marker=line_dict[m], markersize=6.0, markevery=500
                        )
                ax.fill_between([i for i in range(len(data[:10800]))], data[:10800].rolling(500).mean() / 8 + var[:10800].rolling(500).mean() / 8,
                                data[:10800].rolling(500).mean() / 8 - var[:10800].rolling(500).mean() / 8, alpha=0.25, color=color_dict[m])
                ax.set_xlabel(xlabel, fontdict=self.simc_label_font)
                ax.set_ylabel(ylabel, fontdict=self.simc_label_font)
            plt.legend(prop=self.legend_font, loc='upper left')
            plt.show()

    def create_png(self, xlabel='Time(s)', ylabel='Phase', h=-1, w=-1):
        if h < 0 or w < 0:
            fig, ax = plt.subplots(figsize=(self.h, self.w))
        else:
            fig, ax = plt.subplots(figsize=(h, w))
        # x-y label type
        ax.tick_params(labelsize=self.label_size)
        ax.set_xlabel(xlabel, fontdict=self.simc_label_font)
        ax.set_ylabel(ylabel, fontdict=self.label_font)
        self.modify_ax(ax)

        return ax

    def modify_ax(self, ax):
        x_label = ax.get_xticklabels()
        y_label = ax.get_yticklabels()
        [t.set_fontname(self.font_name) for t in x_label]
        [t.set_fontname(self.font_name) for t in y_label]

    @staticmethod
    def mat_hot():
        # plot OD-map
        font = FontProperties(family='Times New Roman', weight='bold', size=12)
        sim_font = FontProperties(family='SimSun', weight='bold', size=12)
        flow = CsvInterpreter("xuancheng.csv")
        flow.flow_xuancheng()
        flow.generator_od_pair(77, 113)  # 77-113
        flow.calculate_od_pair()
        flow.dfs()
        # print(flow.full_od_pair)
        flow.full_od_pair = np.array(flow.full_od_pair)
        flow.full_od_pair = flow.full_od_pair / np.sum(flow.full_od_pair)
        flow.full_od_pair = flow.full_od_pair * 5000
        fig, ax = plt.subplots()
        # print(flow.general_od_pair)
        im = ax.imshow(flow.full_od_pair, cmap='Blues', extent=[-0.5, 11.5, -0.5, 11.5], vmin=0, vmax=175)
        en_font = {'family': 'Times New Roman', 'size': 14}
        cn_font = {'family': "SimSun", 'size': 14}
        # 设置x和y轴刻度位置和标签
        ax.set_xticks(np.arange(0, 12, 1))
        ax.set_yticks(np.arange(0, 12, 1))
        l = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
        ax.set_xticklabels(l, fontdict=en_font)
        ax.set_yticklabels(l[::-1], fontdict=en_font)

        # 添加颜色条
        cbar = fig.colorbar(im)
        # print(cbar.ax.get_yticklabels())
        cbar.ax.set_yticklabels(cbar.get_ticks().astype(int), fontname='Times New Roman', fontsize=14)
        # cbar.ax.set_ticklabels(cbar.ax.get_ticklabels(), fontname='Times New Roman', fontsize=14)
        cbar.set_label('到达率(辆/小时)', fontdict=cn_font)
        for i in range(12):
            for j in range(12):
                ax.text(i, 11-j, str(int(flow.full_od_pair[j][i])), ha='center', va='center', fontdict=en_font)
        plt.show()

    @staticmethod
    def calculate_reward(name="test"):
        data = pd.read_csv("./simudata/" + name + "/reward.csv")
        length = 7200 * 200
        for index, col in data.iteritems():
            length = min(length, len(col) - col.isna().sum())
        mean = data.mean(axis=1)
        data['Mean'] = mean
        var = data.values.std(axis=1)
        data['Var'] = var
        data['Mean'] = data['Mean'][:length]
        data['Var'] = data['Var'][:length]
        data.to_csv("./simudata/" + name + "/reward.csv", index=False, sep=',')

    @staticmethod
    def calculate_loss(name="test"):
        data = pd.read_csv("./simudata/" + name + "/loss.csv")
        length = 7200 * 200
        for index, col in data.iteritems():
            length = min(length, len(col) - col.isna().sum())
        mean = data.mean(axis=1)
        data['Mean'] = mean
        var = data.values.std(axis=1)
        data['Var'] = var
        data['Mean'] = data['Mean'][:length]
        data['Var'] = data['Var'][:length]
        data.to_csv("./simudata/" + name + "/loss.csv", index=False, sep=',')

    def png_local_reward(self, name_list=None):
        if name_list is None:
            print("local reward need specify simulation name")
        ax = self.create_png()
        for name in name_list:
            data = pd.read_csv("./simudata/" + name + "/reward.csv")
            for i in range(1, 10):
                y = data['Agent ' + str(i)].rolling(1000).mean()
                ax.plot([i for i in range(len(y))], y, label='Agent ' + str(i))
                ax.legend(prop=self.legend_font, loc='lower right')
                # plt.title("Rewards")
                ax.set_xlabel('Simulation Step', fontdict=self.label_font)
                ax.set_ylabel('Reward', fontdict=self.label_font)
            plt.savefig('./simudata/' + name + '/local_reward.png', bbox_inches='tight')
        plt.show()

    def png_total_reward(self, name_list=None):
        if name_list is None:
            name_list = {"test"}

        ax = self.create_png()
        color_dict = ['lightcoral', 'royalblue', 'mediumpurple', 'darkorange', 'sienna']
        for (i, name) in enumerate(name_list):
            data = pd.read_csv("./simudata/" + name + "/reward.csv")
            mean = data['Mean'].rolling(1555).mean()
            var = data['Var'].rolling(1555).mean()

            ax.plot([i for i in range(len(mean))], mean, label=name, color=color_dict[i])
            # ax.fill_between([i for i in range(len(mean))], mean + var, mean - var, alpha=0.4, color=color_dict[i])

            ax.legend(prop=self.legend_font, loc='lower right')
            # plt.title("Rewards")
            ax.set_xlabel('Simulation Step', fontdict=self.label_font)
            ax.set_ylabel('Reward', fontdict=self.label_font)
            if len(name_list) == 1:
                plt.savefig('./simudata/' + name + '/reward.png', bbox_inches='tight')
        plt.show()

    def png_total_loss(self, name_list=None):
        if name_list is None:
            name_list = {"test"}
        for name in name_list:
            data = pd.read_csv("./simudata/" + name + "/loss.csv")
            mean = data['Mean'].rolling(1000).mean()
            var = data['Var'].rolling(1000).mean()

            ax = self.create_png()
            ax.plot([i for i in range(len(mean))], mean, label=name.split('-')[0])
            # ax.fill_between([i for i in range(len(mean))], mean + var, mean - var, alpha=0.4)

            ax.legend(prop=self.legend_font, loc='lower right')
            # plt.title("Rewards")
            ax.set_xlabel('Simulation Step', fontdict=self.label_font)
            ax.set_ylabel('Loss', fontdict=self.label_font)
            plt.savefig('./simudata/' + name + '/loss.png', bbox_inches='tight')
            plt.show()

    @staticmethod
    def abortion_radar():
        # low
        results = [{'行程时间': 143.35, '时间损失      ': 58.32, '排队长度     ': 4.40, '     等待时间': 38.48, '   吞吐量': 2770},
                   {'行程时间': 147.43, '时间损失      ': 62.41, '排队长度     ': 4.97, '     等待时间': 42.29, '   吞吐量': 2767},
                   {'行程时间': 145.71, '时间损失      ': 60.69, '排队长度     ': 4.83, '     等待时间': 40.68, '   吞吐量': 2764}]
        # medium
        results = [{'行程时间': 147.58, '时间损失      ': 62.86, '排队长度     ': 7.44, '     等待时间': 41.46, '   吞吐量': 4260},
                   {'行程时间': 155.45, '时间损失      ': 70.75, '排队长度     ': 8.77, '     等待时间': 48.45, '   吞吐量': 4256},
                   {'行程时间': 149.91, '时间损失      ': 65.18, '排队长度     ': 7.90, '     等待时间': 43.37, '   吞吐量': 4254}]
        # high
        results = [{'行程时间': 160.51, '时间损失      ': 75.42, '排队长度      ': 10.73, '     等待时间': 50.86, '   吞吐量': 5069},
                   {'行程时间': 176.21, '时间损失      ': 91.14, '排队长度      ': 13.53, '     等待时间': 64.51, '   吞吐量': 5063},
                   {'行程时间': 163.17, '时间损失      ': 78.10, '排队长度      ': 11.25, '     等待时间': 53.20, '   吞吐量': 5076}]

        data_length = len(results[0])
        # 将极坐标根据数据长度进行等分
        angles = np.linspace(0, 2 * np.pi, data_length, endpoint=False)
        labels = [key for key in results[0].keys()]
        score = [[(v-k)/k*100 for (v, k) in zip(result.values(), results[0].values())] for result in results]
        # 使雷达图数据封闭
        score_a = np.concatenate((score[0], [score[0][0]]))
        score_b = np.concatenate((score[1], [score[1][0]]))
        score_c = np.concatenate((score[2], [score[2][0]]))
        angles = np.concatenate((angles, [angles[0]]))
        labels = np.concatenate((labels, [labels[0]]))
        # 设置图形的大小
        fig = plt.figure(figsize=(8, 6), dpi=100)
        # 新建一个子图
        ax = plt.subplot(111, polar=True)
        # 绘制雷达图
        ax.plot(angles, score_a, 'o-', color='royalblue', linewidth=3.25, markersize=7)
        ax.fill(angles, score_a, alpha=0.35, color='royalblue')
        ax.plot(angles, score_b, 's-', color='mediumpurple', linewidth=3.25, markersize=7)
        ax.fill(angles, score_b, alpha=0.35, color='mediumpurple')
        ax.plot(angles, score_c, 'd-', color='lightsalmon', linewidth=3.25, markersize=9)
        ax.fill(angles, score_c, alpha=0.35, color='lightsalmon')
        # 设置雷达图中每一项的标签显示
        ax.set_thetagrids(angles * 180 / np.pi, labels, fontname='SimSun', fontsize=22.0)
        # 设置雷达图的0度起始位置
        ax.set_theta_zero_location('N')
        # 设置雷达图的坐标刻度范围
        # ax.set_rlim(-5, 25)
        ax.set_rlim(-10, 30)
        # ax.set_ylabel('%', ha='right', va='top', fontdict={'family': 'Times New Roman', 'size': 16})
        # ax.text(np.radians(-85), 28, '%', ha='left', va='top', fontdict={'family': 'Times New Roman', 'size': 20})
        ax.text(np.radians(-85), 34, '%', ha='left', va='top', fontdict={'family': 'Times New Roman', 'size': 20})
        # 设置雷达图的坐标值显示角度，相对于起始角度的偏移量
        ax.set_rlabel_position(270)
        # ax.set_yticklabels(['-10', '0', '10', '20', '30', '40', '50'], ha='center', fontdict={'family': 'Times New Roman', 'size': 20})
        ax.set_yticklabels(['-5', '0', '5', '10', '15', '20', '25', '30'], ha='center', fontdict={'family': 'Times New Roman', 'size': 20})
        # plt.legend(["GAT-GRU", "Remove GRU", "Remove GAT"], loc='center', prop={'family': 'Times New Roman', 'size': 16})

        x_pos, y_pos = 1.2, 0.8  # x_pos表示水平方向的偏移量，y_pos表示垂直方向的偏移量
        legend = ax.legend(["GAT-GRU", "Remove GAT", "Remove GRU", ],
                           loc='lower right', prop={'family': 'Times New Roman', 'size': 18}, bbox_to_anchor=(x_pos, y_pos))
        # 将图例拖离子图
        legend.set_draggable(True)

        plt.show()

class XmlGenerator:
    def __init__(self, name):
        self.name = name
        self.address = "./res/" + name
        self.doc = xml.dom.minidom.Document()

    def generator_turn_def(self, turns):
        root = self.doc.createElement('turns')
        self.doc.appendChild(root)

        interval = self.doc.createElement('interval')
        interval.setAttribute('begin', '0')
        interval.setAttribute('end', '7200')
        root.appendChild(interval)

        for i in range(len(turns)):
            if i == 0:
                interval.appendChild(self.generator_single_flow('N1', '1', '4', turns[0][3]))
                interval.appendChild(self.generator_single_flow('N1', '1', '2', turns[0][1]))
                interval.appendChild(self.generator_single_flow('W1', '1', '4', turns[0][3]))
                interval.appendChild(self.generator_single_flow('W1', '1', '2', turns[0][1]))

                r = random.uniform(0, 100 - turns[0][3])
                interval.appendChild(self.generator_single_flow('2', '1', 'N1', r))
                interval.appendChild(self.generator_single_flow('2', '1', 'W1', 100 - turns[0][3] - r))
                interval.appendChild(self.generator_single_flow('2', '1', '4', turns[0][3]))

                r = random.uniform(0, 100 - turns[0][1])
                interval.appendChild(self.generator_single_flow('4', '1', 'N1', r))
                interval.appendChild(self.generator_single_flow('4', '1', 'W1', 100 - turns[0][1] - r))
                interval.appendChild(self.generator_single_flow('4', '1', '2', turns[0][1]))
            if i == 1:
                interval.appendChild(self.generator_single_flow('N2', '2', '1', turns[i][0]))
                interval.appendChild(self.generator_single_flow('N2', '2', '3', turns[i][2]))
                interval.appendChild(self.generator_single_flow('N2', '2', '5', turns[i][4]))

                interval.appendChild(self.generator_single_flow('3', '2', 'N2', 100 - turns[i][0] - turns[i][4]))
                interval.appendChild(self.generator_single_flow('3', '2', '1', turns[i][0]))
                interval.appendChild(self.generator_single_flow('3', '2', '5', turns[i][4]))

                interval.appendChild(self.generator_single_flow('5', '2', 'N2', 100 - turns[i][0] - turns[i][2]))
                interval.appendChild(self.generator_single_flow('5', '2', '1', turns[i][0]))
                interval.appendChild(self.generator_single_flow('5', '2', '3', turns[i][2]))

                interval.appendChild(self.generator_single_flow('1', '2', 'N2', 100 - turns[i][0] - turns[i][2]))
                interval.appendChild(self.generator_single_flow('1', '2', '3', turns[i][2]))
                interval.appendChild(self.generator_single_flow('1', '2', '5', turns[i][4]))

            if i == 2:
                interval.appendChild(self.generator_single_flow('N3', '3', '2', turns[i][1]))
                interval.appendChild(self.generator_single_flow('N3', '3', '6', turns[i][5]))
                interval.appendChild(self.generator_single_flow('E1', '3', '2', turns[i][1]))
                interval.appendChild(self.generator_single_flow('E1', '3', '6', turns[i][5]))

                r = random.uniform(0, 100 - turns[i][5])
                interval.appendChild(self.generator_single_flow('2', '3', 'N3', r))
                interval.appendChild(self.generator_single_flow('2', '3', 'E1', 100 - turns[i][5] - r))
                interval.appendChild(self.generator_single_flow('2', '3', '6', turns[i][5]))

                r = random.uniform(0, 100 - turns[i][1])
                interval.appendChild(self.generator_single_flow('6', '3', 'N3', r))
                interval.appendChild(self.generator_single_flow('6', '3', 'E1', 100 - turns[i][1] - r))
                interval.appendChild(self.generator_single_flow('6', '3', '2', turns[i][1]))

            if i == 3:
                interval.appendChild(self.generator_single_flow('W2', '4', '1', turns[i][0]))
                interval.appendChild(self.generator_single_flow('W2', '4', '5', turns[i][4]))
                interval.appendChild(self.generator_single_flow('W2', '4', '7', turns[i][6]))

                interval.appendChild(self.generator_single_flow('1', '4', 'W2', 100 - turns[i][4] - turns[i][6]))
                interval.appendChild(self.generator_single_flow('1', '4', '5', turns[i][4]))
                interval.appendChild(self.generator_single_flow('1', '4', '7', turns[i][6]))

                interval.appendChild(self.generator_single_flow('5', '4', 'W2', 100 - turns[i][0] - turns[i][6]))
                interval.appendChild(self.generator_single_flow('5', '4', '1', turns[i][0]))
                interval.appendChild(self.generator_single_flow('5', '4', '7', turns[i][6]))

                interval.appendChild(self.generator_single_flow('7', '4', 'W2', 100 - turns[i][0] - turns[i][4]))
                interval.appendChild(self.generator_single_flow('7', '4', '1', turns[i][0]))
                interval.appendChild(self.generator_single_flow('7', '4', '5', turns[i][4]))

            if i == 4:
                interval.appendChild(self.generator_single_flow('2', '5', '4', turns[i][3]))
                interval.appendChild(self.generator_single_flow('2', '5', '6', turns[i][5]))
                interval.appendChild(self.generator_single_flow('2', '5', '8', turns[i][7]))

                interval.appendChild(self.generator_single_flow('4', '5', '2', turns[i][1]))
                interval.appendChild(self.generator_single_flow('4', '5', '6', turns[i][5]))
                interval.appendChild(self.generator_single_flow('4', '5', '8', turns[i][7]))

                interval.appendChild(self.generator_single_flow('6', '5', '2', turns[i][1]))
                interval.appendChild(self.generator_single_flow('6', '5', '4', turns[i][3]))
                interval.appendChild(self.generator_single_flow('6', '5', '8', turns[i][7]))

                interval.appendChild(self.generator_single_flow('8', '5', '2', turns[i][1]))
                interval.appendChild(self.generator_single_flow('8', '5', '4', turns[i][3]))
                interval.appendChild(self.generator_single_flow('8', '5', '6', turns[i][5]))

            if i == 5:
                interval.appendChild(self.generator_single_flow('E2', '6', '5', turns[i][4]))
                interval.appendChild(self.generator_single_flow('E2', '6', '9', turns[i][8]))
                interval.appendChild(self.generator_single_flow('E2', '6', '3', turns[i][2]))

                interval.appendChild(self.generator_single_flow('3', '6', '5', turns[i][4]))
                interval.appendChild(self.generator_single_flow('3', '6', '9', turns[i][8]))
                interval.appendChild(self.generator_single_flow('3', '6', 'E2', 100 - turns[i][4] - turns[i][8]))

                interval.appendChild(self.generator_single_flow('5', '6', '3', turns[i][2]))
                interval.appendChild(self.generator_single_flow('5', '6', '9', turns[i][8]))
                interval.appendChild(self.generator_single_flow('5', '6', 'E2', 100 - turns[i][2] - turns[i][8]))

                interval.appendChild(self.generator_single_flow('9', '6', '3', turns[i][2]))
                interval.appendChild(self.generator_single_flow('9', '6', '5', turns[i][4]))
                interval.appendChild(self.generator_single_flow('9', '6', 'E2', 100 - turns[i][2] - turns[i][4]))

            if i == 6:
                interval.appendChild(self.generator_single_flow('W3', '7', '4', turns[i][3]))
                interval.appendChild(self.generator_single_flow('W3', '7', '8', turns[i][7]))
                interval.appendChild(self.generator_single_flow('S1', '7', '4', turns[i][3]))
                interval.appendChild(self.generator_single_flow('S1', '7', '8', turns[i][7]))

                r = random.uniform(0, 100 - turns[i][7])
                interval.appendChild(self.generator_single_flow('4', '7', 'W3', r))
                interval.appendChild(self.generator_single_flow('4', '7', '8', turns[i][7]))
                interval.appendChild(self.generator_single_flow('4', '7', 'S1', 100 - r - turns[i][7]))

                r = random.uniform(0, 100 - turns[i][3])
                interval.appendChild(self.generator_single_flow('8', '7', 'W3', r))
                interval.appendChild(self.generator_single_flow('8', '7', '4', turns[i][3]))
                interval.appendChild(self.generator_single_flow('8', '7', 'S1', 100 - r - turns[i][3]))

            if i == 7:
                interval.appendChild(self.generator_single_flow('S2', '8', '5', turns[i][4]))
                interval.appendChild(self.generator_single_flow('S2', '8', '7', turns[i][6]))
                interval.appendChild(self.generator_single_flow('S2', '8', '9', turns[i][8]))

                interval.appendChild(self.generator_single_flow('5', '8', 'S2', 100 - turns[i][8] - turns[i][6]))
                interval.appendChild(self.generator_single_flow('5', '8', '7', turns[i][6]))
                interval.appendChild(self.generator_single_flow('5', '8', '9', turns[i][8]))

                interval.appendChild(self.generator_single_flow('7', '8', 'S2', 100 - turns[i][8] - turns[i][4]))
                interval.appendChild(self.generator_single_flow('7', '8', '5', turns[i][4]))
                interval.appendChild(self.generator_single_flow('7', '8', '9', turns[i][8]))

                interval.appendChild(self.generator_single_flow('9', '8', 'S2', 100 - turns[i][6] - turns[i][4]))
                interval.appendChild(self.generator_single_flow('9', '8', '7', turns[i][6]))
                interval.appendChild(self.generator_single_flow('9', '8', '5', turns[i][4]))

            if i == 8:
                interval.appendChild(self.generator_single_flow('E3', '9', '6', turns[i][5]))
                interval.appendChild(self.generator_single_flow('E3', '9', '8', turns[i][7]))
                interval.appendChild(self.generator_single_flow('S3', '9', '6', turns[i][5]))
                interval.appendChild(self.generator_single_flow('S3', '9', '8', turns[i][7]))

                r = random.uniform(0, 100 - turns[i][7])
                interval.appendChild(self.generator_single_flow('6', '9', 'E3', r))
                interval.appendChild(self.generator_single_flow('6', '9', '8', turns[i][7]))
                interval.appendChild(self.generator_single_flow('6', '9', 'S3', 100 - r - turns[i][7]))

                r = random.uniform(0, 100 - turns[i][5])
                interval.appendChild(self.generator_single_flow('8', '9', 'E3', r))
                interval.appendChild(self.generator_single_flow('8', '9', '6', turns[i][5]))
                interval.appendChild(self.generator_single_flow('8', '9', 'S3', 100 - r - turns[i][5]))

        # print(self.doc.toprettyxml(indent=" "))
        fp = open(self.address, 'w')
        self.doc.writexml(fp, indent='\t', addindent='\t', newl='\n', encoding='utf-8')

    def generator_single_flow(self, f, c, t, p):
        flow = self.doc.createElement('edgeRelation')
        flow.setAttribute('from', f + '_' + c)
        flow.setAttribute('to', c + '_' + t)
        flow.setAttribute('probability', str(p))
        return flow


def gen_flow():
    os.system("cd D:\\coding\\drl_traffic_signal_control\\res\\testFlow")
    handler = "python D:\\coding\\sumo\\tools\\randomTrips.py -n ..\\3_3tl_vl.net.xml"
    subHandler = [" -b 0 -e 2160 --period ", " -b 2161 -e 4080 --period ", " -b 4081 -e 6180 --period ",
                  " -b 6181 -e 7260 --period "]

    tail = " -L -o "

    dir_path = "D:\\coding\\drl_traffic_signal_control\\res\\testFlow\\"
    full_path = dir_path + "genFlow.txt"
    file = open(full_path, 'w')
    base = [14, 9, 12, 8]
    for i in range(-4, 6):
        sumFlow = 0
        fileAgg = []
        for j in range(4):
            t = (base[j] + i) / 10
            sumFlow += t
            f = "3_3xcp" + str(j + 1) + "_" + str(t) + ".trips.xml"
            h = handler + subHandler[j] + str(t) + tail + f
            file.write(h)
            file.write('\n')
            fileAgg.append(f)
            cd = "cd D:\\coding\\drl_traffic_signal_control\\res\\testFlow"
            cmd = cd + " && " + h
            os.system(cmd)

            fp = open(dir_path + f, 'r')
            lines = fp.readlines()
            fp.close()  # 关闭文件
            fp = open(dir_path + f, 'w')
            for s in lines:
                if "id=\"" in s:
                    s = s.replace("id=\"", "id=\"" + str(j + 1) + ".")
                fp.write(s)
            fp.close()  # 关闭文件

        router = "D:\\coding\\sumo\\bin\\duarouter -n ..\\3_3tl_vl.net.xml --route-files="
        for i, f in enumerate(fileAgg):
            router += f
            if i != len(fileAgg) - 1:
                router += ","
        router += " -o 3_3xcp1_4_" + str(round(sumFlow, 2)) + ".rou.xml"
        file.write(router)
        file.write('\n')
        cd = "cd D:\\coding\\drl_traffic_signal_control\\res\\testFlow"
        cmd = cd + " && " + router
        os.system(cmd)


def gen_exe():
    dir_path = "D:\\coding\\drl_traffic_signal_control\\res\\testFlow\\"
    flow = [2.7, 3.1, 3.5, 3.9, 4.3, 4.7, 5.1, 5.5, 5.9, 6.3]
    cfg = "3_3exe.sumocfg"
    fp = open(dir_path + cfg)
    lines = fp.readlines()
    fp.close()  # 关闭文件
    for f in flow:
        fcfg = "3_3exe_" + str(f) + ".sumocfg"
        fp = open(dir_path + fcfg, 'w')
        for s in lines:
            if "<route-files value=\"3_3xcp1_4.rou.xml\"/>" in s:
                s = s.replace("3_3xcp1_4.rou.xml", "3_3xcp1_4_" + str(f) + ".rou.xml")
            fp.write(s)
        fp.close()  # 关闭文件