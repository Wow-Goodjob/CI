import os
import sys
import platform
# os.environ['SUMO_HOME'] = "/usr/local/lib/python3.9/dist-packages/sumo"
# sys.path.append(os.path.join(os.environ.get("SUMO_HOME"), 'tools'))
if platform.system().lower() == 'linux':
    os.environ['SUMO_HOME'] = "/usr/local/lib64/python3.6/site-packages/sumo"
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME"), 'tools'))
import pandas as pd
from utils import XmlGenerator, CsvInterpreter
from utils import Visualization

o = Visualization()

import numpy as np
import statistics
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as colors

import config
import episode
import env
import lights
import tripinfo
from matplotlib.font_manager import FontProperties

# # calculate duration/waiting/loss/throughput through tripinfo.xml
# a, b = tripinfo.get_segment_info()
# o.csv_av(a, file="queuehist", name="loss")
# o.csv_av(b, file="queuehist", name="duration")


names = ['MAT-FC-FC 2023-09-17-22-44-51 MAT seq act with random train', 'MAT-FC-FC 2023-09-18-11-04-51 MAT inner act with random train', 'MAT-FC-FC 2023-09-18-16-24-23 MAT outer act with random train', 'MAT-FC-FC 2023-09-17-16-00-42 MAT queue act with random train']
for i, name in enumerate(names):
    l, m, h = [], [], []
    for sd in range(40, 140, 10):
        ord = pd.read_csv(f'./simudata/{name}/ep150/testFlow1/0_{str(sd)}_queues_list_3.csv')
        l.append(np.mean(ord[str(sd)][:3600]) / 7.5)
        h.append(np.mean(ord[str(sd)][3600:7200]) / 7.5)
        m.append(np.mean(ord[str(sd)][7200:]) / 7.5)
    # df = pd.DataFrame({'Mean': [np.mean(l) / 7.5, np.mean(m) / 7.5, np.mean(h) / 7.5],
    #                    'Var': [np.var(l) / 7.5, np.var(m) / 7.5, np.var(h) / 7.5]})
    df = pd.DataFrame({'Low': l, 'Medium': m, 'Peak': h})
    df.to_csv(f'./simudata/{name}/ep150/testFlow1/01_q3.csv', index=False, sep=',')
    print(df)

# # plot OD-map
# o.mat_hot()

# # plot relative performance
# o.png_bar('模型', '行程时间/s', [1, 4, 7], [[1, 2], [2, 3], [3, 4]], ['a', 'b', 'c'])
# o.png_bar('模型', '排队长度/m', [], [])
# o.png_bar('模型', '时间损耗/s', [], [])
# o.png_bar('', 't/s', [1, 4, 7], [[1, 2, 3], [2, 3, 3], [10, 10, 10], [17, 16, 12]],
#           ['a', 'b', 'c', 'd'], ['low', 'medium', 'high'])

# # 消融试验bar图
# o.png_twin_bar('', '%', [[1, 2, 3], [5, 6, 7], [9, 10, 11]],
#                [[[2.89, 6.55, 11.40], [5.02, 11.17, 15.19], [1.31, 2.63, 2.13]],
#                 [[1.76, 4.00, 5.33], [7.43, 16.46, 17.88], [4.26, 8.57, 6.94]]],
#                ['Remove GRU', 'Remove GAT'],
#                bar1=['行程时间', '延误时间', '排队长度', '行程时间', '延误时间', '排队长度', '行程时间', '延误时间', '排队长度'],
#                bar2=['低流量             中等流量              高流量'])

# # 交通流量变化曲线
# o.png_smooth_line('时间', '车流量 (辆/小时)',
#                   y=[1189, 1177, 1030, 1200, 1012, 1255, 1131, 1489, 1873, 1801, 1821, 1880, 1911, 938, 1384,
#                      1401, 1363, 1438, 1302, 1372, 1446, 1271, 1255, 1340, 1422],
#                   x=["7:00", "7:05", "7:10", "7:15", "7:20", "7:25", "7:30", "7:35", "7:40", "7:45", "7:50", "7:55",
#                      "8:00", "8:05", "8:10", "8:15", "8:20", "8:25", "8:30", "8:35", "8:40", "8:45", "8:50", "8:55",
#                      "9:00"])
# o.png_2flow('时间', '车流量(辆/小时)')

# # 性能表现bar图
# # 高流量
# o.png_bar('', '%', [1, 4, 7],
#           [[-3.082148876, -5.553573143, -28.58121483],
#            [-9.939214191, -17.88594832, -33.44528843],
#            [-13.76269158, -24.76723025, -40.17953142],
#            [-16.05071908, -28.91763623, -43.94648536]],
#           ['IQL', 'PressLight', 'CoLight', 'GAT-GRU'], ['行程时间', '延误时间', '排队长度'])
# # 中等流量
# o.png_bar('', '%', [1, 4, 7],
#           [[6.922897251, 14.50770604, -1.281852181],
#            [-1.023539792, -2.184148898, -13.70372539],
#            [-5.613047561, -11.79175079, -31.03380678],
#            [-9.120030723, -19.06931145, -34.50486391]],
#           ['IQL', 'PressLight', 'CoLight', 'GAT-GRU'], ['行程时间', '延误时间', '排队长度'])
# # 低流量
# o.png_bar('', '%', [1, 4, 7],
#           [[2.227035171, 5.087477791, -12.52246939],
#            [-0.916246107, -2.080205813, -14.11264652],
#            [0.423390325, 0.969555369, -17.40628394],
#            [-5.541389668, -12.59902233, -28.75145115]],
#           ['IQL', 'PressLight', 'CoLight', 'GAT-GRU'], ['行程时间', '延误时间', '排队长度'])

# # 车流量折线图
# data_high = pd.read_csv('./simudata/q3.csv')
# data_median = pd.read_csv('./simudata/q3.9.csv')
# data_low = pd.read_csv('./simudata/q4.3.csv')

# queue_mean = pd.read_csv('./simudata/q3_mean.csv')
# queue_var = pd.read_csv('./simudata/q3_var.csv')
# o.png_flow('时间', '排队长度/米', [queue_mean], [queue_var])

# # Traffic Flow: Intersection i to Intersection 5
# data = pd.read_csv("./data/intersection5direction.csv")
# o = Visualization()
# ax = o.create_png()
# for i in range(1, 5):
#     y = data[str(2 * i) + '_5']
#     ax.plot([i for i in range(len(y))], y.rolling(500).mean(), label='Intersection' + str(2 * i) + ' to Intersection5')
#     ax.legend(prop=o.legend_font, loc='lower right')
#     ax.set_xlabel('Simulation Step', fontdict=o.label_font)
#     ax.set_ylabel('Number of Vehicle', fontdict=o.label_font)
# plt.savefig('./data/intersection5direction.png', bbox_inches='tight')
# plt.show()

# # Graph Attention: Intersection i to Intersection 5
# data = pd.read_csv('./simudata/DQN-GAT-FC 2023-05-31-02-20-16 l_removeGRU_lr_1e-3_segment_state/ep170/GAT-FC-atts.csv')
# o = Visualization()
# ax = o.create_png()
# for i in [2, 4, 6, 8, 5]:
#     y = data['Inter ' + str(i) + ' to Inter 5']
#     ax.plot([i for i in range(len(y))], y.rolling(50).mean(), label='Inter ' + str(i) + ' to Inter 5')
#     slc = int(len(y) / 2)
#     for j in range(0, 2):
#         print(j, i, y[slc * j: slc * (j + 1)].mean())
#     ax.legend(prop=o.legend_font, loc='lower right')
#     ax.set_xlabel('Simulation Step', fontdict=o.label_font)
#     ax.set_ylabel('Attention Score', fontdict=o.label_font)
# plt.savefig('./data/att.png', bbox_inches='tight')
# plt.show()

# # 消融实验3D bar图
'''GAT-GRU	152.58	67.53	5.44	155.24	69.98	6.32	170.04	84.64	8.94
Remove GRU	156.99	71.95	6.06	163.04	77.80	7.28	172.26	86.87	9.13
Remove GAT	155.27	70.23	5.73	166.77	81.50	7.45	177.29	91.89	9.56
'''
# a = [[152.58, 67.53, 5.44], [155.24, 69.98, 6.32], [170.04, 84.64, 8.94]]
# Remove_GRU = [[156.99, 71.95, 6.06], [163.04, 77.80, 7.28], [172.26, 86.87, 9.13]]
# Remove_GAT = [[155.27, 70.23, 5.73], [166.77, 81.50, 7.45], [177.29, 91.89, 9.56]]
# a = [152.58, 67.53, 5.44, 155.24, 69.98, 6.32, 170.04, 84.64, 8.94]
# Remove_GRU = [156.99, 71.95, 6.06, 163.04, 77.80, 7.28, 172.26, 86.87, 9.13]
# Remove_GAT = [155.27, 70.23, 5.73, 166.77, 81.50, 7.45, 177.29, 91.89, 9.56]
# a = [152.58, 67.53, 47.33, 155.24, 69.98, 48.97, 170.04, 84.64, 60.58]
# Remove_GRU = [156.99, 71.95, 51.56, 163.04, 77.80, 56.01, 172.26, 86.87, 63.38]
# Remove_GAT = [155.27, 70.23, 50.01, 166.77, 81.50, 59.57, 177.29, 91.89, 68.04]
# base = (np.divide(np.array(a), np.array(a))*100).tolist()
# rgru = (np.divide(np.array(Remove_GRU)-np.array(a), np.array(a))*100).tolist()
# rgat = (np.divide(np.array(Remove_GAT)-np.array(a), np.array(a))*100).tolist()
#
# o.png_bar_3d(['低流量', '中等流量', '高流量'], ['Remove GAT', 'Remove GRU'], '时间(秒)', rgru+rgat)

# 频率直方图
# data1 = pd.read_csv('./simudata/DQN-GAT-GRU 2023-06-01-09-43-42 l_GAT-GRU_lr_1e-3_segment_state/ep150/queues_list.csv')
# data2 = pd.read_csv('./simudata/DQN-GAT-FC 2023-06-03-05-03-06 l_CoLight_lr_1e-3_segment_state/ep150/queues_list.csv')
# o.png_hist('排队长度(米)', '频率', [data1[1800:5400], data2[1800:5400]], ["GAT-GRU", "CoLight"], 16)
# o.png_hist('排队长度(米)', '频率', [data1[5400:9000], data2[5400:9000]], ["GAT-GRU", "CoLight"], 25)
# o.png_hist('排队长度(米)', '频率', [data1[9000:12600], data2[9000:12600]], ["GAT-GRU", "CoLight"], 16)

# 消融实验雷达图
# o.abortion_radar()

# 各交叉口排队
# data = pd.read_csv('./simudata/junctionq.csv', header=None)
# o.junction_queue('交叉口', '排队长度/米', data)
