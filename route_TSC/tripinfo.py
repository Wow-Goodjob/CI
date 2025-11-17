import xml.etree.ElementTree as ET
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker


def get_average_travel_time():
    tree = ET.parse('./tripinfos.xml')
    root = tree.getroot()

    car_list = []
    total_travel_time = 0

    # 遍历XML元素
    for child in root.findall('tripinfo'):
        car_list.append(child)
        total_travel_time += float(child.attrib['duration'])

    # 计算平均等待时间
    average_travel_time = total_travel_time / len(car_list)
    return average_travel_time


def get_tripinfo_list(info='duration', path='./tripinfos.xml', cav_set=None):
    tree = ET.parse(path)
    root = tree.getroot()

    info_list = []
    for child in root.findall('tripinfo'):
        if cav_set is None:
            info_list.append(child.attrib[info])
        elif cav_set is not None and int(child.attrib['id']) in cav_set:
            info_list.append(child.attrib[info])

    return info_list


def get_tripinfo(info='duration', get_cnt=False):
    tree = ET.parse('./tripinfos.xml')
    root = tree.getroot()

    car_list = []
    cumulated = 0

    # 遍历XML元素
    for child in root.findall('tripinfo'):
        car_list.append(child)
        cumulated += float(child.attrib[info])

    if get_cnt:
        return len(car_list)
    else:
        if len(car_list) == 0:
            return 107
        else:
            return cumulated / len(car_list)


def get_trip_waiting():
    tree = ET.parse('./tripinfos.xml')
    root = tree.getroot()

    car_list = []
    total_waiting_time = 0

    for child in root.findall('tripinfo'):
        car_list.append(child)
        total_waiting_time += float(child.attrib['waitingtime'])

    # 计算平均等待时间
    average_waiting_time = total_waiting_time / len(car_list)
    return average_waiting_time


def get_avg_speed():
    tree = ET.parse('./tripinfos.xml')
    root = tree.getroot()

    car_list = []
    cumulated = 0

    # 遍历XML元素
    for child in root.findall('tripinfo'):
        car_list.append(child)
        cumulated += float(child.attrib['routeLength']) / float(child.attrib['duration'])
    return cumulated / len(car_list)


def get_segment_info():
    waiting = np.zeros(6)
    duration = np.zeros(6)
    waiting_cnt = np.zeros(6, dtype=int)
    duration_cnt = np.zeros(6, dtype=int)
    throughput_cnt = np.zeros(6, dtype=int)
    tree = ET.parse('./tripinfos.xml')
    root = tree.getroot()

    for child in list(root):
        s = int(float(child.attrib['depart']))
        wait = float(child.attrib['timeLoss'])
        dura = float(child.attrib['duration'])
        if s >= 0:
            e = int(float(child.attrib['arrival']))
            l = int(s/3600)
            r = int(e/3600)
            waiting[l] += wait
            waiting_cnt[l] += 1
            duration[l] += dura
            duration_cnt[l] += 1
            throughput_cnt[r] += 1
    #print(waiting_cnt, waiting, duration_cnt, duration)
    #return waiting[:3] / waiting_cnt[:3], duration[:3] / duration_cnt[:3]
    return waiting[0] / waiting_cnt[0], duration[0] / duration_cnt[0]


def get_emission_info(info='CO2_abs'):
    tree = ET.parse('./tripinfos.xml')
    root = tree.getroot()

    car_list = []
    cumulated = 0

    # 遍历XML元素
    for child in root.findall('tripinfo'):
        car_list.append(child)
        em = child.find('emissions')
        cumulated += float(em.attrib[info]) / 1e6
    return cumulated / len(car_list)


def get_cav_info(cav_list=None, info=""):
    tree = ET.parse('./tripinfos.xml')
    root = tree.getroot()

    car_list = []
    cumulated = 0

    # 遍历XML元素
    for child in root.findall('tripinfo'):
        if int(child.attrib['id']) in cav_list:
            car_list.append(child)
            cumulated += float(child.attrib[info])
    return cumulated / len(car_list)


def get_mathot(save_path, info='CO2_abs'):
    in_edge = ['0_4_0', '0_3_0', '0_2_0', '0_1_0', '1_0_1', '2_0_1', '3_0_1', '4_0_1',
               '5_1_2', '5_2_2', '5_3_2', '5_4_2', '4_5_3', '3_5_3', '2_5_3', '1_5_3']
    out_edge = ['1_4_2', '1_3_2', '1_2_2', '1_1_2', '1_1_3', '2_1_3', '3_1_3', '4_1_3',
                '4_1_0', '4_2_0', '4_3_0', '4_4_0', '4_4_1', '3_4_1', '2_4_1', '1_4_1']

    tree = ET.parse('./tripinfos.xml')
    root = tree.getroot()
    od = np.zeros((16, 16))
    cnt = np.zeros((16, 16))

    for child in root.findall('tripinfo'):
        o, d = child.attrib['departLane'][5:-2], child.attrib['arrivalLane'][5:-2]
        if o not in in_edge or d not in out_edge:
            cnt += 1
            continue
        em = child.find('emissions')
        if "abs" in info:
            od[in_edge.index(o)][out_edge.index(d)] += float(em.attrib[info]) / 1e6
        else:
            od[in_edge.index(o)][out_edge.index(d)] += float(child.attrib[info])
        cnt[in_edge.index(o)][out_edge.index(d)] += 1

    im = plt.imshow(od/cnt, cmap='Blues', extent=[-0.5, 15.5, -0.5, 15.5], vmin=0, vmax=np.max(od/cnt))
    en_font = {'family': 'Times New Roman', 'size': 14}
    # 设置x和y轴刻度位置和标签
    l = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
    # l.reverse()
    plt.xticks(np.arange(0, 16, 1), l, fontname='Times New Roman', fontsize=14)
    l.reverse()
    plt.yticks(np.arange(0, 16, 1), l, fontname='Times New Roman', fontsize=14)
    # plt.gca().invert_xaxis()
    plt.tick_params(axis='x', which='both', bottom=False, top=True, labelbottom=False, labeltop=True)
    cbar = plt.colorbar(im)
    # print(cbar.ax.get_yticklabels())
    cbar.ax.set_yticklabels(np.round(cbar.get_ticks().astype(float), 1), fontname='Times New Roman', fontsize=14)
    plt.savefig(save_path + ".png", bbox_inches='tight')
    np.savetxt(save_path + ".csv", od/cnt, delimiter=',')

    # plt.show()
    plt.close()


def get_inter_queue(save_path):

    tree = ET.parse('./res/hangzhou/4_Phase.net.xml')
    root = tree.getroot()
    map_intersection = {}
    map_lane = {}
    for i in range(1, 5):
        for j in range(1, 5):
            map_intersection[f'intersection_{j}_{i}'] = (i-1)*4+j - 1

    for child in root.findall('edge'):
        if 'road_' in child.get('id'):
            dire = int(child.get('id')[-1])
            if '0' not in child.get('to') and '5' not in child.get('to'):
                map_lane[child.get('id')] = map_intersection[child.get('to')] * 4 + dire

    tree = ET.parse('./queues.xml')
    root = tree.getroot()
    # intersection_list = traci.trafficlight.getIDList()
    # queue_map = np.zeros((16, 4))
    queue_all = np.zeros((900, 64))

    for child in root.findall('data'):
        tm = int(float(child.get('timestep'))) // 3
        if tm < 300:
            continue
        tm -= 300
        # print(child.attrib['timestep'])
        lanes = child.find('lanes')
        if lanes.findall('lane') is not None:
            for lane in lanes.findall('lane'):
                if 'road_' in lane.get('id') and lane.get('id')[:-2] in map_lane:
                    queue_all[tm][map_lane[lane.get('id')[:-2]]] = \
                        max(float(lane.get('queueing_length')), queue_all[tm][map_lane[lane.get('id')[:-2]]])

    queue_res = np.mean(queue_all, axis=0)
    re_queue = np.zeros((8, 8))
    for i in range(64):
        sub_idx = i // 4
        sub_x, sub_y = sub_idx // 4, sub_idx % 4
        loc_x, loc_y = sub_x * 2, sub_y * 2
        if i % 4 == 0:
            re_queue[loc_x][loc_y] = queue_res[i]
        elif i % 4 == 1:
            re_queue[loc_x][loc_y+1] = queue_res[i]
        elif i % 4 == 2:
            re_queue[loc_x+1][loc_y+1] = queue_res[i]
        elif i % 4 == 3:
            re_queue[loc_x+1][loc_y] = queue_res[i]
    re_queue = np.flip(re_queue, axis=0)
    re_queue[0, ::2] = 0
    re_queue[-1, 1::2] = 0
    re_queue[1::2, 0] = 0
    re_queue[::2, -1] = 0
    im = plt.imshow(re_queue, cmap='Blues', extent=[-0.5, 7.5, -0.5, 7.5], vmin=0, vmax=np.max(re_queue))
    en_font = {'family': 'Times New Roman', 'size': 14}
    # 设置x和y轴刻度位置和标签
    l = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    # l.reverse()
    plt.xticks(np.arange(0, 8, 1), l, fontname='Times New Roman', fontsize=14)
    l.reverse()
    plt.yticks(np.arange(0, 8, 1), l, fontname='Times New Roman', fontsize=14)
    # plt.gca().invert_xaxis()
    plt.tick_params(axis='x', which='both', bottom=False, top=True, labelbottom=False, labeltop=True)
    cbar = plt.colorbar(im)
    # print(cbar.ax.get_yticklabels())
    cbar.ax.set_yticklabels(np.round(cbar.get_ticks().astype(float), 1), fontname='Times New Roman', fontsize=14)

    for i in range(8):
        for j in range(8):
            plt.text(i, 7 - j, str(int(re_queue[j][i])), ha='center', va='center', fontdict=en_font)

    plt.savefig(save_path + ".png", bbox_inches='tight')
    # np.savetxt(save_path + ".csv", queue_res, delimiter=',')

    plt.show()
    plt.close()

    del tree


def get_avg_qlength():
    tree = ET.parse('./res/hangzhou/4_Phase.net.xml')
    root = tree.getroot()
    map_intersection = {}
    map_lane = {}
    for i in range(1, 5):
        for j in range(1, 5):
            map_intersection[f'intersection_{j}_{i}'] = (i-1)*4+j - 1

    for child in root.findall('edge'):
        if 'road_' in child.get('id'):
            dire = int(child.get('id')[-1])
            if '0' not in child.get('to') and '5' not in child.get('to'):
                map_lane[child.get('id')] = map_intersection[child.get('to')] * 4 + dire

    tree = ET.parse('./queues.xml')
    root = tree.getroot()
    # intersection_list = traci.trafficlight.getIDList()
    # queue_map = np.zeros((16, 4))
    queue_all = np.zeros((900, 64))

    for child in root.findall('data'):
        tm = int(float(child.get('timestep'))) // 3
        if tm < 300:
            continue
        tm -= 300
        # print(child.attrib['timestep'])
        lanes = child.find('lanes')
        if lanes.findall('lane') is not None:
            for lane in lanes.findall('lane'):
                if 'road_' in lane.get('id') and lane.get('id')[:-2] in map_lane:
                    queue_all[tm][map_lane[lane.get('id')[:-2]]] = \
                        max(float(lane.get('queueing_length')), queue_all[tm][map_lane[lane.get('id')[:-2]]])

    del tree

    return np.mean(queue_all)
