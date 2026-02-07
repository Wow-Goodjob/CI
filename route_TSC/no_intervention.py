'''
import traci
import numpy as np
import tripinfo
import env
import xml.etree.ElementTree as ET
import pandas as pd
from datetime import datetime

sumoBinary="sumo"
sumoCmd=[sumoBinary,"-c","./res/hangzhou/train.sumocfg","--start"]

traci.start(sumoCmd)

speed=[]

def get_average_speed():
    vehicle_ids=traci.vehicle.getIDList()
    if not vehicle_ids:
        return 0
    total_speed=0
    count=0
    for veh_id in vehicle_ids:
        try:
            speed=traci.vehicle.getSpeed(veh_id)
            total_speed+=speed
            count+=1
        except traci.exceptions.TraCIException:
            continue
    return total_speed/count if count>0 else 0

for step in range(1800):
    speed.append(get_average_speed())
    traci.simulationStep()

traci.close()

print(f"Average Speed={sum(speed)/len(speed)} m/s")

class TripInfoAnalyzer:
    def __init__(self,tripinfo_file="tripinfo_output.xml"):
        self.tripinfo_file=tripinfo_file
        self.trip_data=None
        self.stats=None

    def parse_tripinfo(self):
        tree=ET.parse(self.tripinfo_file)
        root=tree.getroot()

        data = []
        for tripinfo in root.findall('tripinfo'):
            trip = {
                'id': tripinfo.get('id'),
                'depart': float(tripinfo.get('depart')),
                'arrival': float(tripinfo.get('arrival')),
                'duration': float(tripinfo.get('duration')),  # 行程时间
                'waitingTime': float(tripinfo.get('waitingTime', 0)),
                'timeLoss': float(tripinfo.get('timeLoss', 0)),
                'routeLength': float(tripinfo.get('routeLength')),
                'speed': float(tripinfo.get('speed') or 0),
                'vType': tripinfo.get('vType', 'unknown'),
                'departLane': tripinfo.get('departLane', ''),
                'arrivalLane': tripinfo.get('arrivalLane', '')
            }

            if 'departDelay' in tripinfo.attrib:
                trip['departDelay'] = float(tripinfo.get('departDelay'))

            data.append(trip)

        self.trip_data = pd.DataFrame(data)
        return self.trip_data

    def calculate_statistics(self):
        if self.trip_data is None:
            self.parse_tripinfo()

        df = self.trip_data

        # 基本统计
        stats = {
            'total_vehicles': len(df),
            'avg_trip_time': df['duration'].mean(),
            'median_trip_time': df['duration'].median(),
            'min_trip_time': df['duration'].min(),
            'max_trip_time': df['duration'].max(),
            'std_trip_time': df['duration'].std(),
            'avg_waiting_time': df['waitingTime'].mean(),
            'avg_time_loss': df['timeLoss'].mean(),
            'avg_speed': df['speed'].mean(),
            'avg_distance': df['routeLength'].mean()
        }

        # 添加百分位数
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            stats[f'percentile_{p}'] = np.percentile(df['duration'], p)

        # 按车辆类型分类统计
        if 'vType' in df.columns:
            type_stats = df.groupby('vType')['duration'].agg(['count', 'mean', 'std', 'min', 'max'])
            stats['by_vehicle_type'] = type_stats.to_dict('index')

        self.stats = stats
        return stats

    def print_detailed_report(self):
        """打印详细报告"""
        if self.stats is None:
            self.calculate_statistics()

        stats = self.stats

        print("=" * 60)
        print("行程时间分析报告")
        print("=" * 60)
        print(f"\n总体统计:")
        print(f"  总车辆数: {stats['total_vehicles']}")
        print(f"  平均行程时间: {stats['avg_trip_time']:.2f} 秒")
        print(f"  中位数行程时间: {stats['median_trip_time']:.2f} 秒")
        print(f"  最短行程时间: {stats['min_trip_time']:.2f} 秒")
        print(f"  最长行程时间: {stats['max_trip_time']:.2f} 秒")
        print(f"  行程时间标准差: {stats['std_trip_time']:.2f} 秒")

        print(f"\n其他指标:")
        print(f"  平均等待时间: {stats['avg_waiting_time']:.2f} 秒")
        print(f"  平均时间损失: {stats['avg_time_loss']:.2f} 秒")
        print(f"  平均速度: {stats['avg_speed']:.2f} m/s ({stats['avg_speed'] * 3.6:.2f} km/h)")
        print(f"  平均行程距离: {stats['avg_distance']:.2f} 米")

        print(f"\n行程时间百分位数:")
        for p in [10, 25, 50, 75, 90, 95, 99]:
            print(f"  {p}% 车辆行程时间 ≤ {stats[f'percentile_{p}']:.2f} 秒")

        # 按车辆类型统计
        if 'by_vehicle_type' in stats:
            print(f"\n按车辆类型统计:")
            for vtype, type_stat in stats['by_vehicle_type'].items():
                print(f"\n  {vtype}:")
                print(f"    车辆数: {type_stat['count']}")
                print(f"    平均行程时间: {type_stat['mean']:.2f} 秒")
                print(f"    标准差: {type_stat['std']:.2f} 秒")

    def export_to_csv(self, output_file="trip_analysis.csv"):
        if self.trip_data is None:
            self.parse_tripinfo()

        # 导出原始数据
        self.trip_data.to_csv(output_file, index=False)

        # 导出统计摘要
        stats_df = pd.DataFrame([self.stats])
        stats_df.to_csv(f"stats_{output_file}", index=False)

        print(f"数据已导出到 {output_file}")

    def plot_trip_time_distribution(self, save_path=None):
        import matplotlib.pyplot as plt
        import seaborn as sns

        if self.trip_data is None:
            self.parse_tripinfo()

        df = self.trip_data

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        axes[0, 0].hist(df['duration'], bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('行程时间 (秒)')
        axes[0, 0].set_ylabel('车辆数')
        axes[0, 0].set_title('行程时间分布')
        axes[0, 0].axvline(df['duration'].mean(), color='red', linestyle='--',
                           label=f'平均: {df["duration"].mean():.1f}s')
        axes[0, 0].legend()

        axes[0, 1].boxplot(df['duration'], vert=True)
        axes[0, 1].set_ylabel('行程时间 (秒)')
        axes[0, 1].set_title('行程时间箱线图')

        sorted_times = np.sort(df['duration'])
        yvals = np.arange(len(sorted_times)) / float(len(sorted_times))
        axes[1, 0].plot(sorted_times, yvals)
        axes[1, 0].set_xlabel('行程时间 (秒)')
        axes[1, 0].set_ylabel('累计比例')
        axes[1, 0].set_title('行程时间累计分布')
        axes[1, 0].grid(True)

        axes[1, 1].scatter(df['routeLength'], df['duration'], alpha=0.5)
        axes[1, 1].set_xlabel('行程距离 (米)')
        axes[1, 1].set_ylabel('行程时间 (秒)')
        axes[1, 1].set_title('行程时间 vs 距离')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到 {save_path}")

        plt.show()

if __name__ =="__main__":
    analyzer=TripInfoAnalyzer("tripinfo_output.xml")
    df=analyzer.parse_tripinfo()
    stats = analyzer.calculate_statistics()

    analyzer.print_detailed_report()

    analyzer.export_to_csv("trip_analysis_results.csv")

    analyzer.plot_trip_time_distribution("trip_time_distribution.png")
'''
import matplotlib.pyplot as plt
import numpy as np

# 从提供的CSV内容中提取数据
data = [
    497.5586924219911, 489.016081871345, 492.7214814814815, 492.9117647058824,
    496.49479940564635, 497.2842261904762, 499.2290184921764, 473.3433908045977,
    473.0489208633094, 471.36312056737586, 470.3690987124464, 470.7353361945637,
    469.89078014184395, 468.20714285714286, 465.89161849710985, 465.10593220338984,
    465.9115168539326, 466.19127988748244, 469.82885431400285, 468.16524216524215,
    477.6652298850575, 469.0439093484419, 468.87535816618913, 469.9943181818182,
    471.0818965517241, 473.6025459688826, 469.6623748211731, 471.163610719323,
    471.8868194842407, 471.70258620689657, 471.2207792207792, 471.9265536723164,
    471.22111269614834, 476.1889534883721, 467.18634423897583, 470.1035460992908,
    466.9088319088319, 472.72714285714284, 475.8904694167852, 467.6768115942029,
    471.42571428571426, 469.3059490084986, 470.1039886039886, 472.7245337159254,
    471.43926553672316, 473.8807471264368, 469.41404011461316, 470.5681818181818,
    466.1671469740634, 467.206847360913
]

plt.figure(figsize=(12, 7))

plt.plot(data, linewidth=2, markersize=5, label='Average Time')

plt.xlabel('Index', fontsize=12)
plt.ylabel('Average Time', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)

# 设置x轴刻度
plt.xticks(range(0, len(data), 5))

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()
