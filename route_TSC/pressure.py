import numpy as np
import traci as tc
import traci
import pandas as pd

def start(path):
    binary="sumo"
    sumoCmd=[binary,"-c",path,"--start"]
    tc.start(sumoCmd)

def step(sec=0):
    tc.simulationStep(step=sec)

def close():
    tc.close()

def get_intersection_id():
    intersection_ids=tc.trafficlight.getIDList()
    return intersection_ids

def get_lane_id():
    lane=tc.edge.getIDList()
    lane=[_ for _ in lane if _[0]!=":"]
    return lane

def get_vehicles_in(intersection):
    current_state=traci.trafficlight.getRedYellowGreenState(intersection)
    controlled_lanes=traci.trafficlight.getControlledLanes(intersection)
    count=0
    for lane_id,(lane_id,signal) in enumerate(zip(controlled_lanes,current_state)):
        if signal in ['G','g']:
            count+=len(traci.lane.getLastStepVehicleIDs(lane_id))
    return count


def calc_pressure(intersection):
    #veh_in=get_vehicles_in(intersection)
    #veh_out=get_vehicles_out(intersection)
    count = 0
    controlled_links = traci.trafficlight.getControlledLinks(intersection)
    current_state = traci.trafficlight.getRedYellowGreenState(intersection)
    for i in range(len(current_state)):
        state=current_state[i]
        if state in ['G','g']:
            in_veh=len(traci.lane.getLastStepVehicleIDs(controlled_links[i][0][0]))/3
            out_veh=len(traci.lane.getLastStepVehicleIDs(controlled_links[i][0][1]))
            count+=(in_veh-out_veh)
    return abs(count)

def pressure_table(path):
    start(path)
    intersections=get_intersection_id()
    lanes=get_lane_id()
    data=[]
    for i in range(1800):
        Pre=[]
        for intersection in intersections:
            pressure=calc_pressure(intersection)
            Pre.append(pressure)
        data.append(Pre)
        step()
    close()
    df = pd.DataFrame(data, columns=intersections)
    df.to_csv('Pressure.csv',index=False,encoding='utf-8-sig')

if __name__=='__main__':
    path='./res/hangzhou/train.sumocfg'
    pressure_table(path)