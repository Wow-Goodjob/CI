import traci
import pandas as pd
import xml.etree.ElementTree as ET
import traci as tc

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

def parse_tripinfo(xml_file):
    tree=ET.parse(xml_file)
    root=tree.getroot()
    data=[]
    for child in root:
        if child.tag == 'tripinfo':
            data.append({
                'id':child.attrib['id'],
                'duration': float(child.attrib['duration']),
                'routeLength': float(child.attrib['routeLength']),
                'depart': float(child.attrib['depart']),
            })
    return pd.DataFrame(data)

def calc_ATE(xml_file_base,xml_file_route):
    df_base=parse_tripinfo(xml_file_base)
    df_exp=parse_tripinfo(xml_file_route)

    df_merged=pd.merge(df_base,df_exp,on='id',suffixes=('_base','_exp'))
    print(df_base['duration'].mean())
    print(df_exp['duration'].mean())
    df_merged['diff']=df_merged['duration_exp']-df_merged['duration_base']
    ate=df_merged['diff'].mean()

    return ate

if __name__=='__main__':
    path_base='./res/hangzhou/tripinfo_output.xml'
    path_route='tripinfo_output_old.xml'
    print(calc_ATE(path_base,path_route))