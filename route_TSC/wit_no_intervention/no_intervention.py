import traci
import numpy as np
import tripinfo

sumoBinary="sumo"
sumoCmd=[sumoBinary,"-c","SUMO_data/train.sumocfg","--start"]

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

for step in range(3600):
    speed.append(get_average_speed())
    traci.simulationStep()

traci.close()

print(f"Average Speed={sum(speed)/len(speed)} m/s")