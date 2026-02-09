import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

class EV_simulator:
    def __init__(self, EV_driving_power, EV_location, N_ev_station, N_region=803):
        np.random.seed(0)
        self.EV_driving_power = EV_driving_power
        self.EV_location = EV_location - 4e8 if np.max(EV_location) > 4e8 else EV_location
        self.EV_location = self.EV_location.astype(int)
        self.EV_per_loc = np.zeros((N_region,))
        for i in range(self.EV_location.shape[0]):
            self.EV_per_loc[self.EV_location[i,0]] += 1
        
        self.min_soc = np.sum(EV_driving_power, axis=1) / 4 + 6
        self.max_soc = np.ones_like(self.min_soc) * 60 # max 60kwh
        # self.soc = np.random.uniform(self.min_soc, self.max_soc)
        self.soc = self.min_soc.copy() + 10
        self.N_ev_station = N_ev_station
        self.N_region = N_region

        print("ENV_init finished ")

    def run(self, V2G_target):
        # N_station: array of [N,T] of int or [0-1]
        satisfied = np.zeros((96,))
        if len(V2G_target.shape) == 1 and V2G_target.shape[0] == 96:
            # repeat N times -> [N,T]
            V2G_target = np.repeat(V2G_target[np.newaxis,:], self.N_region, axis=0)
            weight = self.EV_per_loc / np.sum(self.EV_per_loc)
            weight = np.repeat(weight[:,np.newaxis], 96, axis=1)
            V2G_target = V2G_target * weight
        # print(np.sum(V2G_target, axis=0))
        # EV_participation: 0~1, representing the ratio of EVs participating in the V2G
        V2G_power = np.zeros((len(V2G_target),96))
        NEV_node = np.zeros((len(V2G_target),96))
        for t in tqdm(range(96), disable=True):
            for i in range(self.EV_driving_power.shape[0]):
                location_now = self.EV_location[i,t]
                tgt = V2G_target[location_now,t]
                if self.EV_driving_power[i,t] > 0:
                    self.soc[i] -= self.EV_driving_power[i,t] / 4
                else:
                    if (V2G_power[location_now,t] < tgt - 10): # EV 放电
                        if (self.soc[i] > self.min_soc[i] + 2.5):
                            self.soc[i] -= 10 / 4
                            V2G_power[location_now,t] += 10
                            NEV_node[location_now,t] += 1
                    elif V2G_power[location_now,t] > tgt + 10: # EV 充电
                        if (self.soc[i] < self.max_soc[i] - 2.5):
                            self.soc[i] += 10 / 4
                            V2G_power[location_now,t] -= 10
                            NEV_node[location_now,t] += 1
                if np.max(np.abs(V2G_power[:,t] - V2G_target[:,t])) < 10:
                    break
                
            satisfied[t] = 1 - np.sum(V2G_power[:,t]) / np.sum(V2G_target[:,t]) if np.sum(V2G_target[:,t]) > 0 else 1
                
        satisfied = np.clip(satisfied, 0, 1)
        return V2G_power, NEV_node, satisfied

if __name__ == '__main__':
    pass