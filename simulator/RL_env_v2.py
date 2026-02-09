import numpy as np
import pickle
import os
from flow import Flow, get_generator_power
import yaml
import torch
import json
from EV_sim import EV_simulator

BASE = 1e8


def calculate_grid_cost(P_gen):
    max_gen = np.max(P_gen, axis=1)
    large_index = np.where(max_gen > 500)[0]
    mid_index = np.where((max_gen <= 500) & (max_gen > 300))[0]
    small_index = np.where(max_gen <= 300)[0]

    ramping = P_gen[:, 4:] - P_gen[:, :-4]
    cost = np.zeros(P_gen.shape[0])
    for i in large_index:
        row = np.abs(ramping[i])
        cost[i] = (
            np.sum(row[row < max_gen[i] * 0.30]) * 1.96
            + np.sum(row[row >= max_gen[i] * 0.30]) * 1.96 * 6
        )
    for i in mid_index:
        row = np.abs(ramping[i])
        cost[i] = (
            np.sum(row[row < max_gen[i] * 0.35]) * 2.45
            + np.sum(row[row >= max_gen[i] * 0.35]) * 2.45 * 6
        )
    for i in small_index:
        row = np.abs(ramping[i])
        cost[i] = (
            np.sum(row[row < max_gen[i] * 0.32]) * 3.34
            + np.sum(row[row >= max_gen[i] * 0.32]) * 3.34 * 6
        )
    return cost


class myenv:
    def __init__(
        self,
        num_agents=803,
        topo_path="path to the topology",
        power_path="path to the power demand",
        solar_path="path to the solar power",
        trans_2_area="path to the trans_2_area",
        EV_driving_power=None,
        EV_location=None,
        Nev=200000,
    ):

        self.elec_json = json.load(open(topo_path, "r"))
        self.nodes = [x for x in self.elec_json if "id" in x["properties"]]
        ## power demand: N*96
        power_ref = np.load(power_path)
        self.power_ref = power_ref
        self.N_agent = num_agents
        self.N_plant = 25

        solar_ref = np.load(solar_path)

        self.solar_template = solar_ref / np.sum(solar_ref)
        # interpolate 24 - > 96
        self.solar_template = np.interp(
            np.arange(0, 96), np.arange(0, 24) * 4, self.solar_template
        )
        self.solar_template = self.solar_template / np.sum(self.solar_template)
        self.solar_template_ref = self.solar_template.reshape(1, 96)
        self.trans110_2_area = pickle.load(open(trans_2_area, "rb"))

        self.area_vec = np.zeros((self.N_agent,))
        for k, v in self.trans110_2_area.items():
            idx = int(k - 4e8)
            self.area_vec[idx] = v

        self.solar_percentage = np.zeros((self.N_agent,))
        self.solar_power = np.zeros((self.N_agent, 96))
        self.V2G_power = np.zeros((self.N_agent, 96))

        if EV_location is not None:
            self.ev_driving_power = np.load(EV_driving_power)
            self.ev_location = np.load(EV_location)
            self.ev_location = self.ev_location - 4e8 if np.max(self.ev_location) > 4e8 else self.ev_location
            self.ev_location = self.ev_location.astype(int)
            self.Nev = Nev
            self.ev_driving_power = self.ev_driving_power[:self.Nev,:]
            self.ev_location = self.ev_location[:self.Nev,:]

            self.V2G_facility = np.zeros((self.N_agent,))
            # set the peak-shaving and vally-filling target
            for i in range(self.N_agent):
                self.add_pv_by_node(i, 0.15)

            self.tot_load = np.sum(self.power_ref, axis=0) * 1000
            self.PV_load = np.sum(self.solar_power, axis=0) * 1000
            self.default_V2G_target = np.mean(self.tot_load - self.PV_load) + self.PV_load - self.tot_load
            self.default_V2G_target *= -1

            self.N_ev_station = np.zeros((self.N_agent, 96))

            for t in range(96):
                for idx in range(self.ev_location.shape[0]):
                    self.N_ev_station[self.ev_location[idx, t], t] += 1
           

            self.EV_sim = EV_simulator(
                EV_driving_power=self.ev_driving_power,
                EV_location=self.ev_location,
                N_ev_station=self.N_ev_station,
                N_region=self.N_agent,
            )

        self.location = np.array(
            [
                x["geometry"]["coordinates"]
                for x in self.nodes
                if x["properties"]["id"] // BASE == 4
            ]
        )
        closest_power_plant = json.load(
            open(
                "path to closest_generator.json"
            )
        )
        self.closest_power_plant = {
            int(k): int(v - self.N_agent) for k, v in closest_power_plant.items()
        }

    def add_pv_by_node(self, idx, pv_ratio):
        assert idx < self.N_agent and idx >= 0
        self.solar_percentage[idx] = pv_ratio
        self.solar_power[idx] = (
            4 / 1e3 * self.solar_template_ref * self.area_vec[idx] * pv_ratio
        )

    def add_pv_by_bbox(self, bbox, pv_ratio):
        lon_min, lat_min, lon_max, lat_max = bbox
        add_idx = [
            int(x["properties"]["id"] % BASE)
            for x in self.nodes
            if x["geometry"]["coordinates"][0] > lon_min
            and x["geometry"]["coordinates"][0] < lon_max
            and x["geometry"]["coordinates"][1] > lat_min
            and x["geometry"]["coordinates"][1] < lat_max
            and x["properties"]["id"] // BASE == 4
        ]
        for idx in add_idx:
            self.add_pv_by_node(idx, pv_ratio)

    def set_pv_from_vector(self, PV):
        assert PV.shape[0] == self.N_agent
        self.solar_percentage = PV
        action_mat = (self.solar_percentage * self.area_vec).reshape(self.N_agent, 1)
        self.solar_power = 4 / 1e3 * action_mat @ self.solar_template_ref

    def get_power_demand(self):
        return self.power_ref - self.solar_power - self.V2G_power

    def get_mask_by_bbox(self, bbox):
        lon_min, lat_min, lon_max, lat_max = bbox
        add_idx = [
            int(x["properties"]["id"] % BASE)
            for x in self.nodes
            if x["geometry"]["coordinates"][0] > lon_min
            and x["geometry"]["coordinates"][0] < lon_max
            and x["geometry"]["coordinates"][1] > lat_min
            and x["geometry"]["coordinates"][1] < lat_max
            and x["properties"]["id"] // BASE == 4
        ]
        mask = np.zeros((self.N_agent,))
        mask[add_idx] = 1
        return mask

    def simulate_v2g(self, action):
        self.V2G_facility = action
        V2G_power, NEV_node, satisfied = self.EV_sim.run(action)
        self.V2G_power = V2G_power / 1e3

        net_load = self.get_power_demand()
        total_net_load = np.sum(net_load, axis=0)
        ramping = np.sum(np.abs(total_net_load[:1] - total_net_load[1:]))

        return V2G_power, NEV_node, ramping, satisfied
    
    def set_N_ev(self, Nev):
        self.ev_driving_power = self.ev_driving_power[:Nev, :]
        self.ev_location = self.ev_location[:Nev, :]
        self.Nev = Nev

    def set_V2G_target(self, target):
        self.default_V2G_target = target

    def get_feature_info(self):
        avail_feat = {
            "lon": "longitude of the location",
            "lat": "latitude of the location",
            "demand_profile": "np.array, shape (96,), representing the demand profile (MW) of the location at 0.25h (15min) resolution.",
            "V2G_target": "np.array, shape (96,), representing the target V2G power (MW) of the whole city at 0.25h (15min) resolution.",
            "solar_profile": "np.array, shape (96,), representing the solar profile (MW/m^2) of the location at 0.25h (15min) resolution.",
            "rooftop_area": "float, representing the available rooftop area of the location in m^2.",
            "N_ev_number": "np.array, shape (96,), representing the number of EV at the location 0.25h (15min) resolution.",
            "closest_PP_id": "int [0-24], representing the id of the closest power plant to the location. Helps for grid cost calculation.",
        }
        return avail_feat


if __name__ == "__main__":
    pass
