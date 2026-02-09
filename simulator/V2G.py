# task/pv.py
import numpy as np
import sys
sys.path.append("/data1/energy_data/lisb_PV_planning/agent/tools")
from flow import Flow, get_generator_power
from RL_env_v2 import myenv, calculate_grid_cost


def fetch_data(idx, env, feature):
    avail_feat = set(["lon", "lat", "demand_profile", "target_pattern", "N_ev_node", "closest_PP_id"])
    data_dict = {}
    for f in feature:
        if f not in avail_feat:
            data_dict[f] = None
        if f == "lon":
            data_dict[f] = env.location[idx][0]
        elif f == "lat":
            data_dict[f] = env.location[idx][1]
        elif f == "demand_profile":
            data_dict[f] = env.power_ref[idx, :]
        elif f == "target_pattern":
            data_dict[f] = env.default_V2G_target / 1e3
        elif f == "N_ev_node":
            data_dict[f] = env.N_ev_station[idx,:]
        elif f == "closest_PP_id":
            data_dict[f] = env.closest_power_plant[idx]
    return data_dict

class V2GTaskExecutor:
    def __init__(self, N_ev, features):
        self.features = features
        self.Nev = N_ev

    def build_inputs(self, env, flow):
        args = []
        for i in range(len(env.location)):
            args.append(fetch_data(i, env, self.features))
        return args
    
    def post_process(self, action_result, env, flow, mask):
        # set all nan to 0
        action_result = np.nan_to_num(action_result)
        return np.array(action_result) * 1e3

    def analysis_result(self, solution, env, flow, mask):
        V2G_power, NEV_node, ramping, satisfied = env.simulate_v2g(solution)
        power_demand_now = env.get_power_demand()
        gen_power = get_generator_power(flow, power_demand_now)
        ramping_cost = calculate_grid_cost(gen_power)
        ramping = gen_power[:, 4:] - gen_power[:, :-4]
        max_gen = np.max(gen_power, axis=1)
        ramping_sum = np.sum(np.abs(ramping), axis=1)
        unit_ramping_cost = ramping_cost / ramping_sum

        result = {
            "solution": solution,
            "satisfied": satisfied,
            "ramping": ramping,
            "ramping_cost": np.sum(ramping_cost),
            "unit_ramping_cost": unit_ramping_cost,
        }
        return result


    def compute_metrics(self, sim_result, env, flow, mask):
        metrics ={}
        # artifact = {"Info": "V2G_score_{t} represents the percentage of V2G target is satisfied at timestep t, t \in [0,95]."}
        artifact = {}
        ramping_cost = sim_result["ramping_cost"]
        ori_solution = sim_result["solution"]
        mean = np.mean(ori_solution)
        metrics["Numerical_stability"] = 1 if (mean < 0.95 and mean > 0.05) else 0
        metrics["Power_grid_score"] = 10 - np.sum(ramping_cost) / 4 / 1e4
        metrics["combined_score"] = metrics["Power_grid_score"] + metrics["Numerical_stability"]
        metrics["combined_score"] = metrics["Power_grid_score"]
        for t in range(0,96,8):
            metrics[f"V2G_score_{t}"] = sim_result["satisfied"][t]
        for i in range(len(sim_result["unit_ramping_cost"])):
            metrics[f"fitness_power_plant_{i}"] = -1 * sim_result["unit_ramping_cost"][i] + 6
        
        return metrics, artifact
