# task/pv.py
import numpy as np
import sys
sys.path.append("/data1/energy_data/lisb_PV_planning/agent/tools")
from flow import Flow, get_generator_power
from RL_env_v2 import myenv, calculate_grid_cost


def binary_search(result, target_solar, env, mask):
    min_scale = 0
    max_scale = 1000000
    result[result < 1e-3] = 1e-3
    while max_scale - min_scale > 1e-6:
        mid_scale = (min_scale + max_scale) / 2
        test = env.solar_percentage * (1 - mask) + result * mask * mid_scale
        env.set_pv_from_vector(np.clip(test, 0, 1))
        power_per_node = np.sum(env.solar_power, axis=1) / 4
        total_solar = np.sum(power_per_node * mask)
        if total_solar < target_solar:
            min_scale = mid_scale
        else:
            max_scale = mid_scale
    return np.clip(test, 0, 1), min_scale

def fetch_data(idx, env, feature):
    avail_feat = set(["lon", "lat", "demand_profile", "solar_profile", "rooftop_area", "closest_PP_id"])
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
        elif f == "solar_profile":
            data_dict[f] = env.solar_template_ref.reshape((96,))
        elif f == "rooftop_area":
            data_dict[f] = env.area_vec[idx]
        elif f == "closest_PP_id":
            data_dict[f] = env.closest_power_plant[idx]
    return data_dict

class PVTaskExecutor:
    def __init__(self, target_pv, features):
        self.target_pv = target_pv
        self.features = features

    def build_inputs(self, env, flow):
        args = []
        for i in range(len(env.location)):
            args.append(fetch_data(i, env, self.features))
        return args

    def analysis_result(self, result, env, flow, mask):
        solution = result["plan"]
        PV_now = env.solar_percentage
        p = PV_now * (1 - mask) + solution * mask
        
        env.set_pv_from_vector(p)
        power_demand_now = env.power_ref - env.solar_power
        gen_power = get_generator_power(flow, power_demand_now)

        ramping_cost = calculate_grid_cost(gen_power)
        ramping = gen_power[:, 4:] - gen_power[:, :-4]
        max_gen = np.max(gen_power, axis=1)
        ramping_sum = np.sum(np.abs(ramping), axis=1)
        unit_ramping_cost = ramping_cost / ramping_sum

        power_per_node = np.sum(env.solar_power, axis=1) / 4
        total_pv = np.sum(power_per_node * mask) / 1e4
        
        return {
            "ramping_cost": np.sum(ramping_cost)  / 4 / 1e4 ,
             "gen_power": gen_power, "total_pv": total_pv, "unit_ramping_cost": unit_ramping_cost, "ori_plan": result["ori_plan"]}

    def post_process(self, action_result, env, flow, mask):
        print("Statistic of Result:", np.mean(action_result), np.std(action_result), np.min(action_result), np.max(action_result))
        plan = np.array(action_result)
        plan = plan / np.max(plan)
        plan, _ = binary_search(plan, self.target_pv, env, mask)
        # print(plan)
        np.save("plan.npy", plan)
        return {"ori_plan": action_result, "plan": plan}

    def compute_metrics(self, sim_result, env, flow, mask):
        gen_power = sim_result["gen_power"]
        ramping_cost = calculate_grid_cost(gen_power)
        metrics ={}

        for i in range(len(sim_result["unit_ramping_cost"])):
            metrics[f"fitness_power_plant_{i}"] = -1 * sim_result["unit_ramping_cost"][i] + 6

        ori_plan = sim_result["ori_plan"]
        mean, std, min, max = np.mean(ori_plan), np.std(ori_plan), np.min(ori_plan), np.max(ori_plan)
        metrics["Numerical_stability"] = 1 if (mean < 95 and mean > 5) else 0
        metrics["Power_grid_score"] = 10 - np.sum(ramping_cost) / 4 / 1e4
        metrics["combined_score"] = metrics["Power_grid_score"] + metrics["Numerical_stability"]

        artifact = {"INFO": "We aim at increase the fitness score of every power plant, therefore achieve a lower cost of the whole system.","STATISTIC_INFO": f"Mean action: {mean}, std: {std}, min: {min}, max: {max}."}
        return metrics, artifact
