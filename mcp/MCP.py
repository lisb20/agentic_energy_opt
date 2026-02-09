# THE MCP server
from fastmcp import FastMCP
import asyncio
import sys

# add path
sys.path.append("/data1/energy_data/lisb_PV_planning/agent/tools")
from flow import Flow, get_generator_power
from RL_env_v2 import myenv
from RL_env_v2 import calculate_grid_cost as calculate_grid_cost_source
import numpy as np
import yaml
import pickle
from openai import OpenAI
import json

MODEL = "Qwen/Qwen3-8B"
LLM = OpenAI(
    api_key="sk-wymrwwfjljemycwylawaypaaxefkihtrbbxicnfwlrnybvtu",
    base_url="https://api.siliconflow.cn/v1",
)

server = FastMCP("pv_sim_server")
GLOBAL_STATE = {}


@server.tool()
def init(TASK_TYPE):
    global GLOBAL_STATE
    """initialize the server"""
    print("CALLING <init>", "TASK_TYPE", TASK_TYPE)
    if TASK_TYPE == "PV":
        flow_config = "/data1/energy_data/lisb_PV_planning/BEJ/flow/config.yaml"
        config = yaml.safe_load(open(flow_config, "r"))
        flow_data = (
            "/data1/energy_data/lisb_PV_planning/BEJ/flow/elec_flow_input_map0221.json"
        )

        GLOBAL_STATE = {
            "env": myenv(EV_driving_power=None, EV_location=None),
            "flow": Flow(config, flow_data),
            "gen_power": np.zeros((25, 96)),
            "power_demand": np.zeros((803, 96)),
        }
    elif TASK_TYPE == "V2G":
        flow_config = "/data1/energy_data/lisb_PV_planning/BEJ/flow/config.yaml"
        config = yaml.safe_load(open(flow_config, "r"))
        flow_data = (
            "/data1/energy_data/lisb_PV_planning/BEJ/flow/elec_flow_input_map0221.json"
        )

        env = myenv(
            topo_path="/data1/energy_data/lisb_PV_planning/BEJ/flow/elec_flow_input_map0221.json",
            power_path="/data1/energy_data/lisb_PV_planning/BEJ/flow/power_map0221.npy",
            EV_driving_power="/home/lisibo/v2g/opt_data_0221/Pd.npy",
            EV_location="/home/lisibo/v2g/opt_data_0221/loc_mat.npy",
            Nev=300000,
        )
        GLOBAL_STATE = {
            "env": env,
            "flow": Flow(config, flow_data),
            "gen_power": np.zeros((25, 96)),
            "power_demand": np.zeros((803, 96)),
        }

    return "init"


@server.tool(
    description="""
Set the photovoltaic deployment ratio to a same value (0-1) of the grid nodes in a certain bounding box. Only used if no checkpoints is available \
<input format>
bbox: list[float]: [xmin, ymin, xmax, ymax]
ratio: float
<output format>
"""
)
def set_PV_by_bbox(bbox: list, ratio: float) -> str:
    """set the photovoltaic ratio of the grid of bbox"""
    print("CALLING <set_pv_by_bbox> INPUT_ARGS: ", "bbox ", bbox, "ratio ", ratio)
    env = GLOBAL_STATE["env"]
    env.add_pv_by_bbox(bbox, ratio)
    GLOBAL_STATE["env"] = env
    power_demand = env.get_power_demand()
    GLOBAL_STATE["power_demand"] = power_demand
    return "PV set"


@server.tool(
    description="""evaluate the current PV deployment plan, and return the PV generation, power demand and operation cost now.
<input format>
bbox: list[float] or None; the bounding box [xmin, ymin, xmax, ymax] of the optimization area. If None, the optimization is performed on the entire grid.
<output format>
{{
    "PV generation (MWh)": PV_all,
    "Power demand (MWh)": demand_all,
    "Operation cost ($)": cost,
    "info": "Current PV plan is evaluated"
}}
"""
)
def run_PV_simulation(bbox=None):
    """get the power demand, current PV plan and operation cost of the grid"""
    print("CALLING <run_simulation>")
    env = GLOBAL_STATE["env"]
    if bbox is None:
        mask = np.ones((GLOBAL_STATE["env"].N_agent,))
    else:
        mask = env.get_mask_by_bbox(bbox)

    PV_all = np.sum(np.sum(env.solar_power, axis=1) / 4 * mask)
    demand_all = np.sum(np.sum(env.power_ref, axis=1) / 4 * mask)
    gen_power = get_generator_power(GLOBAL_STATE["flow"], GLOBAL_STATE["power_demand"])
    GLOBAL_STATE["gen_power"] = gen_power
    cost = calculate_grid_cost_source(gen_power) / 4
    return {
        "PV generation (MWh)": PV_all,
        "Power demand (MWh)": demand_all,
        "Operation cost ($)": np.sum(cost),
        "info": "Current PV plan is evaluated",
    }


@server.tool(
    description="""Load the PV plan from a file, which will cover the current PV plan. 
<input format>
fp: str: file path
"""
)
def load_PV_from_file(fp):
    """load the PV plan from a file"""
    print("CALLING <set_PV_from_file> INPUT_ARGS: ", "fp ", fp)
    data = np.load(fp)
    GLOBAL_STATE["env"].set_pv_from_vector(data)
    GLOBAL_STATE["power_demand"] = GLOBAL_STATE["env"].get_power_demand()
    GLOBAL_STATE["gen_power"] = get_generator_power(
        GLOBAL_STATE["flow"], GLOBAL_STATE["power_demand"]
    )
    return f"PV set successfully from {fp}!"

@server.tool(
    description="""
Load a V2G plan from a file, which will cover the current V2G plan.
<input format>
<output format>
"""
)
def load_V2G_from_file(fp):
    """load the V2G plan from a file"""
    print("CALLING <set_V2G_from_file> INPUT_ARGS: ", "fp ", fp)
    GLOBAL_STATE["env"].set_V2G_target(np.load(fp))
    return "V2G set successfully from {fp}!"


@server.tool(
    description="""
Simulate the V2G problem with the current V2G target curve.
<input format>
number_ev: int or None; the number of EVs to simulate. If None, the number of EVs is set to the default value.
<output format>
"""
)
def run_V2G_simulation(number_ev=None):
    """run the V2G simulation"""
    print("CALLING <run_V2G_simulation> INPUT_ARGS: ", "number_ev ", number_ev)
    if number_ev is None:
        number_ev = GLOBAL_STATE["env"].Nev
    else:
        GLOBAL_STATE["env"].set_N_ev(number_ev)

    V2G_target = GLOBAL_STATE["env"].default_V2G_target
    print(np.sum(V2G_target, axis=0))
    V2G_power, NEV_node, ramping, satisfied = GLOBAL_STATE["env"].simulate_v2g(
        V2G_target
    )
    GLOBAL_STATE["power_demand"] = GLOBAL_STATE["env"].get_power_demand()
    GLOBAL_STATE["gen_power"] = get_generator_power(
        GLOBAL_STATE["flow"], GLOBAL_STATE["power_demand"]
    )
    V2G_storage = np.sum(V2G_power[V2G_power > 0]) / 1e3
    cost = calculate_grid_cost_source(GLOBAL_STATE["gen_power"]) / 4
    return {
        "V2G storage (MWh)": V2G_storage,
        "Operation cost ($)": np.sum(cost),
        "info": "V2G problem simulated successfully!",
    }


@server.tool(
    description="""
Optimization by the alpha-evolve algorithm. The best result is saved in the file "xx.npy"
<input format>
number_ev: int or None; the number of EVs to simulate. If None, the number of EVs is set to the default value.
domain: str; selected from ["PV" or "V2G"]; the domain to optimize.
task_description: str; the description of the task.
<output format>
"""
)
def optimize_by_alpha_evolve(number_ev=None, total_PV=None, domain=None, task_description=""):
    """optimize by the genetic algorithm"""
    if number_ev is None:
        number_ev = GLOBAL_STATE["env"].Nev
    else:
        GLOBAL_STATE["env"].set_N_ev(number_ev)
    if total_PV is None:
        total_PV = GLOBAL_STATE["env"].solar_power.sum()
        
    mask = np.ones((GLOBAL_STATE["env"].N_agent,))
    ## add codes for automatic generate the initial features
    avail_feat = GLOBAL_STATE["env"].get_feature_info()
    from prompt import get_feat_prompt

    p = get_feat_prompt.format(avail_feat=avail_feat, task_type="task_description")
    result = LLM.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": p},
        ],
        temperature=0.1,
    )
    result = json.loads(result.choices[0].message.content)
    select_feat = result["selected_features"]
    print("selected features: ", select_feat)

    if domain == "PV":
        task_specific_args = {"total_PV": total_PV, "features": select_feat}
    elif domain == "V2G":
        task_specific_args = {"Nev": number_ev, "features": select_feat}

    state_to_opt = {
        "env": GLOBAL_STATE["env"],
        "flow": GLOBAL_STATE["flow"],
        "mask": mask,
        "task_type": domain,
        "task_specific_args": task_specific_args,
    }

    with open("../cache/state_to_opt.pkl", "wb") as f:
        pickle.dump(state_to_opt, f)

    import subprocess

    result = subprocess.run(
        ["bash", "path to/run.sh"],
        capture_output=True,
        text=True,
        check=True
    )

    best_popu = np.load("path to alpha-evolve result.npy")
    if domain == "V2G":
        V2G_power, NEV_node, ramping, satisfied = GLOBAL_STATE["env"].simulate_v2g(
            best_popu
        )
    elif domain == "PV":
        GLOBAL_STATE["env"].set_pv_from_vector(best_popu)

    GLOBAL_STATE["power_demand"] = GLOBAL_STATE["env"].get_power_demand()
    GLOBAL_STATE["gen_power"] = get_generator_power(
        GLOBAL_STATE["flow"], GLOBAL_STATE["power_demand"]
    )
    V2G_storage = np.sum(V2G_power[V2G_power > 0])
    cost = calculate_grid_cost_source(GLOBAL_STATE["gen_power"]) / 4
    return {
        "V2G storage (MWh)": V2G_storage,
        "Operation cost ($)": np.sum(cost),
        "info": "V2G problem, alpha evolve optimized successfully!",
        "select_feat": select_feat,
    }


def main():
    # await server.run_stdio_async()  # 使用 STDIO 作为通信管道
    server.run(transport="http", host="127.0.0.1", port=13579)  # 使用 HTTP 作为通信管道


if __name__ == "__main__":
    main()
