# THE MCP server
from fastmcp import FastMCP
import asyncio
import sys
import os
from pathlib import Path

RELEASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("DATA_DIR", RELEASE_DIR / "data")).expanduser().resolve()
SIMULATOR_DIR = RELEASE_DIR / "simulator"
MCP_DIR = Path(__file__).resolve().parent
EVOLVE_DIR = RELEASE_DIR / "evolve"
sys.path.insert(0, str(SIMULATOR_DIR))
sys.path.insert(0, str(MCP_DIR))
sys.path.insert(0, str(EVOLVE_DIR))
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
    api_key=os.getenv("OPENAI_API_KEY", "not-set"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    timeout=60.0,
    max_retries=0,
)

server = FastMCP("pv_sim_server")
GLOBAL_STATE = {}


@server.tool()
def init(TASK_TYPE):
    global GLOBAL_STATE
    """initialize the server"""
    print("CALLING <init>", "TASK_TYPE", TASK_TYPE)
    if TASK_TYPE == "PV":
        with (DATA_DIR / "flow_config.yaml").open("r") as f:
            config = yaml.safe_load(f)
        flow_data = DATA_DIR / "topology.json"

        env = myenv(EV_driving_power=None, EV_location=None)
        GLOBAL_STATE = {
            "env": env,
            "flow": Flow(config, str(flow_data)),
            "gen_power": np.zeros((25, 96)),
            "power_demand": env.get_power_demand(),
        }
    elif TASK_TYPE == "V2G":
        with (DATA_DIR / "flow_config.yaml").open("r") as f:
            config = yaml.safe_load(f)
        flow_data = DATA_DIR / "topology.json"

        driving_power = os.getenv("V2G_DRIVING_POWER")
        vehicle_locations = os.getenv("V2G_LOCATION")
        if not driving_power or not vehicle_locations:
            raise RuntimeError(
                "V2G_DRIVING_POWER and V2G_LOCATION must point to the private mobility arrays"
            )

        env = myenv(
            topo_path=flow_data,
            power_path=DATA_DIR / "power_demand.npy",
            EV_driving_power=driving_power,
            EV_location=vehicle_locations,
            Nev=300000,
        )
        GLOBAL_STATE = {
            "env": env,
            "flow": Flow(config, str(flow_data)),
            "gen_power": np.zeros((25, 96)),
            "power_demand": env.get_power_demand(),
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
    return f"V2G set successfully from {fp}!"


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
    if domain not in {"PV", "V2G"}:
        raise ValueError("domain must be 'PV' or 'V2G'")
    if domain == "V2G" and number_ev is None:
        number_ev = GLOBAL_STATE["env"].Nev
    elif domain == "V2G":
        GLOBAL_STATE["env"].set_N_ev(number_ev)
    if total_PV is None:
        total_PV = GLOBAL_STATE["env"].solar_power.sum()
        
    mask = np.ones((GLOBAL_STATE["env"].N_agent,))
    ## add codes for automatic generate the initial features
    avail_feat = GLOBAL_STATE["env"].get_feature_info()
    from prompt import get_feat_prompt

    p = get_feat_prompt.format(avail_feat=avail_feat, task_type=task_description or domain)
    default_features = {
        "PV": ["lon", "lat", "demand_profile", "solar_profile", "rooftop_area", "closest_PP_id"],
        "V2G": ["lon", "lat", "demand_profile", "target_pattern", "N_ev_node", "closest_PP_id"],
    }
    try:
        result = LLM.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", MODEL),
            messages=[{"role": "user", "content": p}],
            temperature=0.1,
            max_tokens=1024,
            extra_body={"enable_thinking": False},
        )
        content = result.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0]
        select_feat = json.loads(content)["selected_features"]
    except Exception as exc:
        print(f"Feature selection failed, using defaults: {exc}")
        select_feat = default_features[domain]
    print("selected features: ", select_feat)

    if domain == "PV":
        task_specific_args = {"target_pv": total_PV, "features": select_feat}
    elif domain == "V2G":
        task_specific_args = {"Nev": number_ev, "features": select_feat}

    state_to_opt = {
        "env": GLOBAL_STATE["env"],
        "flow": GLOBAL_STATE["flow"],
        "mask": mask,
        "task_type": domain,
        "task_specific_args": task_specific_args,
    }

    cache_dir = RELEASE_DIR / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    with (cache_dir / "state_to_opt.pkl").open("wb") as f:
        pickle.dump(state_to_opt, f)

    import subprocess

    run_env = os.environ.copy()
    run_env["DOMAIN"] = domain
    result = subprocess.run(
        ["bash", str(RELEASE_DIR / "evolve" / "run.sh")],
        cwd=RELEASE_DIR / "evolve",
        env=run_env,
        capture_output=True,
        text=True,
        check=True
    )

    best_program = RELEASE_DIR / "output" / "best" / "best_program.py"
    if not best_program.exists():
        raise FileNotFoundError(f"OpenEvolve did not create {best_program}")
    sys.path.insert(0, str(RELEASE_DIR / "openevolve"))
    from evaluator import evaluate
    evaluation = evaluate(str(best_program))
    if "error" in evaluation.metrics:
        raise RuntimeError(f"Best program evaluation failed: {evaluation.artifacts}")
    best_popu = np.load(RELEASE_DIR / "evolve" / "plan.npy")
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
    V2G_storage = np.sum(V2G_power[V2G_power > 0]) / 1e3 if domain == "V2G" else 0.0
    cost = calculate_grid_cost_source(GLOBAL_STATE["gen_power"]) / 4
    return {
        "V2G storage (MWh)": V2G_storage,
        "Operation cost ($)": np.sum(cost),
        "info": f"{domain} problem optimized successfully by alpha-evolve",
        "select_feat": select_feat,
    }


def main():
    # await server.run_stdio_async()  # 使用 STDIO 作为通信管道
    port = int(os.getenv("MCP_PORT", "13579"))
    server.run(transport="http", host="127.0.0.1", port=port)  # 使用 HTTP 作为通信管道


if __name__ == "__main__":
    main()
