import json

def task_parser(llm, message):
    task_parsing_prompt = f"""
You are a task parsing agent.

Your job is to extract structured information from the user request.
Do NOT explain.
Do NOT add extra fields.
Do NOT infer anything that is not explicitly stated.

Output ONLY a JSON object in the following schema:

{{
  "task_type": "sim" or "opt",
  "Domain": "PV" or "V2G", # the domain of the task, for optimization on photo-voltaic systems or vehicle-to-grid coordination.
  "parameters": {{
    "checkpoint_path": string or null,
    "spatial_bbox": [lon_min, lat_min, lon_max, lat_max] or null,
    "pv_target_capacity_mw": number or null,
    "number_ev": number or null,
    "city": string or null,
    "optimization_method": GA or alpha-evolve or null,
  }},
  "user_intention": string, # a description of the user's intention, including other information that is not captured in the parameters
}}

Rules:
- task_type = "opt" if the request asks to optimize or deploy PV.
- task_type = "sim" if the request asks to simulate or evaluate a given plan.
- If a parameter is not mentioned, set it to null or empty list. Only keep the parameters that should be used in the current step.
- Use numeric values whenever possible.

User request:
{message}
"""


    messages = [
            (
                "system",
                "You are an expert in urban planning and energy management. You are going to solve the following problem.",
            ),
            ("human", task_parsing_prompt),
        ]
    result = llm.invoke(messages)
    content = result.content
    content = json.loads(content)
    return task_parsing_prompt, content


def describe_agent(opt_tools,sim_tools):
    opt_tools = [x.name for x in opt_tools]
    opt_tools_string = ", ".join(opt_tools)
    sim_tools = [x.name for x in sim_tools]
    sim_tools_string = ", ".join(sim_tools)
    agents = f"""
Here we introduce the available agents and their capabilities:
SimAgent: The SimAgent is responsible for simulating the operation of the power grid and evaluating the impact of different deployment plans. The tools are: [{sim_tools_string}].
OptAgent: The OptAgent is responsible for optimizing the deployment plan based on the simulation results. The tools are: [{opt_tools_string}].
""" 
    return agents