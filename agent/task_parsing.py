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
    content = result.content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0]
    content = json.loads(content)
    return task_parsing_prompt, content


def task_assignment(llm, task, agents):
  
    prompt =f"""
We have the following parsed task:

<task> """\
+ task +\
"""</task>

Available agents and their capabilities:

<agents>"""\
+ agents +\
"""</agents>

Please generate an execution plan.

Each step must include:
- agent
- intent (high-level, capability-aligned)
- context (only task-relevant fields)

Output format:
{
  "execution_plan": [
    {  
      "step": "step_number",
      "agent": "agent_name",
      "intent": "intent",
      "context": "context"
    }
  ]
}
"""

    system_prompt = """You are an Orchestrator for a multi-agent system.

Your responsibility is to generate an execution plan at the agent-intent level.

Rules:
- You MUST NOT refer to any tool names or tool parameters.
- You MUST NOT describe how an agent implements an intent.
- You MUST NOT call the same agent in a row.
- You MUST ONLY use the listed agents and their declared capabilities.
- Your output MUST be valid JSON.
- You MUST decide the execution order explicitly.
"""


    messages = [
            (
                "system",
                system_prompt,
            ),
            ("human", prompt),
        ]
    print(prompt)
    response = llm.invoke(messages)
    print(response.content)
    plan = json.loads(response.content)
    
    return prompt, plan

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
