from simulate_agent import simulate_agent
from task_parsing import task_parser, task_assignment, describe_agent
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient, load_mcp_tools
from fastmcp import Client as FastMCPClient
import asyncio
import re
import os
import json

import httpx
timeout = httpx.Timeout(
    connect=60.0,
    read=1200.0,     # 👈 关键
    write=60.0,
    pool=60.0
)
client = httpx.AsyncClient(timeout=timeout)
llm = ChatOpenAI(model="Qwen/Qwen3-8B",
                    # api_key="sk-xxx",
                    base_url="https://api.siliconflow.cn/v1",
                    http_async_client=client,
                    timeout=(60, 1200),
                    stream=False)

client = MultiServerMCPClient(
            {
                "sim": {
                    "transport": "http",
                    "url": "http://127.0.0.1:13579/mcp",
                    "timeout": 12000,
                    "sse_read_timeout":12000,
                }
            }
        )

manual_client = FastMCPClient("http://127.0.0.1:13579/mcp")

async def init(task_type):
    async with manual_client:
        await manual_client.call_tool("init", {"TASK_TYPE": task_type})

if __name__ == "__main__":
    import time
    time_now_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    message = "Load a PV plan checkpoints from \"xxx/test_percentage.npy\". Use the checkpoint as initial plan, and optimize the deployment plan by alpha-evolve. The target PV is 60000 MW."
    # message = "Load The V2G target curve from checkpoints from \"path to ckpt\". Let's simulate 200000 EVs and see how it performs."
    log = [
        {
            "role": "human",
            "content": message
        }
    ]
    prompt, tasks = task_parser(llm, message)
    print("task_decomposition: ")
    print(tasks)
    # build a log
    log += [
        {
            "role": "assistant",
            "content": prompt
        },
        {
            "role": "assistant",
            "content": json.dumps(tasks)
        }
    ]

    # init the MCP  
    print("Initialize MCP server")
    TASK_TYPE = tasks["Domain"]
    asyncio.run(init(TASK_TYPE))

    sim = simulate_agent(llm=llm, client=client, task_type=TASK_TYPE)
    asyncio.run(sim.creat_sim_agent())
    
    with open(f"./logs/log_{time_now_str}.json", "w") as f:
        json.dump(log, f, indent=4)
    
    # run agents
    result, result_str, tool_log = asyncio.run(sim.main(tasks))
                
    print("result: ", result)
    