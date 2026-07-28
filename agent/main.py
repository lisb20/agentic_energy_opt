from simulate_agent import simulate_agent
from task_parsing import task_parser, task_assignment, describe_agent
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient, load_mcp_tools
from fastmcp import Client as FastMCPClient
import asyncio
import re
import os
import json
import argparse

# Local MCP traffic must not be sent through an HTTP proxy.
for proxy_var in ("NO_PROXY", "no_proxy"):
    proxy_exclusions = os.environ.get(proxy_var, "")
    entries = [value for value in proxy_exclusions.split(",") if value]
    for host in ("127.0.0.1", "localhost"):
        if host not in entries:
            entries.append(host)
    os.environ[proxy_var] = ",".join(entries)

def build_previous_result(logs):
    summary = "Here is the summary of the previous excecution results:\n"
    for i, log in enumerate(logs):
        summary += f"STEP {i+1}: {log}\n"
    return summary

import httpx
timeout = httpx.Timeout(
    connect=60.0,
    read=1200.0,     # 👈 关键
    write=60.0,
    pool=60.0
)
client = httpx.AsyncClient(timeout=timeout)
llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "Qwen/Qwen3-8B"),
                    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                    http_async_client=client,
                    timeout=(60, 1200),
                    stream=False)

MCP_PORT = int(os.getenv("MCP_PORT", "13579"))
MCP_URL = f"http://127.0.0.1:{MCP_PORT}/mcp"

client = MultiServerMCPClient(
            {
                "sim": {
                    "transport": "http",
                    "url": MCP_URL,
                    "timeout": 12000,
                    "sse_read_timeout":12000,
                }
            }
        )

manual_client = FastMCPClient(MCP_URL)

async def init(task_type):
    async with manual_client:
        await manual_client.call_tool("init", {"TASK_TYPE": task_type})

if __name__ == "__main__":
    import time
    parser = argparse.ArgumentParser(description="Run the energy planning agent")
    parser.add_argument("message", nargs="?", help="Task to execute")
    parser.add_argument("--message-file", help="Read the task from a UTF-8 text file")
    args = parser.parse_args()
    time_now_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    if args.message_file:
        with open(args.message_file, "r", encoding="utf-8") as f:
            message = f.read().strip()
    elif args.message:
        message = args.message
    else:
        parser.error("provide a message or --message-file")
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
    
    os.makedirs("./logs", exist_ok=True)
    with open(f"./logs/log_{time_now_str}.json", "w") as f:
        json.dump(log, f, indent=4)
    
    # run agents
    result, result_str, tool_log = asyncio.run(sim.main(tasks))
                
    print("result: ", result)
    
