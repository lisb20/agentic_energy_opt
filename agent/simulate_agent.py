from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient, load_mcp_tools
from langchain.messages import AIMessage, HumanMessage, ToolMessage

# 连接 MCP server（例如 http+runs or ws）
class simulate_agent():
    def __init__(self, allow_tools = ["set_PV_by_bbox","run_PV_simulation","load_PV_from_file","run_V2G_simulation","load_V2G_from_file",], llm = None, client=None, task_type="PV"):
        self.client = client
        self.allow_tools = allow_tools
        self.llm = llm
        self.task_type = task_type

    async def creat_sim_agent(self):
        self.tools = await self.client.get_tools()
        # self.tools = [t for t in self.tools if t.name in self.allow_tools]
        # self.tools = [t for t in self.tools if self.task_type in t.name]
        print("SIMAGENT: creating simulate agent with tools:")
        print([t.name for t in self.tools])
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt="You are an expert in urban energy system."
        )

    async def main(self, message):
        result = await self.agent.ainvoke(
            {"messages": [{"role": "user", "content": message}]}
        )
        result_str = ""
        tool_log = []
        for m in result["messages"]:
            if isinstance(m, AIMessage):
                result_str += m.content
            if isinstance(m, ToolMessage):
                tool_log.append("")
        return result, result_str, tool_log


if __name__ == "__main__":
    pass