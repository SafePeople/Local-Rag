# Import relevant functionality


from langchain_community.tools import tool
from langchain_community.chat_models import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import render_text_description
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent

#### Make function to interact with LLM

@tool
def add_two_tool(input: int) -> int:
    """Applies a magic function to an input."""
    return input + 2


# Create the agent
memory = MemorySaver()                                                  
model = ChatOllama(model="llama3.1")
tools = [add_two_tool]

# a very special prompt embodying the essence of ReAct
prompt = hub.pull("hwchase17/react-json")
prompt = prompt.partial(
    tools=render_text_description(tools),
    tool_names=", ".join([t.name for t in tools]),
)

agent = create_react_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True, verbose=False, format="json")


# Use the agent
while True:
    user_query = input("Enter your query (or 'quit' to exit): ")
    if user_query.lower() == 'quit':
        break

    response = agent_executor.invoke({"input": user_query})
    print(f"Response: {response}")