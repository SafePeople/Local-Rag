from langchain_community.llms.ollama import Ollama
from langchain.memory import ConversationSummaryMemory
from langchain.chains.llm import LLMChain
from transformers import pipeline
from langchain_core.tools import tool
from langchain.agents import initialize_agent
from langchain.agents import AgentType

# ANSI escape codes for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'
    
# Initialize the Ollama model and memory from langchain
ollama_llm = Ollama(model="llama3.1")
memory = ConversationSummaryMemory(llm=ollama_llm)
# prompt = PromptTemplate(
#     input_variables = ["query"],
#     template = "Here is the answer to your question: {query}"
# )
# ollama_chain = LLMChain(llm=ollama_llm, prompt=prompt)


# Setup tools
@tool
def query_model(query: str) -> str:
    """Query the Ollama model with a given prompt when needed."""
    return ollama_llm.invoke(query)

@tool
def exit_agent(reason: str):
    """Use this tool to exit the agent when the query has been answered."""
    return f"Exiting agent: {reason}"

# Define tools for the agent
tools = [
    query_model,
    exit_agent
]

# Initialize the agent with ollama and tools
agent = initialize_agent(
    tools,
    ollama_llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

while True:
    user_input = input(YELLOW + "Ask your question (or type 'quit' to exit): " + RESET_COLOR)
    if user_input.lower() == 'quit':
        break
    # Run the agent on a task
    respones = agent.run(user_input)
    print(respones)