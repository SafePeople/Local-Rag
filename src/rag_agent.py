from langchain_community.llms.ollama import Ollama
from langchain.memory import ConversationBufferMemory
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain_community.document_loaders import TextLoader, PyPDFLoader, JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

def load_document(file):
    if file.endswith(".txt"):
        loader = TextLoader(file, encoding='UTF-8')
    elif file.endswith(".pdf"):
        loader = PyPDFLoader(file)
    elif file.endswith(".json"):
        loader = JSONLoader(file, jq_schema='.', text_content=False)
    else:
        raise Exception("Unknown file extension")
    document = loader.load()
    return document
    

def get_vector_storage():
    cwd = os.getcwd()
    file_path = 'src/data'

    documents = []

    for file in os.listdir(os.path.join(cwd, file_path)):
        doc = load_document(os.path.join(cwd, file_path, file))
        documents.extend(doc)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    all_splits = text_splitter.split_documents(documents)

    local_embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=local_embeddings)

    return vectorstore


# Initialize the Ollama model and memory from langchain
ollama_llm = Ollama(model="llama3.1")
memory = ConversationBufferMemory(memory_key="chat_history")

# Setup tools
@tool
def query_model(query: str) -> str:
    """Query the Ollama model with a given prompt."""
    return ollama_llm.invoke(query)

@tool
def search_storage(query: str) -> str:
    """Search the vector storage for the most relevant document."""
    storage = get_vector_storage()
    return storage.similarity_search(query)

@tool
def exit_agent(reason: str):
    """Use this tool to exit the agent when the task is complete."""
    return f"Exiting agent: {reason}"

# Define tools for the agent
tools = [
    search_storage,
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

# Run the agent on a task
respones = agent.run("What is the name of the ragtag team in CS 489R, use the search_storage tool to get additional info the feed that to the model")
print(respones)