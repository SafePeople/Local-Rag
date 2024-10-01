from langchain_community.llms.ollama import Ollama
from langchain.memory import ConversationBufferMemory
from langchain_core.tools import tool
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, TextLoader, JSONLoader, CSVLoader
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

# ANSI escape codes for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

def load_documents(file_path):
    '''
    Load documents from a file

    Args:
    file_path: (str)

    Returns:
    List: list of documents
    '''
    _, ext = os.path.splitext(file_path)
    if ext == '.pdf':
        loader = PyPDFLoader(file_path)
    elif ext == '.txt':
        loader = TextLoader(file_path)
    elif ext == '.json':
        loader = JSONLoader(file_path)
    elif ext == '.csv':
        loader = CSVLoader(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    
    documents = loader.load()
    return documents

# This function will split uploaded documents into chunks for better processing
def split_document(documents):
    '''
    Split documents into chunks

    Args:
    documents: (list)

    Returns:
    List: list of split documents
    '''

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(documents)
    return split_docs

def create_vectors(documents):
    '''
    Create vectors for the documents

    Args:
    documents: (list)

    Returns:
    FAISS: FIASS Vector
    '''
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-minilm-L6-v2")
    split_docs = split_document(documents)
    vectors = FAISS.from_documents(split_docs, embeddings)
    return vectors

def rag_chain(vectors, prompt, llm):
    
    retriever = vectors.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    ragChain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return ragChain

# Load documents and create vectors
file_path = "src/data/paper.pdf"
documents = load_documents(file_path)
vectors = create_vectors(documents)

# Define the prompt template
prompt = PromptTemplate(template="Answer the following question based on the given context: {context}. Question: {question}", input_variables=["context", "question"])

# Initialize the Ollama model and memory from langchain
ollama_llm = Ollama(model="llama3.1")
memory = ConversationBufferMemory(llm=ollama_llm)
ragChain = rag_chain(vectors, prompt, ollama_llm)

# Setup tools
@tool
def query_model(query: str) -> str:
    """Query the Ollama model with imported documents using the given prompt."""
    response = ragChain.invoke(query)
    if "final" in response:
        return exit_agent("Question has been answered.")
    else:
        return response

@tool
def exit_agent(reason: str):
    """Use this tool to exit the agent when the question has been answered."""
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
    verbose=True,
    
)

while True:
    user_input = input(YELLOW + "Ask your question (or type 'quit' to exit): " + RESET_COLOR)
    if user_input.lower() == 'quit':
        break
    # Run the agent on a task
    agent.handle_parsing_errors = True
    respones = agent.run(user_input)
    print(respones)