from langchain_community.llms.ollama import Ollama
from langchain.memory import ConversationBufferMemory
from langchain_core.tools import tool
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents import AgentType
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader, JSONLoader, CSVLoader
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase
import faiss
import os


# ANSI escape codes for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

def load_document(file_path):
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
def split_document(document):
    '''
    Split documents into chunks

    Args:
    documents: (list)

    Returns:
    List: list of split documents
    '''

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(document)
    return split_docs

def create_vectors(documents, storage: FAISS):
    '''
    Create vectors for the documents

    Args:
    documents: (list)
    storage: storage vector to add embeddings to

    Returns:
    None
    '''
    embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    for document in documents:
        split_docs = split_document(document)
        vectors = FAISS.from_documents(split_docs, embeddings)
        storage.merge_from(vectors)
    return 

def rag_chain(vectors, prompt, llm):
    
    retriever = vectors.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    ragChain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return ragChain

### Load documents and create vectors
# Create Vector Store
embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embeding_model_dimentions = 384
index = faiss.IndexFlatL2(embeding_model_dimentions)
storage = FAISS( embeddings, index, InMemoryDocstore(), {})

# Load Local saved data
if os.path.exists("vectorstore"):
    storage = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)

# Loop for adding new docs
documents = []
user_input = ""
while user_input != "q":
    print("\nNote: if doc is already in storage or was added previously it is still in storage")
    print("Would be helpful if we could check if doc is in storage already or not, but idk how")
    print("You can reset the stored vector storage by deleting the vectorstore directory\n")
    print("You can enter files to be loaded here, or type 'q' to continue...")
    user_input = input("Enter file path: ")

    if os.path.exists(user_input):
        doc = load_document(user_input)
        documents.append(doc)
        print(f"{user_input} was successfully added")

# combine new files to storage
create_vectors(documents, storage)

# Save storage for faster loading next time
storage.save_local("vectorstore")
###

# Define the prompt template
prompt = PromptTemplate(template="Answer the following question based on the given context: {context}. Question: {question}", input_variables=["context", "question"])

# Initialize the Ollama model and memory from langchain
ollama_llm = Ollama(model="llama3.1")
memory = ConversationBufferMemory(llm=ollama_llm)
ragChain = rag_chain(storage, prompt, ollama_llm)

# Initialize database connection and get sql tools
''' almost working, errors in logic a bit but able to connect to mysql
    I think there are other issues causing looping logic so might try 
    looking into updating depreciated stuff but idk. might see if this 
    works better with create_sql_agent thingy'''
protocol = "mysql"
user = "root"
password = "*Supernova54*"
hosts = f"{user}:{password}@localhost"
port = "3306"
database_name = "world"

db = SQLDatabase.from_uri(f"{protocol}://{hosts}:{port}/{database_name}")
toolkit_tools = SQLDatabaseToolkit(db=db, llm=ollama_llm).get_tools()


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
    exit_agent,
    # *toolkit_tools
]

# Initialize the agent with ollama and tools
agent = create_sql_agent(
    extra_tools=tools,
    llm=ollama_llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    db=db,
)

while True:
    user_input = input(YELLOW + "Ask your question (or type 'quit' to exit): " + RESET_COLOR)
    if user_input.lower() == 'quit':
        break
    # Run the agent on a task
    agent.handle_parsing_errors = True
    respones = agent.invoke(user_input)
    print(respones)