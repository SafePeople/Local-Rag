# For Ollama and LangChain Chain
from langchain_community.llms.ollama import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate

# For file vector storage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader, JSONLoader, CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss
import os

# ANSI escape codes for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

def load_document():
    '''
    Load documents from a file

    Args:
    file_path: (str)

    Returns:
    List: list of documents
    '''

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
            _, ext = os.path.splitext(user_input)
            if ext == '.pdf':
                loader = PyPDFLoader(user_input)
            elif ext == '.txt':
                loader = TextLoader(user_input)
            elif ext == '.json':
                loader = JSONLoader(user_input)
            elif ext == '.csv':
                loader = CSVLoader(user_input)
            else:
                raise ValueError(f"Unsupported file format: {ext}")
            
            documents.append(loader.load())
            print(f"{NEON_GREEN} {user_input} was successfully added {RESET_COLOR}")
    return documents

def load_document_st(file_path):
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

def create_vectors(documents, storage: FAISS, embeddings):
    '''
    Create vectors for the documents

    Args:
    documents: (list)
    storage: storage vector to add embeddings to

    Returns:
    None
    '''
    try:
        for document in documents:
            split_docs = split_document(document)
            vectors = FAISS.from_documents(split_docs, embeddings)
            storage.merge_from(vectors)
    except Exception as e:
        print(f"Error: {e}")

def rag_chain(vectors, prompt, llm):
    
    retriever = vectors.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    ragChain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return ragChain

def generate_text_from_documents(prompt: str) -> str:
    '''Generate text using the llama3.2 model'''
    response = ragChain.invoke(prompt)
    return response

def ask_ollama(prompt: str):
    '''Ask a question to the llama3.2 model'''
    response = ollama_llm.invoke(prompt)
    return response

### Load documents and create vectors
# Create Vector Store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
if os.path.exists("vectorstore"):
    storage = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
else:
    index = faiss.IndexFlatL2(384) # 384 is the dimension of the MiniLM model
    storage = FAISS(embeddings, index, InMemoryDocstore(), {})


documents = load_document()
# combine new files to storage
create_vectors(documents, storage, embeddings)

# Save storage for faster loading next time
storage.save_local("vectorstore")

# Define the prompt template
prompt = PromptTemplate(template="Answer the following question using given context: {context}. Otherwise, just use the normal Ollama model. Question: {question}", input_variables=["context", "question"])

# # Load llama3.2 model
ollama_llm = Ollama(model='llama3.2')
ragChain = rag_chain(vectors=storage, prompt=prompt, llm=ollama_llm)

def main():
    while True:
        query = input(NEON_GREEN + 'Enter "q" to quit or enter your prompt: ' + RESET_COLOR)
        if query == 'q':
            break
        # Run the agent
        # response = ask_ollama(query) # Ollama directly
        response = generate_text_from_documents(query)
        print(f"{CYAN} Response: \n {response} {RESET_COLOR} \n")

if __name__ == '__main__':
    main()