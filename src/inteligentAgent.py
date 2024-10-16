from langchain_community.llms.ollama import Ollama
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader, JSONLoader, CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from rag_agent import load_document, create_vectors
import faiss
import os

# ANSI escape codes for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

# Load llama3.2 model
ollama_llm = Ollama(model='llama3.2')
documents = []

def generate_text(prompt: str) -> str:
    '''Generate text using the llama3.2 model'''
    response = ollama_llm.invoke(prompt)
    return response

def ask_question(prompt: str):
    '''Ask a question to the llama3.2 model'''
    response = generate_text(prompt)
    return response

def prompt_docs():
    user_input = ""
    while user_input != 'q':
        user_input = input("\nenter docs, or 'q' to exit\n")

        if os.path.exists(user_input):
            doc = load_document(user_input)
            documents.append(doc)
            print(f"{user_input} was successfully added")

def prepare_embeddings():
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    embeding_model_dimentions = 384
    index = faiss.IndexFlatL2(embeding_model_dimentions)
    storage = FAISS( embeddings, index, InMemoryDocstore(), {})

    prompt_docs()
    create_vectors(documents, storage, embedding_model)

    # Load Local saved data
    if os.path.exists("vectorstore"):
        storage = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)

    # Save storage for faster loading next time
    storage.save_local("vectorstore")
    return storage

def main():
    while True:
        query = input(NEON_GREEN + 'Enter "q" to quit or enter your prompt: ' + RESET_COLOR)
        if query == 'q':
            break
        # Run the agent
        response = ask_question(query)
        print(f"{CYAN} Response: \n {response} {RESET_COLOR} \n")

if __name__ == '__main__':
    main()