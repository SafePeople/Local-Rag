'''
run this command in the terminal to start the app
streamlit run src/chatOllama.py
'''
import streamlit as st

from langchain_community.llms.ollama import Ollama
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate

from database import add_document, add_query, fetch_documents, fetch_queries, clear_table

import subprocess
import time
import requests

# # Function to start Ollama service
# def start_ollama():
#     # Run 'ollama serve' in the background
#     try:
#         # Start Ollama service using subprocess
#         subprocess.Popen(["ollama", "serve"])
#         print("Starting Ollama service...")
#     except Exception as e:
#         print(f"Failed to start Ollama: {e}")

# # Function to check if Ollama is running
# def is_ollama_running():
#     try:
#         # Try sending a request to Ollama's API to check if it's running
#         response = requests.get("http://localhost:11434/api/models")
#         if response.status_code == 200:
#             return True
#     except requests.exceptions.ConnectionError:
#         return False
#     return False

# # Start Ollama if it's not already running
# if not is_ollama_running():
#     start_ollama()
#     time.sleep(5)  # Wait a few seconds to ensure the service is up

# # Confirm Ollama is running
# if is_ollama_running():
#     print("Ollama is running and ready!")
# else:
#     print("Failed to start Ollama service.")


ollama_llm = Ollama(model='llama3.2')

if 'vectors' not in st.session_state:
    st.session_state['vectors'] = None # Initialize the vector store

def ask_ollama(user_input):
    return ollama_llm.invoke(user_input)

def load_document(file):
    # st.write(file)
    if file.type == 'application/pdf':
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    elif file.type == 'text/plain':
        text = file.read().decode("utf-8")
        return text


# Title of the app
st.title("Chat Ollama")

st.sidebar.title("Chat Ollama")
with st.sidebar.expander("Settings"):
    if st.button("Clear chat history"):
        clear_table("queries")
        st.success("Chat history cleared!")
    if st.button("Clear documents"):
        clear_table("documents")
        st.session_state['vectors'] = None # Reset the vectors when documents are cleared
        st.success("Documents cleared!")

# Sidebar for user inputs
st.sidebar.header("Documents")
# Allow users to upload documents
uploaded_files = st.sidebar.file_uploader("",type=["pdf", "txt"], accept_multiple_files=True, label_visibility='collapsed')

# Process uploaded documents and create a vector store
if uploaded_files:
    for uploaded_file in uploaded_files:
        for file_name, _ in fetch_documents():
            if uploaded_file.name == file_name:
                break
        else:
            add_document(file_name=uploaded_file.name, content=load_document(uploaded_file))
            st.sidebar.success("Documents uploaded and processed!")
            st.session_state['vectors'] = None # Reset the vectors when new documents are uploaded


# Create embeddings for the documents
documents = fetch_documents()
if documents and st.session_state['vectors'] is None:
    # Split the texts into manageable chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_doc = text_splitter.split_text("\n".join(text for _, text in documents))

    # st.write(split_doc)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    st.session_state['vectors'] = FAISS.from_texts(split_doc, embeddings)
    st.sidebar.success("Vectors created!")
else:
    st.session_state['vectors'] = None
    

if st.sidebar.button("show documents"):
    for file_name, content in fetch_documents():
        st.sidebar.write(f"Filename: {file_name}")
    
# Create the RAG chain
def rag_chain(vectors, prompt, llm):
    
    retriever = vectors.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    ragChain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return ragChain

# Define the prompt template
prompt = PromptTemplate(template="Answer the following question using given context and return with citations: {context}. Question: {question}", input_variables=["context", "question"])


# Start chat
if (fetch_queries() == []):
    add_query('StartChatWithOllama', "Hello! I'm Ollama. How can I help you today?")

# User input
user_input = st.chat_input(placeholder='Message Ollama')
if user_input:
    # Get response from Ollama
    # Create a context for the question
    # Perform a similarity search in the vector store
    if st.session_state['vectors'] is None:
        response = ask_ollama(f"{user_input}\nAnswer:")
    else:
        # Load llama3.2 model into the rag chain
        ragChain = rag_chain(vectors=st.session_state['vectors'], prompt=prompt, llm=ollama_llm)
        similar = st.session_state['vectors'].similarity_search_with_score(user_input)
        for doc in similar:
            if doc[1] < 1.7:
                response = ragChain.invoke(user_input)
                add_query(user_input, f'FROM TEXT: \n {response}')
                break
            else:
                response = ask_ollama(f"{user_input}\nAnswer:")
                add_query(user_input, response)
                break

# Display chat history
for query, response, timestamp in reversed(fetch_queries()):

    if query == 'StartChatWithOllama':
        pass
    else:
        with st.chat_message('user'):
            st.write(query)

    with st.chat_message('ai'):
        st.write(response)