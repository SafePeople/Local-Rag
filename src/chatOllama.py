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
        st.success("Documents cleared!")

# Sidebar for user inputs
st.sidebar.header("Documents")
# Allow users to upload documents
uploaded_files = st.sidebar.file_uploader("",type=["pdf", "txt"], accept_multiple_files=True, label_visibility='collapsed')

# Process uploaded documents and create a vector store
if uploaded_files:
    all_texts = []
    counter = 0
    for uploaded_file in uploaded_files:
        all_texts.append(load_document(uploaded_file))

        # Split the texts into manageable chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        documents = text_splitter.split_text("\n".join(all_texts))
        
        add_document(file_name=uploaded_file.name, content=documents)
        counter += 1

    st.sidebar.success("Documents uploaded and processed!")
    st.sidebar.write(counter)

    if fetch_documents():
        # Create embeddings for the documents
        documents = fetch_documents()
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_texts([chunk for doc in all_texts for chunk in doc], embeddings)

if st.sidebar.button("show documents"):
    for file_name, content in fetch_documents():
        st.sidebar.write(f"Filename: {file_name} - Content: {content[:100]}...")
    

# Start chat
if (fetch_queries() == []):
    add_query('StartChatWithOllama', "Hello! I'm Ollama. How can I help you today?")

# User input
user_input = st.chat_input(placeholder='Message Ollama')
if user_input:
    # Get response from Ollama
    # Create a context for the question
    try:
         # Perform a similarity search in the vector store
        relevant_docs = vector_store.similarity_search(user_input)

        context = "\n".join([doc.page_content for doc in relevant_docs])
        response = ask_ollama(f"{context}\n\nQuestion: {user_input}\nAnswer:")
    except Exception as e:
        response = ask_ollama(f"{user_input}\nAnswer:")

    add_query(user_input, response)

# Display chat history
for query, response, timestamp in reversed(fetch_queries()):

    if query == 'StartChatWithOllama':
        pass
    else:
        with st.chat_message('user'):
            st.write(query)

    with st.chat_message('ai'):
        st.write(response)