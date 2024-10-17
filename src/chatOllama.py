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

# Sidebar for user inputs
st.sidebar.header("Documents")
# Allow users to upload documents
uploaded_files = st.sidebar.file_uploader("Upload documents", type=["pdf", "txt"], accept_multiple_files=True)

# Process uploaded documents and create a vector store
if uploaded_files:
    all_texts = []
    for uploaded_file in uploaded_files:
        all_texts.append(load_document(uploaded_file))

    # Split the texts into manageable chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    documents = text_splitter.split_text("\n".join(all_texts))
    # st.write(documents)

    # Create embeddings for the documents
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(documents, embeddings)

    st.success("Documents uploaded and processed!")

# Start chat
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = [{"role": "ai", "content": "Hi! How can I help you today?"}]

# User input
user_input = st.chat_input(placeholder='Message Ollama')
if user_input and 'vector_store' in locals():
    # Perform a similarity search in the vector store
    relevant_docs = vector_store.similarity_search(user_input)

    # Add user message to chat history
    st.session_state['chat_history'].append({'role': 'user', 'content': user_input})

    # Get response from Ollama
    # Create a context for the question
    context = "\n".join([doc.page_content for doc in relevant_docs])
    response = ask_ollama(f"{context}\n\nQuestion: {user_input}\nAnswer:")
    
    st.session_state['chat_history'].append({'role': 'ai', 'content': response})

for chat in st.session_state['chat_history']:
    if chat['role'] == 'user':
        with st.chat_message('user'):
            st.write(chat['content'])
    else:
        with st.chat_message('ai'):
            st.write(chat['content'])