import streamlit as st

# Title of the app
st.title("ChatOllama")

# Sidebar for user inputs
st.sidebar.header("Documents")
user_name = st.sidebar.text_input("Enter your name:", "Guest")
age = st.sidebar.slider("Select your age:", 0, 100, 25)

# Main content
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def chat_ollama(prompt: str):
    '''Generate text using the llama3.2 model'''
    st.write(f'{prompt}')
    response = "Hello, I am ChatOllama. How can I help you?"
    return response

st.subheader('Chat History')
chat_history = st.session_state['chat_history']
for chat in chat_history:
    st.text_area(label='', value=chat, height=100, max_chars=None, key=f'chat_{chat_history.index(chat)}')

# User input
with st.chat_message('ai'):
    st.write(st.chat_input(placeholder='Message Ollama'))