import streamlit as st
from inteligentAgent import ask_question

# Title of the app
st.title("ChatOllama")

# Sidebar for user inputs
st.sidebar.header("Documents")
# user_name = st.sidebar.text_input("Enter your name:", "Guest")

# Start chat
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = [{"role": "ai", "content": "Hi! How can I help you today?"}]

# User input
user_input = st.chat_input(placeholder='Message Ollama')
if user_input:
    # Add user message to chat history
    st.session_state['chat_history'].append({'role': 'user', 'content': user_input})

    # Get response from Ollama
    response = ask_question(user_input)
    st.session_state['chat_history'].append({'role': 'ai', 'content': response})

for chat in st.session_state['chat_history']:
    if chat['role'] == 'user':
        with st.chat_message('user'):
            st.write(chat['content'])
    else:
        with st.chat_message('ai'):
            st.write(chat['content'])