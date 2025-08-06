import streamlit as st
import os
import time
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv('../.env')
GROQ_API = os.getenv('GROQ_API')

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API)

# System prompt
system_prompt = """You're a comedy guy and your name is Farsi, an AI created by Mr. Aryan Bhattarai. Whatever user asks you will reply in very polite way"""

# Page configuration
st.set_page_config(
    page_title="Farsi AI Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state for conversation memory
if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = [{
        'role': 'system',
        'content': system_prompt
    }]

# Initialize messages for display (without system message)
if "messages" not in st.session_state:
    st.session_state.messages = []

# App title and description
st.title("🤖 Farsi AI Chatbot")
st.markdown("*A polite comedy AI created by Mr. Aryan Bhattarai*")
st.divider()

# Chat container with custom styling
chat_container = st.container()

with chat_container:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Input section at the bottom
with st.container():
    col1, col2 = st.columns([6, 1])
    
    with col1:
        user_input = st.text_input(
            "Type your message here...",
            key="user_input",
            placeholder="Ask anything to Farsi!",
            label_visibility="collapsed"
        )
    
    with col2:
        send_button = st.button("Send", type="primary", use_container_width=True)

# Handle message sending
if send_button and user_input.strip():
    # Add user message to session state
    user_message = {"role": "user", "content": user_input}
    st.session_state.messages.append(user_message)
    st.session_state.conversation_memory.append(user_message)
    
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Generate AI response
    with st.chat_message("assistant"):
        with st.spinner("Farsi is thinking..."):
            try:
                response = groq_client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=st.session_state.conversation_memory,
                    max_tokens=2000,
                    temperature=0
                )
                
                ai_response = response.choices[0].message.content
                st.markdown(ai_response)
                
                # Add AI response to session state
                ai_message = {"role": "assistant", "content": ai_response}
                st.session_state.messages.append(ai_message)
                st.session_state.conversation_memory.append(ai_message)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.error("Please check your API key and internet connection.")
    
    # Clear input and rerun to update the interface
    st.rerun()


# Sidebar with additional features
with st.sidebar:
    st.header("Chat Controls")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.conversation_memory = [{
            'role': 'system',
            'content': system_prompt
        }]
        st.rerun()
    
    st.divider()
    
    st.header("Chat Statistics")
    total_messages = len(st.session_state.messages)
    user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
    ai_messages = len([m for m in st.session_state.messages if m["role"] == "assistant"])
    
    st.metric("Total Messages", total_messages)
    st.metric("Your Messages", user_messages)
    st.metric("Farsi's Messages", ai_messages)
    
    st.divider()
    
    st.header("About")
    st.info("Farsi is a polite comedy AI assistant created by Mr. Aryan Bhattarai. Feel free to ask anything!")

# Custom CSS for better styling
st.markdown("""
<style>
    .stTextInput > div > div > input {
        border-radius: 20px;
    }
    
    .stButton > button {
        border-radius: 20px;
        height: 55px;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    .user-message {
        background-color: #007bff;
        color: white;
        margin-left: 20%;
    }
    
    .assistant-message {
        background-color: #f8f9fa;
        margin-right: 20%;
    }
</style>
""", unsafe_allow_html=True)
