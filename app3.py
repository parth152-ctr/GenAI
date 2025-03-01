import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)

# Custom CSS styling
st.markdown("""
<style>
    body {
        font-family: 'Arial', sans-serif;
        background-color: #121212;
        color: #ffffff;
    }
    .main {
        background-color: #1a1a1a;
        color: #ffffff;
        padding: 20px;
        border-radius: 10px;
    }
    .sidebar .sidebar-content {
        background-color: #2d2d2d;
        padding: 15px;
        border-radius: 10px;
        display: block !important;
    }
    .stTextInput textarea, .stSelectbox div[data-baseweb="select"] {
        color: #ffffff !important;
        background-color: #3d3d3d !important;
        border-radius: 5px;
    }
    .stSelectbox svg {
        fill: white !important;
    }
    .stSelectbox option, div[role="listbox"] div {
        background-color: #2d2d2d !important;
        color: white !important;
    }
    .sidebar-links {
        text-align: center;
        margin-top: 20px;
    }
    .sidebar-links a {
        display: block;
        color: #ffffff;
        text-decoration: none;
        padding: 8px 0;
        font-weight: bold;
    }
    .sidebar-links a:hover {
        color: #1db954;
        text-decoration: underline;
    }
    .sidebar-img {
        display: flex;
        justify-content: center;
        margin-top: 15px;
    }
    .stButton>button {
        background-color: #1db954 !important;
        color: #ffffff !important;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #17a147 !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧠 Generic AI Text Generation")
st.caption("🚀 Your AI Pair Programmer with Debugging Superpowers")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    st.subheader("👥 Team: MechaMinds")
    st.markdown("""
    - 1️⃣ Rikin Pithadia
    - 2️⃣ Darji Kunj
    - 3️⃣ Parth Prajapati
    - 4️⃣ Yagnit Baraiya
    """)
    st.divider()
    selected_model = st.selectbox(
        "Choose Model",
        ["deepseek-r1:1.5b", "deepseek-r1:3b"],
        index=0
    )
    st.divider()
    st.markdown("### Model Capabilities")
    st.markdown("""
    - 🐍 Python Expert
    - 🐞 Debugging Assistant
    - 📝 Code Documentation
    - 💡 Solution Design
    """)
    st.divider()
    
    # Sidebar Image
    st.markdown('<div class="Genai.png img"><img src="Genai.png img" width="150"></div>', unsafe_allow_html=True)
    
    # External links
    st.markdown("""
    <div class="sidebar-links">
        <a href="https://github.com/parth152-ctr/GenAI  " target="_blank">🔗 GitHub Repos</a>
        <a href="https://www.linkedin.com/in/rikin-pithadia-20b94729b/" target="_blank">🔗 LinkedIn</a>
        <a href="https://docs.google.com/forms/d/e/1FAIpQLSfpaxeJk9vFaG1qdX5wSir6uvdr2R_3SHOM0zx83HV6hh6DdQ/viewform?usp=dialog" target="_blank">📝 Feedback</a>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Rating in stars
    rating = st.slider("⭐ Rate Us", 1, 5, 5)
    st.markdown(f"You rated: {rating} ⭐")
    
    st.divider()
    st.markdown("Built with [Ollama](https://ollama.ai/) | [LangChain](https://python.langchain.com/)")

# Initiate the chat engine
llm_engine = ChatOllama(
    model=selected_model,
    base_url="http://localhost:11434",
    temperature=0.3
)

# System prompt configuration
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are an expert AI coding assistant. Provide concise, correct solutions "
    "with strategic print statements for debugging. Always respond in English."
)

# Session state management
if "message_log" not in st.session_state:
    st.session_state.message_log = [{"role": "ai", "content": "Hi! I'm a GenAI. How can I help you today? 💻"}]

# Chat container
chat_container = st.container()

# Display chat messages
with chat_container:
    for message in st.session_state.message_log:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input and processing
user_query = st.chat_input("Type your question here...")

def generate_ai_response(prompt_chain):
    processing_pipeline = prompt_chain | llm_engine | StrOutputParser()
    return processing_pipeline.invoke({})

def build_prompt_chain():
    prompt_sequence = [system_prompt]
    for msg in st.session_state.message_log:
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
        elif msg["role"] == "ai":
            prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
    return ChatPromptTemplate.from_messages(prompt_sequence)

if user_query:
    # Add user message to log
    st.session_state.message_log.append({"role": "user", "content": user_query})
    
    # Generate AI response
    with st.spinner("🧠 Processing..."):
        prompt_chain = build_prompt_chain()
        ai_response = generate_ai_response(prompt_chain)
    
    # Add AI response to log
    st.session_state.message_log.append({"role": "ai", "content": ai_response})
    
    # Rerun to update chat display
    st.rerun()
