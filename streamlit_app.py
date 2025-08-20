import streamlit as st
from pathlib import Path
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_cerebras import ChatCerebras
from langchain_core.output_parsers import StrOutputParser
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

# Load environment
load_dotenv()

# --- Setup FAISS vectorstore + retriever ---
embeddings = NVIDIAEmbeddings(nvidia_api_key=os.getenv("NVIDIA_API_KEY"))
rl_vectorstore = FAISS.load_local(
    "faiss_index_rl_learn", embeddings, allow_dangerous_deserialization=True
)
retriever = rl_vectorstore.as_retriever()

# --- Setup LLM + prompt ---
llm_streaming = ChatCerebras(
    model="gpt-oss-120b", api_key=os.getenv("CEREBRAS_API_KEY"),

)

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful assistant help user based on context + use your own knowledge to respond.
You are specifically Trained On RL Data Reinforcement learning domain.

Response should be from context whenever required for simple query not requiring context then do not use context use own knowledge for resppond but ensure we have to stuck with out RL subject.

Whenever user used word *context* use given prompt context to you inside <context> </context> XML's.

Response should be direct clear and meaningful and considering examples whenever required

One note in respodn where ever uses conding use gymnassium package instead of gym as gym is deprecated also freedom to use wrappers and classes.

All code generated must be follow PIP - 8 RULES + OOPs concept based.

<context>
{context}
</context>

<query>
{query}
</query>
"""
)

chain_streaming = prompt | llm_streaming | StrOutputParser()

# --- Streamlit UI ---
st.set_page_config(page_title="RL Assistant", page_icon="🤖", layout="wide")
st.title("🤖 Reinforcement Learning Assistant")
st.caption("Powered by Cerebras + FAISS + NVIDIA Embeddings")

# Initialize session state for conversation history
if "history" not in st.session_state:
    st.session_state.history = []  # store tuples (query, response)


def stream_answer(query):
    """Streams response tokens and returns final full response as string."""
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join(d.page_content for d in docs)

    response_text = ""
    with st.chat_message("assistant"):
        md_container = st.empty()
        for chunk in chain_streaming.stream({"context": context, "query": query}):
            response_text += chunk
            md_container.markdown(response_text, unsafe_allow_html=True)

    return response_text


# --- Render existing history first ---
for role, content in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content)

# --- Chat input box ---
user_query = st.chat_input("Ask me about Reinforcement Learning...")
if user_query:
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_query)

    # Get streaming response
    response = stream_answer(user_query)

    # Store in history (only keep last 3)
    st.session_state.history.append(("user", user_query))
    st.session_state.history.append(("assistant", response))

    if len(st.session_state.history) > 6:  # 3 pairs (Q+A = 6 entries)
        st.session_state.history = st.session_state.history[-6:]

# --- Render history ---
if st.session_state.history:
    st.sidebar.subheader("📝 Last 3 Conversations")
    for i, (q, r) in enumerate(st.session_state.history, 1):
        st.sidebar.markdown(f"**Q{i}:** {q}")
        st.sidebar.markdown(f"**A{i}:** {r[:200]}...")  # preview response
