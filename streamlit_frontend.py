import streamlit as st
from agents.coordinator_agent import CoordinatorAgent
import json
from pathlib import Path
from datetime import datetime
import time 

# streamlit run streamlit_frontend.py

# --- Setup local database and paths ---
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
SAVE_PATH = DATA_DIR / "nba_chat_sessions.json"

def save_sessions(sessions):
    """Save all chat sessions to local JSON file."""
    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(sessions, f, ensure_ascii=False, indent=2)

def load_sessions():
    """Load all chat sessions from local JSON file."""
    if SAVE_PATH.exists():
        with open(SAVE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

# --- Session State Initialization ---

# Load chat sessions from database
if "all_sessions" not in st.session_state:
    st.session_state.all_sessions = load_sessions()

# Initialize agent (one per app run)
if "coordinator" not in st.session_state:
    st.session_state.coordinator = CoordinatorAgent()

# Generate a new chat session and set as default if none exists
if "selected_session" not in st.session_state:
    new_key = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.all_sessions[new_key] = []
    st.session_state.selected_session = new_key

session_keys = list(st.session_state.all_sessions.keys())

# --- Sidebar: Chat session selector and new session button ---
st.sidebar.title("NBA Chat History")

if st.sidebar.button("➕ New Chat Session"):
    new_key = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.all_sessions[new_key] = []
    st.session_state.selected_session = new_key
    session_keys.append(new_key)

selected_session = st.sidebar.selectbox(
    "Choose a chat session:",
    options=session_keys[::-1],
    index=0
)
st.session_state.selected_session = selected_session

# Switch chat context when session changes
if "messages" not in st.session_state or selected_session != st.session_state.get("current_session"):
    st.session_state.messages = st.session_state.all_sessions[selected_session]
    st.session_state.current_session = selected_session

# --- Main UI ---
st.title("NBA Fan Engagement AI ChatBot")

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat input and agent response logic ---
if prompt := st.chat_input("What would you like to ask?"):
    # Display user message and add to session history
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Assistant response with spinner during processing
    with st.chat_message("assistant"):
        with st.spinner("Processing your message..."):
            # Call your coordinator agent to process user input
            result = st.session_state.coordinator.process_director_request(prompt)
            response = result["final_response"]
            workflow_steps = result.get("workflow_steps", [])

        # Display AI's response
        st.markdown(response)

        # Optionally show expanded agent workflow details
        if result.get("tool_calls") or result.get("agent_results"):
            with st.expander("🤖 Agent Workflow Details", expanded=False):
                st.write(f"**Iterations:** {result.get('iterations', 0)}")

                # Show agents/tools that were called
                if result.get("tool_calls"):
                    st.write("**Agents Called:**")
                    agents_used = set()
                    for tool_call in result["tool_calls"]:
                        agent_name = tool_call["tool"]
                        agents_used.add(agent_name)
                        st.write(f"• {agent_name} (Iteration {tool_call['iteration']})")
                    if agents_used:
                        agent_list = ", ".join(sorted(agents_used))
                        st.success(f"🤖 **Agents that contributed:** {agent_list}")

                # Show a summary for each agent's results
                if result.get("agent_results"):
                    st.write("**Agent Results:**")
                    for agent_name, results in result["agent_results"].items():
                        status = "✅ Success" if all(r.get("status") == "success" for r in results) else "⚠️ Partial/Mixed"
                        st.write(f"• **{agent_name}:** {status} ({len(results)} calls)")

    st.session_state.messages.append({"role": "assistant", "content": response})
    # Save all sessions after message update
    st.session_state.all_sessions[selected_session] = st.session_state.messages
    save_sessions(st.session_state.all_sessions)
