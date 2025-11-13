import streamlit as st
from agents.coordinator_agent import CoordinatorAgent

# Initialize the coordinator agent (do this once, not on every request)
if "coordinator" not in st.session_state:
    st.session_state.coordinator = CoordinatorAgent()

st.title("NBA Fan Engagement AI ChatBot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Process the request using the coordinator agent
    result = st.session_state.coordinator.process_director_request(prompt)
    response = result["final_response"]
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
        
        # Show agent workflow details in an expander
        if result.get("tool_calls") or result.get("agent_results"):
            with st.expander("🤖 Agent Workflow Details", expanded=False):
                # Show iterations
                st.write(f"**Iterations:** {result.get('iterations', 0)}")
                
                # Show which agents/tools were called
                if result.get("tool_calls"):
                    st.write("**Agents Called:**")
                    agents_used = set()
                    for tool_call in result["tool_calls"]:
                        agent_name = tool_call["tool"]
                        agents_used.add(agent_name)
                        st.write(f"• {agent_name} (Iteration {tool_call['iteration']})")
                    
                    # Show summary of agents
                    if agents_used:
                        agent_list = ", ".join(sorted(agents_used))
                        st.success(f"🤖 **Agents that contributed:** {agent_list}")
                
                # Show agent results summary
                if result.get("agent_results"):
                    st.write("**Agent Results:**")
                    for agent_name, results in result["agent_results"].items():
                        status = "✅ Success" if all(r.get("status") == "success" for r in results) else "⚠️ Partial/Mixed"
                        st.write(f"• **{agent_name}:** {status} ({len(results)} calls)")
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})