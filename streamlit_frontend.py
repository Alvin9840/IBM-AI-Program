import streamlit as st 
from nba_api.stats.static import teams

# streamlit run streamlit_frontend.py
# Sidebar
st.sidebar.title("NBA Fan Engagement Tool")
nba_teams = teams.get_teams()
team_names = [team['full_name'] for team in nba_teams]
team = st.sidebar.selectbox("Select Team", team_names)

tab = st.sidebar.radio("Function", ["Fan Sentiment", "Content Generation", "Trend Analysis"])

# Main content
st.title("NBA AI Marketing Dashboard")

if tab == "Fan Sentiment":
    st.header(f"Fan Sentiment for {team}")

elif tab == "Content Generation":
    st.header("Social Media Content Generator")
    user_input = st.text_input("Describe your event or promotion for the team...")
    if st.button("Generate Post"):
        pass

elif tab == "Trend Analysis":
    st.header("Fan Engagement Trend")
