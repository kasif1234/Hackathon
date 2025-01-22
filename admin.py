import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from groq import Groq
import json
import warnings

# Must be the first Streamlit command
st.set_page_config(
    page_title="ANTNA Admin",
    page_icon="🐜",
    layout="wide",
    initial_sidebar_state="expanded",
)

warnings.filterwarnings('ignore')

# Load CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

try:
    load_css('styles.css')
except:
    st.warning("styles.css not found. Using default styling.")

# Initialize Groq client
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    GROQ_API_KEY = st.secrets.GROQ_API_KEY

try:
    groq_client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    st.error(f"Failed to initialize Groq client: {str(e)}")
    st.stop()

def generate_disaster_data(groq_client, scenario_prompt):
    """Generate structured disaster alerts data"""
    messages = [
        {"role": "system", "content": """You are a disaster data generator for Qatar's emergency management system. 
        Generate exactly 10 emergency alerts as a JSON array with this exact structure for each alert:
        {
            "type": "one of [Sandstorm, Heat Wave, Flash Flood, Dust Storm, Strong Winds, Thunderstorm]",
            "severity": "one of [Low, Medium, High]",
            "location": "specific Qatar location",
            "time": "current time in YYYY-MM-DD HH:MM format",
            "description": "detailed description of the alert"
        }"""},
        {"role": "user", "content": f"Generate 10 structured alerts for: {scenario_prompt}"}
    ]
    
    try:
        response = groq_client.chat.completions.create(
            messages=messages,
            model="mixtral-8x7b-32768",
            temperature=0.7,
            max_tokens=1000
        )
        alerts = json.loads(response.choices[0].message.content)
        return pd.DataFrame(alerts)
    except Exception as e:
        st.error(f"Error generating disaster data: {str(e)}")
        return pd.DataFrame()

def generate_resource_data(groq_client, scenario_prompt):
    """Generate structured facility resource data"""
    messages = [
        {"role": "system", "content": """You are a facility resource manager for Qatar's emergency management system. 
        Generate exactly 10 facility reports as a JSON array with this exact structure for each facility:
        {
            "facility": "name of Qatar facility",
            "water": "water supply (1000-10000)",
            "food": "food supply (500-5000)",
            "medical": "medical supplies (0-100)",
            "beds": "total beds (100-1000)",
            "current_occupancy": "current occupants (less than beds)",
            "last_updated": "current time in YYYY-MM-DD HH:MM format"
        }"""},
        {"role": "user", "content": f"Generate 10 structured facility reports for: {scenario_prompt}"}
    ]
    
    try:
        response = groq_client.chat.completions.create(
            messages=messages,
            model="mixtral-8x7b-32768",
            temperature=0.7,
            max_tokens=1000
        )
        resources = json.loads(response.choices[0].message.content)
        return pd.DataFrame(resources)
    except Exception as e:
        st.error(f"Error generating resource data: {str(e)}")
        return pd.DataFrame()

def generate_social_updates(groq_client, scenario_prompt):
    """Generate structured social media updates"""
    messages = [
        {"role": "system", "content": """You are a social media feed generator for Qatar's emergency management system. 
        Generate exactly 10 social updates as a JSON array with this exact structure for each update:
        {
            "source_type": "one of [Official, Healthcare, Emergency, Media, Citizen]",
            "username": "Twitter handle with @",
            "message": "update content",
            "location": "Qatar location",
            "verified": "true or false",
            "trust_score": "0.0 to 1.0",
            "timestamp": "current time in YYYY-MM-DD HH:MM format",
            "engagement": "100 to 5000"
        }"""},
        {"role": "user", "content": f"Generate 10 structured social updates for: {scenario_prompt}"}
    ]
    
    try:
        response = groq_client.chat.completions.create(
            messages=messages,
            model="mixtral-8x7b-32768",
            temperature=0.7,
            max_tokens=1000
        )
        updates = json.loads(response.choices[0].message.content)
        return pd.DataFrame(updates)
    except Exception as e:
        st.error(f"Error generating social updates: {str(e)}")
        return pd.DataFrame()

def process_scenario(prompt, groq_client):
    """Process scenario and generate all data"""
    try:
        st.session_state.disasters = pd.DataFrame()
        st.session_state.resources = pd.DataFrame()
        st.session_state.updates = pd.DataFrame()

        with st.spinner("Generating disaster alerts..."):
            disasters_df = generate_disaster_data(groq_client, prompt)
            if not disasters_df.empty:
                st.session_state.disasters = disasters_df
                st.success("✅ Disaster alerts generated!")
        
        with st.spinner("Generating resource data..."):
            resources_df = generate_resource_data(groq_client, prompt)
            if not resources_df.empty:
                st.session_state.resources = resources_df
                st.success("✅ Resource data generated!")
        
        with st.spinner("Generating social updates..."):
            updates_df = generate_social_updates(groq_client, prompt)
            if not updates_df.empty:
                st.session_state.updates = updates_df
                st.success("✅ Social updates generated!")
        
        return True
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return False

# Initialize session state with empty DataFrames
if 'disasters' not in st.session_state:
    st.session_state.disasters = pd.DataFrame()
if 'resources' not in st.session_state:
    st.session_state.resources = pd.DataFrame()
if 'updates' not in st.session_state:
    st.session_state.updates = pd.DataFrame()

# Export data function for app.py
def get_simulation_data():
    return {
        'disasters': st.session_state.disasters,
        'resources': st.session_state.resources,
        'updates': st.session_state.updates
    }

# Main interface
st.markdown("""
    <div class="title-block">
        <h1>🐜 ANTNA Admin</h1>
        <p><span class="status-indicator status-active"></span>Crisis Simulation Control</p>
    </div>
""", unsafe_allow_html=True)

# Create tabs
tab1, tab2 = st.tabs(["💭 Scenario Generator", "📊 Current Data"])

# Scenario Generator Tab
with tab1:
    st.markdown("<h2>💭 Generate Emergency Scenario</h2>", unsafe_allow_html=True)
    
    # Example scenarios
    examples = {
        "Sandstorm": "A severe sandstorm is approaching Doha with winds exceeding 80km/h. Visibility is dropping rapidly.",
        "Heatwave": "A heatwave has hit Qatar with temperatures reaching 50°C, causing widespread power outages.",
        "Flooding": "Heavy rainfall has caused flash flooding in Al Wakrah, with water levels rising rapidly.",
        "Multiple": "Multiple dust storms are affecting northern Qatar regions with strong winds."
    }
    
    scenario_type = st.selectbox("Select Scenario Type", ["Custom"] + list(examples.keys()))
    
    if scenario_type == "Custom":
        prompt = st.text_area(
            "Describe the emergency scenario",
            placeholder="Describe the emergency scenario in detail...",
            height=150
        )
    else:
        prompt = examples[scenario_type]
        st.text_area("Scenario Description", value=prompt, height=150, disabled=True)
    
    if st.button("Generate Scenario", type="primary"):
        if prompt:
            success = process_scenario(prompt, groq_client)
            if success:
                st.balloons()
        else:
            st.warning("Please enter a scenario description")

# Data Viewer Tab
with tab2:
    st.markdown("<h2>📊 Current Simulation Data</h2>", unsafe_allow_html=True)
    
    if not st.session_state.disasters.empty:
        st.markdown("<h3>🚨 Active Disasters</h3>", unsafe_allow_html=True)
        st.dataframe(st.session_state.disasters, use_container_width=True)
    
    if not st.session_state.resources.empty:
        st.markdown("<h3>🏥 Resource Levels</h3>", unsafe_allow_html=True)
        st.dataframe(st.session_state.resources, use_container_width=True)
    
    if not st.session_state.updates.empty:
        st.markdown("<h3>📱 Social Updates</h3>", unsafe_allow_html=True)
        st.dataframe(st.session_state.updates, use_container_width=True)
    
    if (st.session_state.disasters.empty and 
        st.session_state.resources.empty and 
        st.session_state.updates.empty):
        st.info("No simulation data available. Generate a scenario to see data here.")
