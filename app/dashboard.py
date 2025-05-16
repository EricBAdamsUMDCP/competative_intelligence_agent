# app/dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import os
import sys
import networkx as nx
import matplotlib.pyplot as plt
import io
from PIL import Image
import base64
import json
from datetime import datetime, timedelta

def get_api_key_expiration():
    try:
        response = requests.get(
            f"{API_URL}/api-key-status",
            headers=HEADERS
        )
        response.raise_for_status()
        data = response.json()
        
        if data and "sam_gov" in data:
            sam_gov = data["sam_gov"]
            days_old = sam_gov.get("days_old")
            
            if days_old is not None:
                days_left = 90 - days_old
                if days_left <= 0:
                    return {"status": "expired", "days_left": 0, "message": "⚠️ API KEY EXPIRED! Please rotate immediately."}
                elif days_left <= 10:
                    return {"status": "critical", "days_left": days_left, "message": f"⚠️ API KEY EXPIRES IN {days_left} DAYS! Rotate soon."}
                elif days_left <= 30:
                    return {"status": "warning", "days_left": days_left, "message": f"⚠️ API KEY EXPIRES IN {days_left} DAYS"}
                else:
                    return {"status": "ok", "days_left": days_left, "message": f"API Key valid for {days_left} more days"}
            
        return {"status": "unknown", "days_left": None, "message": "API Key status unknown"}
    except Exception:
        return {"status": "error", "days_left": None, "message": "Cannot check API Key status"}

# Get key status
key_status = get_api_key_expiration()

# Display key expiration prominently if needed
if key_status["status"] in ["expired", "critical", "warning"]:
    st.warning(key_status["message"], icon="⚠️")

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure page
st.set_page_config(
    page_title="GovCon Intelligence Dashboard",
    page_icon="📊",
    layout="wide"
)

# API connection settings
API_URL = os.environ.get("API_URL", "http://api:8000")
API_KEY = os.environ.get("API_KEY", "dev_key")
HEADERS = {"X-API-Key": API_KEY}

# Function to fetch data from API
def get_search_results(query):
    try:
        response = requests.post(
            f"{API_URL}/search",
            json={"query": query},
            headers=HEADERS
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return []

# Function to collect new data
def collect_data(sources=None):
    try:
        url = f"{API_URL}/collect"
        if sources:
            url += f"?sources={sources}"
            
        response = requests.get(
            url,
            headers=HEADERS
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error collecting data: {str(e)}")
        return {"status": "error", "message": str(e)}

# Function to extract entities
def extract_entities(text):
    try:
        response = requests.post(
            f"{API_URL}/extract-entities",
            json={"text": text},
            headers=HEADERS
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error extracting entities: {str(e)}")
        return {"entities": {}, "summary": {}}

# Function to get competitors
def get_competitors():
    try:
        response = requests.get(
            f"{API_URL}/competitors",
            headers=HEADERS
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching competitors: {str(e)}")
        return []

# Function to get agencies
def get_agencies():
    try:
        response = requests.get(
            f"{API_URL}/agencies",
            headers=HEADERS
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching agencies: {str(e)}")
        return []

# Function to get entity statistics
def get_entity_stats():
    try:
        response = requests.get(
            f"{API_URL}/entity-stats",
            headers=HEADERS
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching entity statistics: {str(e)}")
        return {}

# Function to get data sources
def get_data_sources():
    try:
        response = requests.get(
            f"{API_URL}/sources",
            headers=HEADERS
        )
        response.raise_for_status()
        return response.json().get("sources", [])
    except Exception as e:
        st.error(f"Error fetching data sources: {str(e)}")
        return []

# Function to get source details
def get_source_details(source_id):
    try:
        response = requests.get(
            f"{API_URL}/source/{source_id}",
            headers=HEADERS
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching source details: {str(e)}")
        return {}

# Function to get API key status
def get_api_key_status():
    try:
        response = requests.get(
            f"{API_URL}/api-key-status",
            headers=HEADERS
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching API key status: {str(e)}")
        return {}

# Function to get system health
def get_system_health():
    try:
        response = requests.get(
            f"{API_URL}/health",
            headers=HEADERS
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching system health: {str(e)}")
        return {"status": "error", "components": {}}

# Function to create entity network visualization
def create_entity_network(entities):
    G = nx.Graph()
    
    # Add nodes for each entity type
    colors = []
    
    # Define color mapping
    color_map = {
        "TECHNOLOGY": "skyblue",
        "REGULATION": "lightgreen",
        "CLEARANCE": "lightcoral",
        "AGENCY": "gold",
        "ORG": "lightgrey",
        "PERSON": "pink"
    }
    
    # Add nodes
    for entity_type, entity_list in entities.items():
        for entity in entity_list:
            G.add_node(entity["text"], type=entity_type)
    
    # Assign colors to nodes in the order of G.nodes()
    colors = [color_map.get(G.nodes[n].get("type", ""), "lightgrey") for n in G.nodes()]
    
    # Add edges between related entities
    added_edges = set()
    for entity_type, entity_list in entities.items():
        for entity in entity_list:
            entity_text = entity["text"]
            
            # Connect entities based on type relationships
            if entity_type == "TECHNOLOGY":
                # Connect technologies to regulations
                for reg in entities.get("REGULATION", []):
                    edge = (entity_text, reg["text"])
                    if edge not in added_edges and edge[::-1] not in added_edges:
                        G.add_edge(entity_text, reg["text"], weight=1)
                        added_edges.add(edge)
            
            elif entity_type == "AGENCY":
                # Connect agencies to clearances
                for clearance in entities.get("CLEARANCE", []):
                    edge = (entity_text, clearance["text"])
                    if edge not in added_edges and edge[::-1] not in added_edges:
                        G.add_edge(entity_text, clearance["text"], weight=1)
                        added_edges.add(edge)
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)
    node_count = len(G.nodes())
    node_colors = colors if len(colors) == node_count else ["lightgray"] * node_count
    if len(node_colors) == len(G.nodes()):
        nx.draw_networkx_nodes(G, pos, nodelist=list(G.nodes()), node_color=node_colors, node_size=500, alpha=0.8)
    else:
        nx.draw_networkx_nodes(G, pos, nodelist=list(G.nodes()), node_color=["lightgray"] * len(G.nodes()), node_size=500, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=1, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=10)
    
    # Add legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                               label=f"{entity_type}",
                               markerfacecolor=color, markersize=10)
                    for entity_type, color in color_map.items()]
    plt.legend(handles=legend_elements, loc='upper left')
    
    plt.title("Entity Relationship Network")
    plt.axis('off')
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    
    return img

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page",
    ["Overview", "Search Opportunities", "Competitor Analysis", "Entity Analysis", "Data Sources"]
)

#side bar api key warning
st.sidebar.write("### API Key Status")
if key_status["days_left"] is not None:
    if key_status["status"] == "expired":
        st.sidebar.error(f"SAM.gov API key expired! Please rotate now.", icon="🔴")
    elif key_status["status"] == "critical":
        st.sidebar.warning(f"SAM.gov API key expires in {key_status['days_left']} days", icon="⚠️")
    elif key_status["status"] == "warning":
        st.sidebar.warning(f"SAM.gov API key expires in {key_status['days_left']} days", icon="⚠️")
    else:
        st.sidebar.info(f"SAM.gov API key valid for {key_status['days_left']} days", icon="✅")
else:
    st.sidebar.info("SAM.gov API key status unknown", icon="❓")

# System status indicator in sidebar
system_health = get_system_health()
system_status = system_health.get("status", "unknown")
status_color = "🟢" if system_status == "healthy" else "🔴"

st.sidebar.write(f"{status_color} System Status: {system_status.capitalize()}")

# Check API key status
api_key_status = get_api_key_status()
if api_key_status and "sam_gov" in api_key_status:
    sam_gov_status = api_key_status["sam_gov"]
    if sam_gov_status.get("requires_update"):
        st.sidebar.warning(f"⚠️ SAM.gov API key needs rotation ({sam_gov_status.get('days_old', '?')} days old)")

# Overview page
if page == "Overview":
    st.title("Government Contracting Intelligence Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active Opportunities", "127", "+4")
    
    with col2:
        st.metric("Competitor Win Rate", "23%", "-2%")
    
    with col3:
        st.metric("Agency Sentiment", "Positive", "+5%")
    
    with col4:
        st.metric("Intelligence Coverage", "87%", "+3%")
    
    # Data collection
    st.subheader("Data Collection")
    
    # Get available sources
    sources = get_data_sources()
    source_ids = [s["id"] for s in sources]
    source_names = [s["name"] for s in sources]
    
    # Create multiselect for data sources
    selected_sources = st.multiselect(
        "Select data sources to collect from:",
        options=source_ids,
        default=source_ids,
        format_func=lambda x: next((s["name"] for s in sources if s["id"] == x), x)
    )
    
    # Button to collect data
    if st.button("Collect Latest Data"):
        with st.spinner("Collecting data..."):
            # Join selected sources with commas
            sources_param = ",".join(selected_sources) if selected_sources else None
            result = collect_data(sources_param)
            
            if result.get("status") == "success":
                source_results = result.get("sources", {})
                collected_items = [f"{source}: {count} items" for source, count in source_results.items()]
                
                st.success(f"Successfully collected data:\n" + "\n".join(collected_items))
                st.write(f"Total collected: {result.get('total_collected')}")
                st.write(f"Processed: {result.get('processed')}")
                st.write(f"Stored: {result.get('stored')}")
            else:
                st.error(f"Error: {result.get('message')}")
    
    # Sources breakdown
    if st.checkbox("Show Data Sources Breakdown"):
        st.subheader("Data Sources")
        
        # Create tabs for different source types
        source_tabs = st.tabs([s["name"] for s in sources])
        
        for i, source in enumerate(sources):
            with source_tabs[i]:
                st.write(f"### {source['name']}")
                st.write(f"Description: {source['description']}")
                
                # Show specific data based on source type
                if source["id"] == "sam.gov":
                    st.write("Latest update: Yesterday at 6:00 PM")
                    st.metric("Active Opportunities", "85", "+12")
                    
                    # API key management notice
                    st.info("⚠️ **API Key Management**: SAM.gov requires API keys to be updated every 90 days. Make sure to track and rotate your key regularly.")
                    
                    # Sample opportunities table
                    st.write("Recent Opportunities:")
                    sample_sam = pd.DataFrame({
                        "Title": ["Cybersecurity Services", "Cloud Migration", "IT Support"],
                        "Agency": ["DoD", "HHS", "GSA"],
                        "Posted Date": ["2025-05-12", "2025-05-10", "2025-05-08"],
                        "Value ($)": ["5,600,000", "3,200,000", "1,800,000"]
                    })
                    st.dataframe(sample_sam)
                    
                    # SAM.gov terms of use
                    with st.expander("SAM.gov Data Usage Terms"):
                        st.markdown("""
                        ### SAM.gov Data Usage Terms
                        
                        Data provided by SAM.gov is subject to specific terms of use. By using this system, you agree to:
                        
                        - Only access data you are authorized to access
                        - Not use the data for unauthorized purposes
                        - Not share your API key with unauthorized parties
                        - Update your API key every 90 days
                        
                        For complete terms, visit [SAM.gov](https://sam.gov).
                        """)
                
                elif source["id"] == "usaspending.gov":
                    st.write("Latest update: Today at 8:30 AM")
                    st.metric("Contract Awards", "132", "+8")
                    
                    # Sample awards table
                    st.write("Recent Awards:")
                    sample_usa = pd.DataFrame({
                        "Award ID": ["CONT2023001", "CONT2023002", "CONT2023003"],
                        "Recipient": ["TechDefense Solutions", "CloudSys Inc", "Federal IT Partners"],
                        "Agency": ["DoD", "HHS", "DHS"],
                        "Amount ($)": ["4,200,000", "2,800,000", "1,500,000"]
                    })
                    st.dataframe(sample_usa)
                
                elif source["id"] == "industry_news":
                    st.write("Latest update: Today at 10:15 AM")
                    st.metric("News Articles", "47", "+5")
                    
                    # Sample news articles
                    st.write("Recent Articles:")
                    sample_news = pd.DataFrame({
                        "Title": [
                            "DoD Awards $50M Cybersecurity Contract", 
                            "GSA Announces New Cloud Initiative", 
                            "HHS Modernization Program Expands"
                        ],
                        "Source": ["Washington Technology", "Federal News Network", "FCW"],
                        "Date": ["2025-05-13", "2025-05-12", "2025-05-11"]
                    })
                    st.dataframe(sample_news)
    
    # Sample visualization
    st.subheader("Contract Awards by Agency")
    
    # Mock data for now
    data = {
        "Agency": ["DoD", "HHS", "GSA", "DHS", "DoE"],
        "Value ($M)": [120, 85, 65, 45, 30]
    }
    
    df = pd.DataFrame(data)
    fig = px.bar(df, x="Agency", y="Value ($M)", color="Agency")
    st.plotly_chart(fig, use_container_width=True)
    
    # Technology breakdown
    st.subheader("Technology Requirements Breakdown")
    tech_data = {
        "Technology": ["Cloud", "Cybersecurity", "AI/ML", "DevSecOps", "5G"],
        "Frequency": [42, 35, 28, 20, 15]
    }
    
    tech_df = pd.DataFrame(tech_data)
    tech_fig = px.pie(tech_df, values="Frequency", names="Technology", hole=0.4)
    st.plotly_chart(tech_fig, use_container_width=True)
    
    # Recent awards
    st.subheader("Recent Contract Awards")
    
    # Mock recent awards
    recent_awards = [
        {
            "title": "Cybersecurity Services for Department of Defense",
            "agency": "Department of Defense",
            "contractor": "TechDefense Solutions",
            "value": 5600000,
            "date": (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
        },
        {
            "title": "Cloud Migration Services for HHS",
            "agency": "Department of Health and Human Services",
            "contractor": "CloudSys Inc",
            "value": 3200000,
            "date": (datetime.now() - timedelta(days=12)).strftime("%Y-%m-%d")
        }
    ]
    
    for award in recent_awards:
        with st.expander(f"{award['title']} - ${award['value']:,}"):
            st.write(f"**Agency:** {award['agency']}")
            st.write(f"**Contractor:** {award['contractor']}")
            st.write(f"**Award Date:** {award['date']}")

# Search page
elif page == "Search Opportunities":
    st.title("Search Contract Opportunities")
    
    query = st.text_input("Search for opportunities", "cybersecurity")
    
    if st.button("Search"):
        with st.spinner("Searching..."):
            results = get_search_results(query)
            
            if results:
                st.success(f"Found {len(results)} results")
                for result in results:
                    with st.expander(f"{result.get('title')}"):
                        st.write(f"**ID:** {result.get('id')}")
                        st.write(f"**Description:** {result.get('description')}")
                        st.write(f"**Value:** ${result.get('value', 0):,.2f}")
                        st.write(f"**Award Date:** {result.get('award_date')}")
                        
                        # Add a button to analyze entities in this opportunity
                        if st.button(f"Analyze Entities for {result.get('id')}", key=f"analyze_{result.get('id')}"):
                            with st.spinner("Analyzing entities..."):
                                entities = extract_entities(result.get('description', ''))
                                
                                if entities and "entities" in entities:
                                    st.write("**Extracted Entities:**")
                                    for entity_type, entity_list in entities.get("entities", {}).items():
                                        if entity_list:
                                            st.write(f"*{entity_type}*: {', '.join([e['text'] for e in entity_list])}")
            else:
                st.info("No results found. Try a different search term or collect some data first.")

# Competitor Analysis
elif page == "Competitor Analysis":
    st.title("Competitor Analysis")
    
    # Get competitor list
    competitors = [c.get("name") for c in get_competitors()]
    
    if not competitors:
        competitors = ["TechDefense Solutions", "CloudSys Inc", "Federal IT Partners", "CyberSecure Solutions"]
    
    selected_competitor = st.selectbox("Select Competitor", competitors)
    
    st.subheader(f"Analysis for {selected_competitor}")
    
    # Mock contract history
    contract_data = pd.DataFrame({
        "Agency": ["DoD", "HHS", "GSA", "DHS", "DoJ"],
        "Value ($M)": [12.4, 8.7, 5.2, 3.8, 2.1],
        "Award Date": pd.date_range(start="1/1/2023", periods=5, freq="M")
    })
    
    # Show contract history
    st.subheader("Contract History")
    st.dataframe(contract_data)
    
    # Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Contract Value by Agency")
        fig1 = px.pie(contract_data, values="Value ($M)", names="Agency")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader("Contract Value Over Time")
        contract_data["Month"] = contract_data["Award Date"].dt.strftime("%b %Y")
        fig2 = px.line(contract_data, x="Month", y="Value ($M)", markers=True)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Technology capabilities
    st.subheader("Technology Capabilities")
    
    # Mock technology data
    tech_data = [
        {"technology": "Cloud", "expertise": 0.9, "market_share": 0.15},
        {"technology": "Cybersecurity", "expertise": 0.8, "market_share": 0.12},
        {"technology": "AI/ML", "expertise": 0.6, "market_share": 0.08},
        {"technology": "DevSecOps", "expertise": 0.7, "market_share": 0.10},
        {"technology": "Data Analytics", "expertise": 0.5, "market_share": 0.05},
    ]
    
    tech_df = pd.DataFrame(tech_data)
    
    # Create radar chart
    fig3 = px.line_polar(tech_df, r="expertise", theta="technology", line_close=True)
    fig3.update_traces(fill='toself')
    st.plotly_chart(fig3, use_container_width=True)

# Entity Analysis page
elif page == "Entity Analysis":
    st.title("Contract Entity Analysis")
    
    # Text input for analysis
    st.subheader("Analyze Contract Text")
    contract_text = st.text_area("Enter contract description for analysis", 
                               "The Department of Defense is seeking cybersecurity services to enhance the security posture of critical infrastructure. Services include vulnerability assessment, penetration testing, and security monitoring. Contractor must have Top Secret clearance and comply with NIST 800-53 and CMMC Level 3 requirements.")
    
    if st.button("Analyze Entities"):
        with st.spinner("Analyzing text..."):
            # Extract entities
            result = extract_entities(contract_text)
            
            if result and "entities" in result:
                entities = result["entities"]
                summary = result.get("summary", {})
                
                # Show entity summary
                st.subheader("Entity Summary")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Technologies:**")
                    for tech in summary.get("tech_stack", []):
                        st.write(f"- {tech}")
                    
                    st.write("**Regulatory Requirements:**")
                    for reg in summary.get("regulatory_requirements", []):
                        st.write(f"- {reg}")
                
                with col2:
                    st.write("**Clearance Requirements:**")
                    for clearance in summary.get("clearance_requirements", []):
                        st.write(f"- {clearance}")
                    
                    st.write("**Agencies Involved:**")
                    for agency in summary.get("agencies_involved", []):
                        st.write(f"- {agency}")
                
                # Show detailed entities
                st.subheader("Detailed Entities")
                for entity_type, entity_list in entities.items():
                    if entity_list:
                        with st.expander(f"{entity_type} ({len(entity_list)})"):
                            for entity in entity_list:
                                st.write(f"- {entity['text']}")
                
                # Show entity network visualization
                st.subheader("Entity Relationship Network")
                if len(entities) > 1:  # Only show if we have multiple entity types
                    img = create_entity_network(entities)
                    st.image(img, use_column_width=True)
                else:
                    st.info("Not enough entities to create a relationship network.")
                
                # Competitive intelligence insights
                st.subheader("Competitive Intelligence Insights")
                
                # Mock data for demonstration
                st.write("**Top contractors for similar requirements:**")
                
                contractors = [
                    {"name": "TechDefense Solutions", "win_rate": 0.65, "past_performance": 4.8},
                    {"name": "CyberSecure Inc", "win_rate": 0.55, "past_performance": 4.6},
                    {"name": "InfoSec Partners", "win_rate": 0.48, "past_performance": 4.3}
                ]
                
                contractor_df = pd.DataFrame(contractors)
                st.dataframe(contractor_df)
                
                # Technology trends
                st.write("**Technology trends in this market segment:**")
                
                trends = [
                    {"technology": "Zero Trust Architecture", "growth_rate": 0.35, "market_adoption": 0.68},
                    {"technology": "Container Security", "growth_rate": 0.28, "market_adoption": 0.52},
                    {"technology": "Cloud Security Posture Management", "growth_rate": 0.22, "market_adoption": 0.45}
                ]
                
                trends_df = pd.DataFrame(trends)
                st.dataframe(trends_df)
                
                # Strategic recommendations
                st.subheader("Strategic Recommendations")
                st.write("Based on the extracted entities and market analysis:")
                st.write("1. Emphasize expertise in Zero Trust Architecture in proposals")
                st.write("2. Highlight past performance with DoD cybersecurity projects")
                st.write("3. Ensure team has appropriate security clearances")
                st.write("4. Demonstrate compliance with NIST 800-53 and CMMC Level 3")
                
            else:
                st.warning("No entities found in the text.")

    # Entity statistics from knowledge graph
    st.subheader("Entity Statistics from Knowledge Graph")
    
    # Get entity statistics
    entity_stats = get_entity_stats()
    
    if entity_stats:
        # Technology stats
        tech_stats = entity_stats.get("technology", {})
        if tech_stats and "top_entities" in tech_stats:
            st.write("**Top Technologies in Contracts:**")
            
            tech_data = tech_stats["top_entities"]
            tech_df = pd.DataFrame(tech_data)
            
            fig = px.bar(tech_df, x="name", y="count", color="name")
            st.plotly_chart(fig, use_container_width=True)
        
        # Regulation stats
        reg_stats = entity_stats.get("regulation", {})
        if reg_stats and "top_entities" in reg_stats:
            st.write("**Top Regulatory Requirements:**")
            
            reg_data = reg_stats["top_entities"]
            reg_df = pd.DataFrame(reg_data)
            
            fig = px.bar(reg_df, x="name", y="count", color="name")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No entity statistics available yet. Try collecting more data.")

# Data Sources page (new page)
elif page == "Data Sources":
    st.title("Data Sources Management")
    
    # Get available sources
    sources = get_data_sources()
    
    st.write("## Available Data Sources")
    st.write("The system integrates data from the following sources:")
    
    # Create a table of sources
    source_table = []
    for source in sources:
        # Get detailed information for each source
        details = get_source_details(source["id"])
        
        # Add status indicator
        status_color = "🟢" if details.get("status") == "active" else "🔴"
        
        # Format last updated time
        last_updated = details.get("last_updated")
        if last_updated:
            formatted_time = datetime.fromisoformat(last_updated).strftime("%Y-%m-%d %H:%M")
        else:
            formatted_time = "Never"
        
        # Add to table
        source_table.append({
            "Source": f"{status_color} {source['name']}",
            "Description": source["description"],
            "Last Updated": formatted_time,
            "Items": details.get("item_count", 0)
        })
    
    # Display table
    source_df = pd.DataFrame(source_table)
    st.table(source_df)
    
    # Add data collection section
    st.write("## Data Collection")
    
    source_tabs = st.tabs([s["name"] for s in sources])
    
    for i, source in enumerate(sources):
        with source_tabs[i]:
            st.write(f"### {source['name']} Configuration")
            
            # Get source details
            details = get_source_details(source["id"])
            
            # Show source-specific information
            if source["id"] == "sam.gov":
                st.write("**Data types:** Opportunities, Awards, Entities")
                st.write("**API endpoint:** https://api.sam.gov/opportunities/v1/search")
                
                # Show API key status
                api_key_status = details.get("api_key_status", {})
                if api_key_status.get("present"):
                    key_created = api_key_status.get("created")
                    days_old = api_key_status.get("days_old")
                    
                    if days_old and days_old > 90:
                        st.error(f"⚠️ API Key is expired ({days_old} days old). Please rotate immediately.")
                    elif days_old and days_old > 80:
                        st.warning(f"⚠️ API Key rotation needed soon ({days_old} days old).")
                    elif days_old:
                        st.success(f"API Key is valid ({days_old} days old).")
                    
                    if key_created:
                        st.write(f"API Key created on: {key_created}")
                else:
                    st.error("⚠️ SAM.gov API key is not configured.")
                
                # SAM.gov Data Usage Terms
                st.markdown("""
                ### SAM.gov Data Usage Terms
                
                Data provided by SAM.gov is subject to specific terms of use. By using this system, you agree to:
                
                - Only access data you are authorized to access
                - Not use the data for unauthorized purposes
                - Not share your API key with unauthorized parties
                - Update your API key every 90 days
                
                For complete terms, visit [SAM.gov](https://sam.gov).
                """)
                
                # Mock configuration options
                date_range = st.date_input(
                    "Date range to collect",
                    value=(datetime.now() - timedelta(days=30), datetime.now()),
                    max_value=datetime.now()
                )
                
                search_params = st.text_input("Search filters (optional)", "")
                
                st.button(f"Collect from {source['name']}", key=f"collect_{source['id']}")
            
            elif source["id"] == "usaspending.gov":
                st.write("**Data types:** Awards, Agencies, Recipients")
                st.write("**API endpoint:** https://api.usaspending.gov/api/v2/search/spending_by_award/")
                
                # Mock configuration options
                date_range = st.date_input(
                    "Date range to collect",
                    value=(datetime.now() - timedelta(days=90), datetime.now()),
                    max_value=datetime.now()
                )
                
                award_types = st.multiselect(
                    "Award types",
                    ["contracts", "grants", "loans", "direct_payments", "other"],
                    default=["contracts"]
                )
                
                st.button(f"Collect from {source['name']}", key=f"collect_{source['id']}")
            
            elif source["id"] == "industry_news":
                st.write("**Data types:** News, Trends, Announcements")
                st.write("**Sources:** Washington Technology, Federal News Network, FCW")
                
                # Mock configuration options
                sources_list = st.multiselect(
                    "News sources",
                    ["Washington Technology", "Federal News Network", "FCW", "GovExec", "NextGov"],
                    default=["Washington Technology", "Federal News Network", "FCW"]
                )
                
                max_articles = st.slider("Maximum articles per source", 5, 50, 10)
                
                st.button(f"Collect from {source['name']}", key=f"collect_{source['id']}")
    
    # Integration management section
    st.write("## Data Integration Status")
    
    integration_data = {
        "Component": [
            "SAM.gov API", 
            "USASpending.gov API", 
            "News Scraper", 
            "Entity Extraction", 
            "Knowledge Graph"
        ],
        "Status": ["Active", "Active", "Active", "Active", "Active"],
        "Last Run": [
            "2025-05-13 10:30", 
            "2025-05-14 08:15", 
            "2025-05-14 09:45", 
            "2025-05-14 09:45", 
            "2025-05-14 09:45"
        ],
        "Items Processed": [85, 132, 47, 264, 264]
    }
    
    integration_df = pd.DataFrame(integration_data)
    st.table(integration_df)
    
    # System statistics
    st.write("## System Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Data Sources", "3", "+1")
    
    with col2:
        st.metric("Total Items Collected", "264", "+42")
    
    with col3:
        st.metric("Integration Success Rate", "98.5%", "+0.5%")

# Add footer
st.markdown("---")
st.markdown("Government Contracting Competitive Intelligence System | Developed with Claude")