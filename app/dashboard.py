# app/dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import json
import os
import sys
import networkx as nx
import matplotlib.pyplot as plt
import io
from PIL import Image
import base64
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure page
st.set_page_config(
    page_title="GovCon Intelligence Dashboard",
    page_icon="📊",
    layout="wide"
)

# API connection settings
API_URL = "http://api:8000"
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
def collect_data():
    try:
        response = requests.get(
            f"{API_URL}/collect",
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

# Function to analyze opportunity
def analyze_opportunity(opportunity_data):
    try:
        response = requests.post(
            f"{API_URL}/analyze-opportunity",
            json=opportunity_data,
            headers=HEADERS
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error analyzing opportunity: {str(e)}")
        return {"status": "error", "message": str(e)}

# Function to analyze competitor
def analyze_competitor(competitor_id, include_technologies=True, include_agencies=True):
    try:
        response = requests.post(
            f"{API_URL}/analyze-competitor",
            json={
                "competitor_id": competitor_id,
                "include_technologies": include_technologies,
                "include_agencies": include_agencies
            },
            headers=HEADERS
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error analyzing competitor: {str(e)}")
        return {"status": "error", "message": str(e)}

# Function to record bid feedback
def record_bid_feedback(opportunity_id, bid_decision, win_result=None, feedback_notes=""):
    try:
        response = requests.post(
            f"{API_URL}/bid-feedback",
            json={
                "opportunity_id": opportunity_id,
                "bid_decision": bid_decision,
                "win_result": win_result,
                "feedback_notes": feedback_notes
            },
            headers=HEADERS
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error recording bid feedback: {str(e)}")
        return {"status": "error", "message": str(e)}

# Function to get market intelligence
def get_market_intelligence(agency_id=None, technology=None, start_date=None, end_date=None):
    try:
        response = requests.post(
            f"{API_URL}/market-intelligence",
            json={
                "agency_id": agency_id,
                "technology": technology,
                "start_date": start_date,
                "end_date": end_date
            },
            headers=HEADERS
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error getting market intelligence: {str(e)}")
        return {"status": "error", "message": str(e)}

# Function to get workflow state
def get_workflow_state(workflow_id):
    try:
        response = requests.get(
            f"{API_URL}/workflow-state/{workflow_id}",
            headers=HEADERS
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error getting workflow state: {str(e)}")
        return {"status": "error", "message": str(e)}

# Function to get all agents
def get_agents():
    try:
        response = requests.get(
            f"{API_URL}/agents",
            headers=HEADERS
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error getting agents: {str(e)}")
        return {"status": "error", "message": str(e)}

# Function to run a specific agent
def run_agent(agent_id, params):
    try:
        response = requests.post(
            f"{API_URL}/agents/{agent_id}/run",
            json=params,
            headers=HEADERS
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error running agent: {str(e)}")
        return {"status": "error", "message": str(e)}

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
            colors.append(color_map.get(entity_type, "lightgrey"))
    
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
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=500, alpha=0.8)
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
    ["Overview", "Search Opportunities", "Opportunity Analysis", "Competitor Analysis", "Market Intelligence", "Agent System"]
)

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
    
    # Button to collect data
    if st.button("Collect Latest Data"):
        with st.spinner("Collecting data..."):
            result = collect_data()
            if result.get("status") == "success":
                st.success(f"Successfully started data collection workflow. Workflow ID: {result.get('workflow_id')}")
                
                # Show workflow details if available
                if result.get("workflow_id"):
                    workflow_state = get_workflow_state(result["workflow_id"])
                    if workflow_state.get("status") == "success":
                        st.json(workflow_state["state"])
            else:
                st.error(f"Error: {result.get('message')}")
    
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
                        
                        # Add a button to analyze this opportunity
                        if st.button(f"Full Analysis for {result.get('id')}", key=f"analyze_{result.get('id')}"):
                            st.session_state[f"analyze_opp_{result.get('id')}"] = result
                            st.experimental_rerun()
            else:
                st.info("No results found. Try a different search term or collect some data first.")
    
    # If an opportunity was selected for analysis
    for key in list(st.session_state.keys()):
        if key.startswith("analyze_opp_") and st.session_state[key]:
            opportunity = st.session_state[key]
            
            st.subheader(f"Analysis for: {opportunity.get('title')}")
            
            # Prepare opportunity data for analysis
            opp_data = {
                "opportunity_id": opportunity.get('id'),
                "title": opportunity.get('title'),
                "agency": opportunity.get('agency', 'Unknown'),
                "value": float(opportunity.get('value', 0)),
                "description": opportunity.get('description', '')
            }
            
            with st.spinner("Running opportunity analysis..."):
                analysis_result = analyze_opportunity(opp_data)
                
                if analysis_result.get("status") == "success":
                    analysis = analysis_result.get("analysis", {})
                    
                    # Display analysis results
                    st.write("### Analysis Results")
                    
                    # Overall score and recommendation
                    overall_score = analysis.get("analyzed_opportunities", [{}])[0].get("overall_score", 0)
                    recommendation = analysis.get("analyzed_opportunities", [{}])[0].get("recommendation", "No recommendation")
                    
                    # Create a gauge chart for the score
                    score_fig = px.pie(values=[overall_score, 100-overall_score], 
                                    names=["Score", ""],
                                    hole=0.7,
                                    color_discrete_sequence=["#32CD32" if overall_score >= 70 else "#FFA500" if overall_score >= 50 else "#FF0000", "#EEEEEE"])
                    score_fig.update_layout(annotations=[dict(text=f"{overall_score:.1f}<br>Score", x=0.5, y=0.5, font_size=20, showarrow=False)])
                    
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.plotly_chart(score_fig)
                    
                    with col2:
                        st.write(f"**Recommendation:** {recommendation}")
                        
                        if "analyzed_opportunities" in analysis and analysis["analyzed_opportunities"]:
                            opp = analysis["analyzed_opportunities"][0]
                            
                            if "scores" in opp:
                                scores = opp["scores"]
                                score_data = {
                                    "Category": list(scores.keys()),
                                    "Score": list(scores.values())
                                }
                                score_df = pd.DataFrame(score_data)
                                score_chart = px.bar(score_df, x="Category", y="Score", 
                                                  color="Score", 
                                                  color_continuous_scale=["red", "orange", "green"],
                                                  range_color=[0, 100])
                                st.plotly_chart(score_chart)
                    
                    # Display explanations
                    if "analyzed_opportunities" in analysis and analysis["analyzed_opportunities"]:
                        opp = analysis["analyzed_opportunities"][0]
                        
                        if "explanations" in opp:
                            st.write("### Key Insights")
                            explanations = opp["explanations"]
                            
                            for factor, explanation in explanations.items():
                                st.write(f"**{factor.replace('_', ' ').title()}:** {explanation}")
                        
                        # Display requirements
                        if "requirements" in opp:
                            st.write("### Requirements")
                            requirements = opp["requirements"]
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.write("**Technologies:**")
                                for tech in requirements.get("technologies", []):
                                    st.write(f"- {tech}")
                            
                            with col2:
                                st.write("**Regulations:**")
                                for reg in requirements.get("regulations", []):
                                    st.write(f"- {reg}")
                            
                            with col3:
                                st.write("**Clearances:**")
                                for clearance in requirements.get("clearances", []):
                                    st.write(f"- {clearance}")
                    
                    # Add a button to record bid decision
                    st.write("### Record Bid Decision")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("Record Bid", key=f"bid_{opportunity.get('id')}"):
                            bid_result = record_bid_feedback(
                                opportunity_id=opportunity.get('id'),
                                bid_decision=True
                            )
                            if bid_result.get("status") == "success":
                                st.success("Bid decision recorded successfully!")
                                
                                # Clear the session state for this opportunity
                                del st.session_state[key]
                            else:
                                st.error(f"Error recording bid decision: {bid_result.get('message')}")
                    
                    with col2:
                        if st.button("Record No Bid", key=f"no_bid_{opportunity.get('id')}"):
                            no_bid_result = record_bid_feedback(
                                opportunity_id=opportunity.get('id'),
                                bid_decision=False
                            )
                            if no_bid_result.get("status") == "success":
                                st.success("No bid decision recorded successfully!")
                                
                                # Clear the session state for this opportunity
                                del st.session_state[key]
                            else:
                                st.error(f"Error recording no bid decision: {no_bid_result.get('message')}")
                else:
                    st.error(f"Error analyzing opportunity: {analysis_result.get('message')}")

# Opportunity Analysis page
elif page == "Opportunity Analysis":
    st.title("Opportunity Analysis")
    
    st.write("""
    This page allows you to analyze new contract opportunities to determine their fit with your company's capabilities 
    and make informed bid/no-bid decisions.
    """)
    
    # Form for entering opportunity details
    st.subheader("Enter Opportunity Details")
    
    with st.form("opportunity_form"):
        opp_title = st.text_input("Opportunity Title")
        opp_agency = st.text_input("Agency")
        opp_value = st.number_input("Estimated Value ($)", min_value=0.0, step=10000.0)
        opp_description = st.text_area("Opportunity Description", height=200)
        
        # Submit button
        submit_button = st.form_submit_button("Analyze Opportunity")
    
    if submit_button:
        if not opp_title or not opp_agency or not opp_description:
            st.error("Please fill out all required fields.")
        else:
            # Prepare opportunity data
            opp_data = {
                "title": opp_title,
                "agency": opp_agency,
                "value": opp_value,
                "description": opp_description
            }
            
            with st.spinner("Analyzing opportunity..."):
                analysis_result = analyze_opportunity(opp_data)
                
                if analysis_result.get("status") == "success":
                    st.success("Analysis completed successfully!")
                    
                    # Store the analysis in session state for display
                    st.session_state.current_analysis = analysis_result
                else:
                    st.error(f"Error analyzing opportunity: {analysis_result.get('message')}")
    
    # Display analysis results if available
    if "current_analysis" in st.session_state:
        analysis = st.session_state.current_analysis
        
        st.subheader("Analysis Results")
        
        # Get the opportunity details and analysis
        opportunity_id = analysis.get("opportunity_id")
        bid_analysis = analysis.get("analysis", {})
        
        # Display overall score and recommendation
        if "analyzed_opportunities" in bid_analysis and bid_analysis["analyzed_opportunities"]:
            opp = bid_analysis["analyzed_opportunities"][0]
            overall_score = opp.get("overall_score", 0)
            recommendation = opp.get("recommendation", "No recommendation")
            
            # Create columns for score and details
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Create a gauge chart for the score
                score_fig = px.pie(values=[overall_score, 100-overall_score], 
                                names=["Score", ""],
                                hole=0.7,
                                color_discrete_sequence=["#32CD32" if overall_score >= 70 else "#FFA500" if overall_score >= 50 else "#FF0000", "#EEEEEE"])
                score_fig.update_layout(annotations=[dict(text=f"{overall_score:.1f}<br>Score", x=0.5, y=0.5, font_size=20, showarrow=False)])
                st.plotly_chart(score_fig)
            
            with col2:
                st.write(f"**Recommendation:** {recommendation}")
                
                if "scores" in opp:
                    scores = opp["scores"]
                    score_data = {
                        "Category": list(scores.keys()),
                        "Score": list(scores.values())
                    }
                    score_df = pd.DataFrame(score_data)
                    score_chart = px.bar(score_df, x="Category", y="Score", 
                                      color="Score", 
                                      color_continuous_scale=["red", "orange", "green"],
                                      range_color=[0, 100])
                    st.plotly_chart(score_chart)
            
            # Display explanations
            if "explanations" in opp:
                st.subheader("Key Insights")
                explanations = opp["explanations"]
                
                for factor, explanation in explanations.items():
                    st.write(f"**{factor.replace('_', ' ').title()}:** {explanation}")
            
            # Display requirements
            if "requirements" in opp:
                st.subheader("Requirements")
                requirements = opp["requirements"]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**Technologies:**")
                    for tech in requirements.get("technologies", []):
                        st.write(f"- {tech}")
                
                with col2:
                    st.write("**Regulations:**")
                    for reg in requirements.get("regulations", []):
                        st.write(f"- {reg}")
                
                with col3:
                    st.write("**Clearances:**")
                    for clearance in requirements.get("clearances", []):
                        st.write(f"- {clearance}")
            
            # Add a section for recording bid decision
            st.subheader("Record Bid Decision")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Bid", key=f"bid_button_{opportunity_id}"):
                    bid_result = record_bid_feedback(
                        opportunity_id=opportunity_id,
                        bid_decision=True
                    )
                    if bid_result.get("status") == "success":
                        st.success("Bid decision recorded successfully!")
                    else:
                        st.error(f"Error recording bid decision: {bid_result.get('message')}")
            
            with col2:
                if st.button("No Bid", key=f"no_bid_button_{opportunity_id}"):
                    no_bid_result = record_bid_feedback(
                        opportunity_id=opportunity_id,
                        bid_decision=False
                    )
                    if no_bid_result.get("status") == "success":
                        st.success("No bid decision recorded successfully!")
                    else:
                        st.error(f"Error recording no bid decision: {no_bid_result.get('message')}")

# Competitor Analysis
elif page == "Competitor Analysis":
    st.title("Competitor Analysis")
    
    # Get competitor list
    competitors = get_competitors()
    if isinstance(competitors, list):
        competitor_names = [comp.get("name") for comp in competitors if comp.get("name")]
    else:
        competitor_names = ["TechDefense Solutions", "CloudSys Inc", "Federal IT Partners", "CyberSecure Solutions"]
    
    selected_competitor = st.selectbox("Select Competitor", competitor_names)
    
    # Get competitor ID from name
    competitor_id = None
    for comp in competitors:
        if comp.get("name") == selected_competitor:
            competitor_id = comp.get("id")
            break
    
    if not competitor_id:
        competitor_id = "comp1"  # Default for testing
    
    # Option to run analysis
    if st.button("Analyze Competitor"):
        with st.spinner(f"Analyzing {selected_competitor}..."):
            analysis_result = analyze_competitor(competitor_id)
            
            if analysis_result.get("status") == "success":
                st.success("Analysis completed successfully!")
                
                # Store analysis in session state
                st.session_state.competitor_analysis = analysis_result
            else:
                st.error(f"Error analyzing competitor: {analysis_result.get('message')}")
    
    # Display analysis if available
    if "competitor_analysis" in st.session_state:
        analysis = st.session_state.competitor_analysis
        competitor_data = analysis.get("analysis", {}).get("competitor_data", {})
        technology_data = analysis.get("analysis", {}).get("technology_data", {})
        insights = analysis.get("analysis", {}).get("insights", {})
        
        st.subheader(f"Analysis for {selected_competitor}")
        
        # Competitor contract history by agency
        if "results" in competitor_data:
            st.write("### Contract History by Agency")
            
            contract_data = pd.DataFrame(competitor_data["results"])
            if not contract_data.empty:
                fig = px.pie(contract_data, values="total_value", names="agency_name", title="Contract Value by Agency")
                st.plotly_chart(fig, use_container_width=True)
                
                # Contract history table
                st.dataframe(contract_data)
            else:
                st.info("No contract history data available for this competitor.")
        
        # Technology landscape
        if "results" in technology_data:
            st.write("### Technology Landscape")
            
            tech_data = pd.DataFrame(technology_data["results"])
            if not tech_data.empty:
                fig = px.bar(tech_data, x="technology", y="opportunity_count", 
                          color="total_value", 
                          title="Technologies by Opportunity Count and Value")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No technology landscape data available.")
        
        # Insights summary
        if insights:
            st.write("### Competitor Insights")
            
            # Win rate and bid rate
            win_rate = insights.get("win_rate", 0)
            bid_rate = insights.get("bid_rate", 0)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Win Rate", f"{win_rate:.1f}%")
            
            with col2:
                st.metric("Bid Rate", f"{bid_rate:.1f}%")
            
            # Average scores
            avg_scores = insights.get("average_scores", {})
            if avg_scores:
                st.write("#### Average Scores")
                
                score_data = {
                    "Category": list(avg_scores.keys()),
                    "Score": list(avg_scores.values())
                }
                score_df = pd.DataFrame(score_data)
                score_chart = px.bar(score_df, x="Category", y="Score", 
                                  color="Score", 
                                  color_continuous_scale=["red", "orange", "green"],
                                  range_color=[0, 100])
                st.plotly_chart(score_chart)

# Market Intelligence page
elif page == "Market Intelligence":
    st.title("Market Intelligence")
    
    st.write("""
    This page provides market intelligence analysis to help identify trends, opportunities, 
    and competitive landscape in the government contracting market.
    """)
    
    # Form for market intelligence query
    st.subheader("Market Intelligence Query")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Get list of agencies
        agencies = get_agencies()
        if isinstance(agencies, list):
            agency_names = [agency.get("name") for agency in agencies if agency.get("name")]
        else:
            agency_names = ["Department of Defense", "Department of Health & Human Services", 
                         "General Services Administration", "Department of Homeland Security"]
        
        agency_names.insert(0, "All Agencies")
        selected_agency = st.selectbox("Select Agency", agency_names)
        
        # Get agency ID from name
        selected_agency_id = None
        if selected_agency != "All Agencies":
            for agency in agencies:
                if agency.get("name") == selected_agency:
                    selected_agency_id = agency.get("id")
                    break
    
    with col2:
        # Technology selection
        tech_options = ["All Technologies", "Cloud", "Cybersecurity", "Artificial Intelligence", 
                      "Machine Learning", "DevSecOps", "Zero Trust", "5G"]
        selected_tech = st.selectbox("Select Technology", tech_options)
        
        if selected_tech == "All Technologies":
            selected_tech = None
    
    # Date range
    st.write("Date Range (Optional)")
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input("Start Date", value=None)
    
    with col2:
        end_date = st.date_input("End Date", value=None)
    
    # Convert dates to strings if selected
    start_date_str = start_date.strftime("%Y-%m-%d") if start_date else None
    end_date_str = end_date.strftime("%Y-%m-%d") if end_date else None
    
    # Run analysis button
    if st.button("Generate Market Intelligence"):
        with st.spinner("Generating market intelligence..."):
            intelligence_result = get_market_intelligence(
                agency_id=selected_agency_id,
                technology=selected_tech,
                start_date=start_date_str,
                end_date=end_date_str
            )
            
            if intelligence_result.get("status") == "success":
                st.success("Market intelligence generated successfully!")
                
                # Store in session state
                st.session_state.market_intelligence = intelligence_result
            else:
                st.error(f"Error generating market intelligence: {intelligence_result.get('message')}")
    
    # Display market intelligence if available
    if "market_intelligence" in st.session_state:
        intelligence = st.session_state.market_intelligence
        intelligence_data = intelligence.get("intelligence", {})
        
        st.subheader("Market Intelligence Results")
        
        # Agency data
        agency_data = intelligence_data.get("agency_data", {})
        agency_insights = intelligence_data.get("agency_insights", {})
        
        st.write("### Agency Insights")
        
        if "agency_insights" in agency_insights and agency_insights["agency_insights"]:
            agency_insight_data = agency_insights["agency_insights"]
            
            # Convert to DataFrame
            agencies_list = []
            for agency, data in agency_insight_data.items():
                data["agency"] = agency
                agencies_list.append(data)
            
            agency_df = pd.DataFrame(agencies_list)
            
            # Visualize agency win rates
            if not agency_df.empty and "win_rate" in agency_df.columns:
                fig = px.bar(agency_df, x="agency", y="win_rate", 
                          color="win_rate",
                          title="Agency Win Rates",
                          labels={"win_rate": "Win Rate (%)", "agency": "Agency"},
                          color_continuous_scale=["red", "orange", "green"],
                          range_color=[0, 100])
                st.plotly_chart(fig, use_container_width=True)
                
                # Opportunity count by agency
                fig = px.bar(agency_df, x="agency", y="opportunity_count",
                          title="Opportunity Count by Agency",
                          labels={"opportunity_count": "Opportunity Count", "agency": "Agency"})
                st.plotly_chart(fig, use_container_width=True)
                
                # Agency data table
                st.dataframe(agency_df)
            else:
                st.info("No agency insight data available.")
        else:
            st.info("No agency insights available.")
        
        # Technology data
        technology_data = intelligence_data.get("technology_data", {})
        technology_insights = intelligence_data.get("technology_insights", {})
        
        st.write("### Technology Insights")
        
        if "technology_insights" in technology_insights and technology_insights["technology_insights"]:
            tech_insight_data = technology_insights["technology_insights"]
            
            # Convert to DataFrame
            techs_list = []
            for tech, data in tech_insight_data.items():
                data["technology"] = tech
                techs_list.append(data)
            
            tech_df = pd.DataFrame(techs_list)
            
            # Visualize technology win rates
            if not tech_df.empty and "win_rate" in tech_df.columns:
                fig = px.bar(tech_df, x="technology", y="win_rate", 
                          color="win_rate",
                          title="Technology Win Rates",
                          labels={"win_rate": "Win Rate (%)", "technology": "Technology"},
                          color_continuous_scale=["red", "orange", "green"],
                          range_color=[0, 100])
                st.plotly_chart(fig, use_container_width=True)
                
                # Opportunity count by technology
                fig = px.bar(tech_df, x="technology", y="opportunity_count",
                          title="Opportunity Count by Technology",
                          labels={"opportunity_count": "Opportunity Count", "technology": "Technology"})
                st.plotly_chart(fig, use_container_width=True)
                
                # Technology data table
                st.dataframe(tech_df)
            else:
                st.info("No technology insight data available.")
        else:
            st.info("No technology insights available.")
        
        # Success factors
        success_factors = intelligence_data.get("success_factors", {})
        
        st.write("### Success Factors")
        
        if "key_success_factors" in success_factors and success_factors["key_success_factors"]:
            st.write("#### Key Success Factors")
            
            for factor in success_factors["key_success_factors"]:
                factor_name = factor.get("factor", "").replace("_", " ").title()
                difference = factor.get("difference", 0)
                won_avg = factor.get("won_average", 0)
                lost_avg = factor.get("lost_average", 0)
                
                st.write(f"**{factor_name}**: Difference of {difference:.1f} points (Won: {won_avg:.1f} vs Lost: {lost_avg:.1f})")
            
            # Win/loss score comparison
            if "won_scores" in success_factors and "lost_scores" in success_factors:
                won_scores = success_factors["won_scores"]
                lost_scores = success_factors["lost_scores"]
                
                if won_scores and lost_scores:
                    # Create DataFrame for comparison
                    compare_data = []
                    
                    for factor in won_scores:
                        compare_data.append({
                            "Factor": factor.replace("_", " ").title(),
                            "Won": won_scores[factor],
                            "Lost": lost_scores[factor]
                        })
                    
                    compare_df = pd.DataFrame(compare_data)
                    
                    # Create grouped bar chart
                    fig = px.bar(compare_df, x="Factor", y=["Won", "Lost"], 
                              barmode="group",
                              title="Won vs Lost Bid Scores",
                              labels={"value": "Score", "variable": "Outcome"})
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No success factor insights available.")

# Agent System page
elif page == "Agent System":
    st.title("Agent System Status")
    
    st.write("""
    This page provides information about the multi-agent system powering the Government Contracting 
    Competitive Intelligence platform.
    """)
    
    # Get agent status
    if st.button("Refresh Agent Status"):
        with st.spinner("Fetching agent status..."):
            agents_result = get_agents()
            
            if agents_result.get("status") == "success":
                st.success(f"Found {agents_result.get('agent_count', 0)} agents")
                
                # Store in session state
                st.session_state.agents = agents_result
            else:
                st.error(f"Error fetching agents: {agents_result.get('message')}")
    
    # Display agent information
    if "agents" in st.session_state:
        agents = st.session_state.agents
        agent_states = agents.get("agents", {})
        
        st.subheader("Active Agents")
        
        # Create agent cards
        for agent_id, state in agent_states.items():
            with st.expander(f"{state.get('name', 'Unknown')} ({agent_id})"):
                st.write(f"**Status:** {state.get('state', 'Unknown')}")
                st.write(f"**Last Task:** {state.get('last_task_id', 'None')}")
                
                # Option to run agent directly
                st.write("#### Run Agent")
                
                with st.form(f"run_agent_{agent_id}"):
                    params_json = st.text_area("Parameters (JSON)", "{}", key=f"params_{agent_id}")
                    
                    # Try to parse the JSON
                    try:
                        params = json.loads(params_json)
                    except json.JSONDecodeError:
                        params = {}
                    
                    run_button = st.form_submit_button("Run Agent")
                
                if run_button:
                    with st.spinner(f"Running agent {state.get('name')}..."):
                        run_result = run_agent(agent_id, params)
                        
                        if run_result.get("status") == "success":
                            st.success("Agent execution completed successfully!")
                            st.json(run_result.get("results", {}))
                        else:
                            st.error(f"Error running agent: {run_result.get('message')}")
    
    # Workflow status section
    st.subheader("Workflow Status")
    
    workflow_id = st.text_input("Enter Workflow ID")
    
    if workflow_id and st.button("Check Workflow Status"):
        with st.spinner(f"Checking status of workflow {workflow_id}..."):
            workflow_result = get_workflow_state(workflow_id)
            
            if workflow_result.get("status") == "success":
                st.success("Retrieved workflow state successfully!")
                
                # Display workflow state
                workflow_state = workflow_result.get("state", {})
                
                st.write(f"**Workflow ID:** {workflow_id}")
                st.write(f"**Status:** {workflow_state.get('status', 'Unknown')}")
                st.write(f"**Start Time:** {workflow_state.get('start_time', 'Unknown')}")
                st.write(f"**End Time:** {workflow_state.get('end_time', 'Unknown')}")
                
                # Steps
                if "steps" in workflow_state:
                    st.write("#### Workflow Steps")
                    
                    steps = workflow_state["steps"]
                    for i, step in enumerate(steps):
                        st.write(f"**Step {i+1}:** {step.get('name', 'Unknown')} ({step.get('agent_id', 'Unknown')})")
                        st.write(f"Status: {step.get('status', 'Unknown')}")
                        st.write(f"Duration: {step.get('duration', 0):.2f} seconds")
                        st.write("---")
                
                # Results
                if "results" in workflow_state:
                    st.write("#### Workflow Results")
                    st.json(workflow_state["results"])
            else:
                st.error(f"Error checking workflow status: {workflow_result.get('message')}")

# Add footer
st.markdown("---")
st.markdown("Government Contracting Competitive Intelligence System | Developed with Claude")