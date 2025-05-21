# app/monitoring_dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import metrics collector
from core.monitoring.metrics import get_metrics_collector

# Configure page
st.set_page_config(
    page_title="Agent System Monitoring",
    page_icon="📊",
    layout="wide"
)

# API connection settings
API_URL = "http://api:8000"
API_KEY = os.environ.get("API_KEY", "dev_key")
HEADERS = {"X-API-Key": API_KEY}

# Function to load metrics data
@st.cache_data(ttl=60)  # Cache for 60 seconds
def load_metrics_data():
    metrics = get_metrics_collector()
    return metrics.get_metrics()

# Function to load agent states from API
@st.cache_data(ttl=30)  # Cache for 30 seconds
def load_agent_states():
    try:
        response = requests.get(
            f"{API_URL}/agents",
            headers=HEADERS
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error loading agent states: {str(e)}")
        return {"agents": {}}

# Function to load recent workflows from API
@st.cache_data(ttl=60)  # Cache for 60 seconds
def load_recent_workflows():
    try:
        # This would be a real endpoint in production
        # For now, we'll return mock data
        return {
            "workflows": [
                {
                    "id": "data_collection_20230615_123456",
                    "type": "data_collection",
                    "status": "completed",
                    "start_time": (datetime.now() - timedelta(minutes=30)).isoformat(),
                    "end_time": (datetime.now() - timedelta(minutes=29)).isoformat(),
                    "duration": 60.5,
                    "steps": 3
                },
                {
                    "id": "opportunity_analysis_20230615_123457",
                    "type": "opportunity_analysis",
                    "status": "completed",
                    "start_time": (datetime.now() - timedelta(minutes=25)).isoformat(),
                    "end_time": (datetime.now() - timedelta(minutes=24)).isoformat(),
                    "duration": 45.2,
                    "steps": 3
                },
                {
                    "id": "competitor_analysis_20230615_123458",
                    "type": "competitor_analysis",
                    "status": "failed",
                    "start_time": (datetime.now() - timedelta(minutes=20)).isoformat(),
                    "end_time": (datetime.now() - timedelta(minutes=19)).isoformat(),
                    "duration": 15.8,
                    "steps": 2
                },
                {
                    "id": "market_intelligence_20230615_123459",
                    "type": "market_intelligence",
                    "status": "completed",
                    "start_time": (datetime.now() - timedelta(minutes=15)).isoformat(),
                    "end_time": (datetime.now() - timedelta(minutes=14)).isoformat(),
                    "duration": 75.3,
                    "steps": 5
                },
                {
                    "id": "data_collection_20230615_123460",
                    "type": "data_collection",
                    "status": "completed",
                    "start_time": (datetime.now() - timedelta(minutes=10)).isoformat(),
                    "end_time": (datetime.now() - timedelta(minutes=9)).isoformat(),
                    "duration": 58.7,
                    "steps": 3
                }
            ]
        }
    except Exception as e:
        st.error(f"Error loading recent workflows: {str(e)}")
        return {"workflows": []}

# Function to load recent errors from API
@st.cache_data(ttl=60)  # Cache for 60 seconds
def load_recent_errors():
    try:
        # This would be a real endpoint in production
        # For now, we'll return mock data
        return {
            "errors": [
                {
                    "timestamp": (datetime.now() - timedelta(minutes=20)).isoformat(),
                    "source": "agent:knowledge_graph_agent",
                    "error_type": "ConnectionError",
                    "message": "Failed to connect to Neo4j database",
                    "details": {
                        "task_id": "task_123458",
                        "operation": "query"
                    }
                },
                {
                    "timestamp": (datetime.now() - timedelta(minutes=18)).isoformat(),
                    "source": "workflow:competitor_analysis",
                    "error_type": "ValueError",
                    "message": "No data available for competitor comp123",
                    "details": {
                        "workflow_id": "competitor_analysis_20230615_123458",
                        "step": 1
                    }
                },
                {
                    "timestamp": (datetime.now() - timedelta(minutes=5)).isoformat(),
                    "source": "api",
                    "error_type": "HTTPException",
                    "message": "Not Found",
                    "details": {
                        "endpoint": "/workflow-state/nonexistent_workflow",
                        "method": "GET",
                        "status_code": 404
                    }
                }
            ]
        }
    except Exception as e:
        st.error(f"Error loading recent errors: {str(e)}")
        return {"errors": []}

# Sidebar navigation
st.sidebar.title("Monitoring Dashboard")
page = st.sidebar.radio(
    "Select a page",
    ["System Overview", "Agent Performance", "Workflow Analysis", "API Performance", "Error Tracking"]
)

# Load data
metrics_data = load_metrics_data()
agent_states = load_agent_states()
recent_workflows = load_recent_workflows()
recent_errors = load_recent_errors()

# System Overview page
if page == "System Overview":
    st.title("System Overview")
    
    # Last update time
    st.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # System health metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        agent_success_rate = metrics_data["agent_execution"]["success_rate"]
        st.metric(
            "Agent Success Rate", 
            f"{agent_success_rate:.1f}%",
            delta="+0.5%" if agent_success_rate > 0 else "0%",
            delta_color="normal"
        )
    
    with col2:
        workflow_completion_rate = metrics_data["workflow_execution"]["completion_rate"]
        st.metric(
            "Workflow Completion", 
            f"{workflow_completion_rate:.1f}%",
            delta="-1.2%" if workflow_completion_rate < 100 else "0%",
            delta_color="normal"
        )
    
    with col3:
        api_success_rate = metrics_data["api_requests"]["success_rate"]
        st.metric(
            "API Success Rate", 
            f"{api_success_rate:.1f}%",
            delta="+0.3%" if api_success_rate > 0 else "0%",
            delta_color="normal"
        )
    
    with col4:
        error_count = metrics_data["error_counts"]["total_errors"]
        st.metric(
            "Total Errors", 
            error_count,
            delta="+3" if error_count > 0 else "0",
            delta_color="inverse"
        )
    
    # Agent status overview
    st.subheader("Agent Status")
    
    # Create a DataFrame for agent states
    agent_data = []
    for agent_id, state in agent_states.get("agents", {}).items():
        agent_data.append({
            "Agent ID": agent_id,
            "Name": state.get("name", "Unknown"),
            "Status": state.get("state", "Unknown"),
            "Last Task": state.get("last_task_id", "None")
        })
    
    agent_df = pd.DataFrame(agent_data)
    if not agent_df.empty:
        # Add status color
        status_colors = {
            "idle": "#32CD32",  # Green
            "running": "#FFA500",  # Orange
            "error": "#FF0000",  # Red
            "initialized": "#1E90FF"  # Blue
        }
        
        # Create a custom styler to color the Status column
        def status_color(val):
            color = status_colors.get(val.lower(), "#808080")  # Gray for unknown
            return f'background-color: {color}; color: white'
        
        agent_df_styled = agent_df.style.applymap(status_color, subset=['Status'])
        st.dataframe(agent_df_styled, use_container_width=True)
    else:
        st.info("No agent data available")
    
    # Recent workflows
    st.subheader("Recent Workflows")
    
    workflow_data = []
    for workflow in recent_workflows.get("workflows", []):
        workflow_data.append({
            "ID": workflow.get("id"),
            "Type": workflow.get("type", "Unknown"),
            "Status": workflow.get("status", "Unknown"),
            "Start Time": workflow.get("start_time"),
            "Duration (s)": workflow.get("duration", 0),
            "Steps": workflow.get("steps", 0)
        })
    
    workflow_df = pd.DataFrame(workflow_data)
    if not workflow_df.empty:
        # Convert times to datetime for better display
        if "Start Time" in workflow_df.columns:
            workflow_df["Start Time"] = pd.to_datetime(workflow_df["Start Time"])
        
        # Add status color
        status_colors = {
            "completed": "#32CD32",  # Green
            "running": "#FFA500",  # Orange
            "failed": "#FF0000"  # Red
        }
        
        # Create a custom styler to color the Status column
        def status_color(val):
            color = status_colors.get(val.lower(), "#808080")  # Gray for unknown
            return f'background-color: {color}; color: white'
        
        workflow_df_styled = workflow_df.style.applymap(status_color, subset=['Status'])
        st.dataframe(workflow_df_styled, use_container_width=True)
    else:
        st.info("No workflow data available")
    
    # Recent errors
    st.subheader("Recent Errors")
    
    error_data = []
    for error in recent_errors.get("errors", []):
        error_data.append({
            "Timestamp": error.get("timestamp"),
            "Source": error.get("source", "Unknown"),
            "Error Type": error.get("error_type", "Unknown"),
            "Message": error.get("message", "")
        })
    
    error_df = pd.DataFrame(error_data)
    if not error_df.empty:
        # Convert times to datetime for better display
        if "Timestamp" in error_df.columns:
            error_df["Timestamp"] = pd.to_datetime(error_df["Timestamp"])
        
        st.dataframe(error_df, use_container_width=True)
    else:
        st.info("No error data available")

# Agent Performance page
elif page == "Agent Performance":
    st.title("Agent Performance")
    
    # Last update time
    st.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Agent execution metrics
    st.subheader("Agent Execution Metrics")
    
    # Create a DataFrame for agent metrics
    agent_metrics = metrics_data["agent_execution"]["agents"]
    
    if agent_metrics:
        agent_df = pd.DataFrame(agent_metrics)
        
        # Create bar chart of executions by agent
        fig_executions = px.bar(
            agent_df, 
            x="agent_id", 
            y="executions",
            title="Executions by Agent",
            color="agent_id"
        )
        st.plotly_chart(fig_executions, use_container_width=True)
        
        # Create bar chart of average duration by agent
        fig_duration = px.bar(
            agent_df, 
            x="agent_id", 
            y="avg_duration",
            title="Average Duration by Agent (seconds)",
            color="agent_id"
        )
        st.plotly_chart(fig_duration, use_container_width=True)
        
        # Create bar chart of success rate by agent
        fig_success = px.bar(
            agent_df, 
            x="agent_id", 
            y="success_rate",
            title="Success Rate by Agent (%)",
            color="agent_id",
            color_continuous_scale=["red", "yellow", "green"],
            range_color=[0, 100]
        )
        st.plotly_chart(fig_success, use_container_width=True)
        
        # Raw data table
        st.subheader("Agent Metrics Data")
        st.dataframe(agent_df, use_container_width=True)
    else:
        st.info("No agent metrics available")
    
    # Agent task history (from raw cache)
    st.subheader("Agent Task History")
    
    # Get the agent execution cache from metrics data
    agent_cache = metrics_data.get("raw_cache", {}).get("agent_execution", {})
    
    if agent_cache:
        # Select an agent to view
        agent_ids = list(agent_cache.keys())
        selected_agent = st.selectbox("Select Agent", agent_ids)
        
        if selected_agent and selected_agent in agent_cache:
            agent_tasks = agent_cache[selected_agent].get("recent_tasks", [])
            
            if agent_tasks:
                task_df = pd.DataFrame(agent_tasks)
                
                # Convert timestamp to datetime
                if "timestamp" in task_df.columns:
                    task_df["timestamp"] = pd.to_datetime(task_df["timestamp"])
                
                # Sort by timestamp descending
                task_df = task_df.sort_values(by="timestamp", ascending=False)
                
                # Add status color
                status_colors = {
                    "success": "#32CD32",  # Green
                    "error": "#FF0000"  # Red
                }
                
                # Create a custom styler to color the Status column
                def status_color(val):
                    color = status_colors.get(val.lower(), "#808080")  # Gray for unknown
                    return f'background-color: {color}; color: white'
                
                task_df_styled = task_df.style.applymap(status_color, subset=['status'])
                st.dataframe(task_df_styled, use_container_width=True)
            else:
                st.info("No task history available for this agent")
    else:
        st.info("No agent task history available")

# Workflow Analysis page
elif page == "Workflow Analysis":
    st.title("Workflow Analysis")
    
    # Last update time
    st.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Workflow execution metrics
    st.subheader("Workflow Execution Metrics")
    
    # Create a DataFrame for workflow metrics
    workflow_metrics = metrics_data["workflow_execution"]["workflow_types"]
    
    if workflow_metrics:
        workflow_df = pd.DataFrame(workflow_metrics)
        
        # Create bar chart of executions by workflow type
        fig_executions = px.bar(
            workflow_df, 
            x="workflow_type", 
            y="executions",
            title="Executions by Workflow Type",
            color="workflow_type"
        )
        st.plotly_chart(fig_executions, use_container_width=True)
        
        # Create bar chart of average duration by workflow type
        fig_duration = px.bar(
            workflow_df, 
            x="workflow_type", 
            y="avg_duration",
            title="Average Duration by Workflow Type (seconds)",
            color="workflow_type"
        )
        st.plotly_chart(fig_duration, use_container_width=True)
        
        # Create bar chart of completion rate by workflow type
        fig_completion = px.bar(
            workflow_df, 
            x="workflow_type", 
            y="completion_rate",
            title="Completion Rate by Workflow Type (%)",
            color="workflow_type",
            color_continuous_scale=["red", "yellow", "green"],
            range_color=[0, 100]
        )
        st.plotly_chart(fig_completion, use_container_width=True)
        
        # Raw data table
        st.subheader("Workflow Metrics Data")
        st.dataframe(workflow_df, use_container_width=True)
    else:
        st.info("No workflow metrics available")
    
    # Workflow history (from raw cache)
    st.subheader("Workflow History")
    
    # Get the workflow execution cache from metrics data
    workflow_cache = metrics_data.get("raw_cache", {}).get("workflow_execution", {})
    
    if workflow_cache:
        # Select a workflow type to view
        workflow_types = list(workflow_cache.keys())
        selected_workflow = st.selectbox("Select Workflow Type", workflow_types)
        
        if selected_workflow and selected_workflow in workflow_cache:
            workflows = workflow_cache[selected_workflow].get("recent_workflows", [])
            
            if workflows:
                workflow_df = pd.DataFrame(workflows)
                
                # Convert timestamp to datetime
                if "timestamp" in workflow_df.columns:
                    workflow_df["timestamp"] = pd.to_datetime(workflow_df["timestamp"])
                
                # Sort by timestamp descending
                workflow_df = workflow_df.sort_values(by="timestamp", ascending=False)
                
                # Add status color
                status_colors = {
                    "completed": "#32CD32",  # Green
                    "failed": "#FF0000",  # Red
                    "running": "#FFA500"  # Orange
                }
                
                # Create a custom styler to color the Status column
                def status_color(val):
                    color = status_colors.get(val.lower(), "#808080")  # Gray for unknown
                    return f'background-color: {color}; color: white'
                
                workflow_df_styled = workflow_df.style.applymap(status_color, subset=['status'])
                st.dataframe(workflow_df_styled, use_container_width=True)
            else:
                st.info("No workflow history available for this type")
    else:
        st.info("No workflow history available")

# API Performance page
elif page == "API Performance":
    st.title("API Performance")
    
    # Last update time
    st.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # API request metrics
    st.subheader("API Request Metrics")
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_requests = metrics_data["api_requests"]["total_requests"]
        st.metric("Total Requests", total_requests)
    
    with col2:
        avg_duration = metrics_data["api_requests"]["avg_duration"]
        st.metric("Average Duration (s)", f"{avg_duration:.3f}")
    
    with col3:
        success_rate = metrics_data["api_requests"]["success_rate"]
        st.metric("Success Rate (%)", f"{success_rate:.1f}%")
    
    # Create a DataFrame for API endpoint metrics
    endpoint_metrics = metrics_data["api_requests"]["endpoints"]
    
    if endpoint_metrics:
        endpoint_df = pd.DataFrame(endpoint_metrics)
        
        # Create composite method + endpoint column
        endpoint_df["method_endpoint"] = endpoint_df["method"] + " " + endpoint_df["endpoint"]
        
        # Create bar chart of requests by endpoint
        fig_requests = px.bar(
            endpoint_df, 
            x="method_endpoint", 
            y="count",
            title="Requests by Endpoint",
            color="method"
        )
        st.plotly_chart(fig_requests, use_container_width=True)
        
        # Create bar chart of average duration by endpoint
        fig_duration = px.bar(
            endpoint_df, 
            x="method_endpoint", 
            y="avg_duration",
            title="Average Duration by Endpoint (seconds)",
            color="method"
        )
        st.plotly_chart(fig_duration, use_container_width=True)
        
        # Create a range chart for min, avg, max duration
        fig_range = go.Figure()
        
        # Add min duration
        fig_range.add_trace(go.Bar(
            name="Min Duration",
            x=endpoint_df["method_endpoint"],
            y=endpoint_df["min_duration"],
            marker_color="green"
        ))
        
        # Add avg duration
        fig_range.add_trace(go.Bar(
            name="Avg Duration",
            x=endpoint_df["method_endpoint"],
            y=endpoint_df["avg_duration"],
            marker_color="blue"
        ))
        
        # Add max duration
        fig_range.add_trace(go.Bar(
            name="Max Duration",
            x=endpoint_df["method_endpoint"],
            y=endpoint_df["max_duration"],
            marker_color="red"
        ))
        
        # Set the title and layout
        fig_range.update_layout(
            title="Duration Range by Endpoint (seconds)",
            barmode="group"
        )
        
        st.plotly_chart(fig_range, use_container_width=True)
        
        # Raw data table
        st.subheader("API Endpoint Metrics Data")
        st.dataframe(endpoint_df, use_container_width=True)
    else:
        st.info("No API endpoint metrics available")

# Error Tracking page
elif page == "Error Tracking":
    st.title("Error Tracking")
    
    # Last update time
    st.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Error metrics summary
    st.subheader("Error Summary")
    
    total_errors = metrics_data["error_counts"]["total_errors"]
    st.metric("Total Errors", total_errors)
    
    # Error sources breakdown
    error_sources = metrics_data["error_counts"]["sources"]
    
    if error_sources:
        # Create a DataFrame for error sources
        source_data = []
        for source, data in error_sources.items():
            source_data.append({
                "Source": source,
                "Error Count": data["total"]
            })
        
        source_df = pd.DataFrame(source_data)
        
        # Create pie chart of errors by source
        fig_sources = px.pie(
            source_df,
            values="Error Count",
            names="Source",
            title="Errors by Source"
        )
        st.plotly_chart(fig_sources, use_container_width=True)
        
        # Error types by source
        st.subheader("Error Types by Source")
        
        # Select a source to view
        source_list = list(error_sources.keys())
        selected_source = st.selectbox("Select Source", source_list)
        
        if selected_source and selected_source in error_sources:
            error_types = error_sources[selected_source]["types"]
            
            if error_types:
                # Create a DataFrame for error types
                type_data = []
                for error_type, count in error_types.items():
                    type_data.append({
                        "Error Type": error_type,
                        "Count": count
                    })
                
                type_df = pd.DataFrame(type_data)
                
                # Create bar chart of errors by type
                fig_types = px.bar(
                    type_df,
                    x="Error Type",
                    y="Count",
                    title=f"Error Types for {selected_source}",
                    color="Error Type"
                )
                st.plotly_chart(fig_types, use_container_width=True)
                
                # Raw data table
                st.dataframe(type_df, use_container_width=True)
            else:
                st.info("No error type data available for this source")
    else:
        st.info("No error source data available")
    
    # Recent errors
    st.subheader("Recent Errors")
    
    error_data = []
    for error in recent_errors.get("errors", []):
        error_data.append({
            "Timestamp": error.get("timestamp"),
            "Source": error.get("source", "Unknown"),
            "Error Type": error.get("error_type", "Unknown"),
            "Message": error.get("message", "")
        })
    
    error_df = pd.DataFrame(error_data)
    if not error_df.empty:
        # Convert times to datetime for better display
        if "Timestamp" in error_df.columns:
            error_df["Timestamp"] = pd.to_datetime(error_df["Timestamp"])
        
        # Display the error details
        for i, error in enumerate(recent_errors.get("errors", [])):
            with st.expander(f"{error.get('error_type')}: {error.get('message')}"):
                st.write(f"**Source:** {error.get('source', 'Unknown')}")
                st.write(f"**Timestamp:** {error.get('timestamp')}")
                st.write(f"**Error Type:** {error.get('error_type', 'Unknown')}")
                st.write(f"**Message:** {error.get('message', '')}")
                
                # Display details if available
                details = error.get("details", {})
                if details:
                    st.subheader("Details")
                    for key, value in details.items():
                        st.write(f"**{key}:** {value}")
    else:
        st.info("No recent error data available")

# Add footer
st.markdown("---")
st.markdown("Government Contracting Competitive Intelligence System | Monitoring Dashboard")