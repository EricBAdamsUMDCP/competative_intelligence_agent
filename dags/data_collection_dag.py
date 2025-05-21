# dags/data_collection_dag.py
import os
import sys
import requests
import json
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator

# API connection settings
API_URL = os.environ.get("API_URL", "http://api:8000")
API_KEY = os.environ.get("API_KEY", "dev_key")
HEADERS = {"X-API-Key": API_KEY}

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def run_collection_workflow(**kwargs):
    """Run a data collection workflow."""
    # Create the workflow parameters
    params = {
        "id": f"airflow_collection_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "collector_type": "core.collectors.sam_gov.SamGovCollector",
        "collector_config": {
            "api_key": os.environ.get("SAM_GOV_API_KEY", "DEMO_KEY"),
            # Collect data from the last day
            "published_since": (datetime.now() - timedelta(days=1)).isoformat()
        }
    }
    
    # Run the workflow
    response = requests.post(
        f"{API_URL}/run-workflow",
        json={
            "workflow_type": "data_collection",
            "params": params
        },
        headers=HEADERS
    )
    
    # Check the response
    if response.status_code == 200:
        result = response.json()
        workflow_id = result.get("workflow_id")
        print(f"Started workflow: {workflow_id}")
        
        # Push the workflow ID to XCom for other tasks to use
        kwargs['ti'].xcom_push(key='workflow_id', value=workflow_id)
        
        return workflow_id
    else:
        print(f"Error starting workflow: {response.text}")
        raise Exception(f"Error starting workflow: {response.text}")

def check_workflow_state(**kwargs):
    """Check the state of a workflow."""
    # Get the workflow ID from XCom
    ti = kwargs['ti']
    workflow_id = ti.xcom_pull(task_ids='run_collection', key='workflow_id')
    
    if not workflow_id:
        raise Exception("No workflow ID found in XCom")
    
    # Check the workflow state
    response = requests.get(
        f"{API_URL}/workflow-state/{workflow_id}",
        headers=HEADERS
    )
    
    # Check the response
    if response.status_code == 200:
        result = response.json()
        state = result.get("state", {})
        status = state.get("status")
        
        print(f"Workflow {workflow_id} status: {status}")
        
        if status == "completed":
            return True
        elif status == "failed":
            raise Exception(f"Workflow {workflow_id} failed")
        else:
            # Still running
            return False
    else:
        print(f"Error checking workflow state: {response.text}")
        raise Exception(f"Error checking workflow state: {response.text}")

def run_market_intelligence(**kwargs):
    """Run a market intelligence workflow."""
    # Get the workflow ID from XCom to ensure this runs after collection
    ti = kwargs['ti']
    collection_workflow_id = ti.xcom_pull(task_ids='run_collection', key='workflow_id')
    
    if not collection_workflow_id:
        raise Exception("No collection workflow ID found in XCom")
    
    # Create the workflow parameters for market intelligence
    params = {
        "id": f"airflow_market_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "agency_id": None,  # All agencies
        "technology": None,  # All technologies
        "time_period": {
            "start_date": (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
            "end_date": datetime.now().strftime("%Y-%m-%d")
        }
    }
    
    # Run the workflow
    response = requests.post(
        f"{API_URL}/run-workflow",
        json={
            "workflow_type": "market_intelligence",
            "params": params
        },
        headers=HEADERS
    )
    
    # Check the response
    if response.status_code == 200:
        result = response.json()
        workflow_id = result.get("workflow_id")
        print(f"Started market intelligence workflow: {workflow_id}")
        
        # Push the workflow ID to XCom for other tasks to use
        kwargs['ti'].xcom_push(key='market_workflow_id', value=workflow_id)
        
        return workflow_id
    else:
        print(f"Error starting market intelligence workflow: {response.text}")
        raise Exception(f"Error starting market intelligence workflow: {response.text}")

# Define the DAG
with DAG(
    'daily_data_collection',
    default_args=default_args,
    description='Daily data collection from SAM.gov',
    schedule_interval='0 6 * * *',  # Run at 6:00 AM every day
    start_date=datetime(2023, 6, 1),
    catchup=False,
    tags=['govcon', 'data_collection'],
) as dag:
    
    # Start task
    start_task = DummyOperator(
        task_id='start',
    )
    
    # Run data collection workflow
    run_collection_task = PythonOperator(
        task_id='run_collection',
        python_callable=run_collection_workflow,
        provide_context=True,
    )
    
    # Check workflow state
    check_workflow_task = PythonOperator(
        task_id='check_workflow',
        python_callable=check_workflow_state,
        provide_context=True,
    )
    
    # Run market intelligence
    run_market_intelligence_task = PythonOperator(
        task_id='run_market_intelligence',
        python_callable=run_market_intelligence,
        provide_context=True,
    )
    
    # End task
    end_task = DummyOperator(
        task_id='end',
    )
    
    # Define task dependencies
    start_task >> run_collection_task >> check_workflow_task >> run_market_intelligence_task >> end_task

# Weekly analysis DAG
# dags/weekly_analysis_dag.py

def run_competitor_analysis(**kwargs):
    """Run competitor analysis for top competitors."""
    # Get top competitors (could be from a configuration file or database)
    top_competitors = [
        {"id": "comp1", "name": "TechGov Solutions"},
        {"id": "comp2", "name": "Federal Systems Inc"},
        {"id": "comp3", "name": "Government IT Partners"}
    ]
    
    workflow_ids = []
    
    # Run analysis for each competitor
    for competitor in top_competitors:
        # Create the workflow parameters
        params = {
            "id": f"airflow_competitor_{competitor['id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "competitor_id": competitor["id"],
            "include_technologies": True,
            "include_agencies": True
        }
        
        # Run the workflow
        response = requests.post(
            f"{API_URL}/run-workflow",
            json={
                "workflow_type": "competitor_analysis",
                "params": params
            },
            headers=HEADERS
        )
        
        # Check the response
        if response.status_code == 200:
            result = response.json()
            workflow_id = result.get("workflow_id")
            print(f"Started competitor analysis workflow for {competitor['name']}: {workflow_id}")
            workflow_ids.append(workflow_id)
        else:
            print(f"Error starting competitor analysis workflow for {competitor['name']}: {response.text}")
    
    # Push the workflow IDs to XCom
    kwargs['ti'].xcom_push(key='competitor_workflow_ids', value=workflow_ids)
    
    return workflow_ids

def run_feedback_model_update(**kwargs):
    """Run a feedback model update workflow."""
    # Create the workflow parameters
    params = {
        "id": f"airflow_feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "operation": "update_model",
        "force": True  # Force update even if not enough new data
    }
    
    # Run the workflow
    response = requests.post(
        f"{API_URL}/agents/feedback_learning_agent/run",
        json=params,
        headers=HEADERS
    )
    
    # Check the response
    if response.status_code == 200:
        result = response.json()
        task_id = result.get("task_id")
        print(f"Started feedback model update: {task_id}")
        
        # Push the task ID to XCom
        kwargs['ti'].xcom_push(key='feedback_task_id', value=task_id)
        
        return task_id
    else:
        print(f"Error starting feedback model update: {response.text}")
        raise Exception(f"Error starting feedback model update: {response.text}")

# Define the weekly analysis DAG
with DAG(
    'weekly_analysis',
    default_args=default_args,
    description='Weekly analysis of competitors and model updates',
    schedule_interval='0 8 * * 1',  # Run at 8:00 AM every Monday
    start_date=datetime(2023, 6, 1),
    catchup=False,
    tags=['govcon', 'analysis'],
) as weekly_dag:
    
    # Start task
    start_task = DummyOperator(
        task_id='start',
    )
    
    # Run competitor analysis
    run_competitor_analysis_task = PythonOperator(
        task_id='run_competitor_analysis',
        python_callable=run_competitor_analysis,
        provide_context=True,
    )
    
    # Run feedback model update
    run_feedback_model_update_task = PythonOperator(
        task_id='run_feedback_model_update',
        python_callable=run_feedback_model_update,
        provide_context=True,
    )
    
    # End task
    end_task = DummyOperator(
        task_id='end',
    )
    
    # Define task dependencies
    start_task >> run_competitor_analysis_task >> run_feedback_model_update_task >> end_task

# Monthly opportunity analysis DAG
# dags/monthly_opportunity_analysis_dag.py

def get_active_opportunities(**kwargs):
    """Get all active opportunities."""
    # Search for active opportunities
    response = requests.post(
        f"{API_URL}/search",
        json={
            "query": "active",
            "filters": {
                "status": "active",
                "value_min": 1000000  # Only analyze opportunities over $1M
            }
        },
        headers=HEADERS
    )
    
    # Check the response
    if response.status_code == 200:
        opportunities = response.json()
        print(f"Found {len(opportunities)} active opportunities")
        
        # Push the opportunities to XCom
        kwargs['ti'].xcom_push(key='active_opportunities', value=opportunities)
        
        return opportunities
    else:
        print(f"Error searching for active opportunities: {response.text}")
        raise Exception(f"Error searching for active opportunities: {response.text}")

def analyze_opportunities(**kwargs):
    """Analyze all active opportunities."""
    # Get the opportunities from XCom
    ti = kwargs['ti']
    opportunities = ti.xcom_pull(task_ids='get_active_opportunities', key='active_opportunities')
    
    if not opportunities:
        print("No active opportunities found")
        return []
    
    workflow_ids = []
    
    # Run analysis for each opportunity
    for opportunity in opportunities:
        # Create the opportunity data
        opportunity_data = {
            "opportunity_id": opportunity.get("id"),
            "title": opportunity.get("title"),
            "agency": opportunity.get("agency", "Unknown"),
            "value": float(opportunity.get("value", 0)),
            "description": opportunity.get("description", "")
        }
        
        # Run the workflow
        response = requests.post(
            f"{API_URL}/analyze-opportunity",
            json=opportunity_data,
            headers=HEADERS
        )
        
        # Check the response
        if response.status_code == 200:
            result = response.json()
            workflow_id = result.get("workflow_id")
            print(f"Started opportunity analysis workflow for {opportunity.get('title')}: {workflow_id}")
            workflow_ids.append(workflow_id)
        else:
            print(f"Error starting opportunity analysis workflow for {opportunity.get('title')}: {response.text}")
    
    # Push the workflow IDs to XCom
    kwargs['ti'].xcom_push(key='opportunity_workflow_ids', value=workflow_ids)
    
    return workflow_ids

# Define the monthly opportunity analysis DAG
with DAG(
    'monthly_opportunity_analysis',
    default_args=default_args,
    description='Monthly analysis of all active opportunities',
    schedule_interval='0 8 1 * *',  # Run at 8:00 AM on the 1st day of each month
    start_date=datetime(2023, 6, 1),
    catchup=False,
    tags=['govcon', 'analysis'],
) as monthly_dag:
    
    # Start task
    start_task = DummyOperator(
        task_id='start',
    )
    
    # Get active opportunities
    get_active_opportunities_task = PythonOperator(
        task_id='get_active_opportunities',
        python_callable=get_active_opportunities,
        provide_context=True,
    )
    
    # Analyze opportunities
    analyze_opportunities_task = PythonOperator(
        task_id='analyze_opportunities',
        python_callable=analyze_opportunities,
        provide_context=True,
    )
    
    # End task
    end_task = DummyOperator(
        task_id='end',
    )
    
    # Define task dependencies
    start_task >> get_active_opportunities_task >> analyze_opportunities_task >> end_task