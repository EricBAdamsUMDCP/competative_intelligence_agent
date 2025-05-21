# scripts/initialize_agents.py
import os
import sys
import json
import logging
import asyncio
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agents.factory import AgentFactory
from core.agents.manager import AgentManager
from core.agents.orchestrator import AgentOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("agent.init")

# Sample company profile for testing
SAMPLE_COMPANY_PROFILE = {
    "name": "TechGov Solutions",
    "technologies": [
        "cloud", "cybersecurity", "artificial intelligence", 
        "machine learning", "zero trust", "DevSecOps"
    ],
    "regulations": [
        "NIST", "CMMC", "FedRAMP", "FISMA", "HIPAA"
    ],
    "clearances": [
        "Top Secret", "TS/SCI", "Secret"
    ],
    "past_contracts": [
        {
            "agency": "Department of Defense",
            "title": "Cloud Migration Services",
            "value": 5600000,
            "performance_rating": 4.8,
            "technologies": ["cloud", "DevSecOps", "cybersecurity"]
        },
        {
            "agency": "Department of Health and Human Services",
            "title": "AI-Driven Health Data Analytics",
            "value": 3200000,
            "performance_rating": 4.5,
            "technologies": ["artificial intelligence", "machine learning"]
        },
        {
            "agency": "Department of Homeland Security",
            "title": "Cybersecurity Operations Support",
            "value": 4800000,
            "performance_rating": 4.7,
            "technologies": ["cybersecurity", "zero trust"]
        }
    ],
    "agency_relationships": [
        {
            "agency": "Department of Defense",
            "strength": 4,
            "contacts": 5
        },
        {
            "agency": "Department of Health and Human Services",
            "strength": 3,
            "contacts": 3
        },
        {
            "agency": "Department of Homeland Security",
            "strength": 4,
            "contacts": 4
        }
    ],
    "technology_strengths": [
        "cybersecurity", "zero trust", "cloud"
    ],
    "min_contract_value": 500000,
    "target_contract_value": 5000000,
    "max_contract_value": 25000000,
    "strategic_agencies": [
        "Department of Defense", "Department of Homeland Security", "General Services Administration"
    ],
    "strategic_technologies": [
        "zero trust", "artificial intelligence", "DevSecOps"
    ]
}

# Agent configurations for initialization
AGENT_CONFIGS = [
    {
        "type": "collection_agent",
        "id": "sam_gov_agent",
        "name": "SAM.gov Collection Agent",
        "config": {
            "collector_type": "core.collectors.sam_gov.SamGovCollector",
            "collector_config": {
                "api_key": os.environ.get("SAM_GOV_API_KEY", "DEMO_KEY")
            }
        }
    },
    {
        "type": "entity_analysis_agent",
        "id": "entity_analysis_agent",
        "name": "Entity Analysis Agent",
        "config": {
            "model_name": "en_core_web_lg"
        }
    },
    {
        "type": "knowledge_graph_agent",
        "id": "knowledge_graph_agent",
        "name": "Knowledge Graph Agent",
        "config": {
            "uri": os.environ.get("NEO4J_URI", "bolt://neo4j:7687"),
            "user": os.environ.get("NEO4J_USER", "neo4j"),
            "password": os.environ.get("NEO4J_PASSWORD", "password")
        }
    },
    {
        "type": "bid_analysis_agent",
        "id": "bid_analysis_agent",
        "name": "Bid Analysis Agent",
        "config": {
            "company_profile": SAMPLE_COMPANY_PROFILE
        }
    },
    {
        "type": "feedback_learning_agent",
        "id": "feedback_learning_agent",
        "name": "Feedback Learning Agent",
        "config": {
            "storage_path": "data/feedback"
        }
    }
]

async def run_test_workflow(manager: AgentManager):
    """Run a test workflow to verify the agent system is working."""
    try:
        logger.info("Running test data collection workflow...")
        
        result = await manager.run_workflow(
            workflow_type="data_collection",
            params={
                "id": "test",
                "collector_type": "core.collectors.sam_gov.SamGovCollector"
            }
        )
        
        logger.info(f"Test workflow completed with status: {result.get('status', 'unknown')}")
        
        # Check workflow results
        workflow_id = result.get("workflow_id")
        if workflow_id:
            workflow_state = manager.get_workflow_state(workflow_id)
            logger.info(f"Workflow execution details: {json.dumps(workflow_state, indent=2)}")
        
        return result
    except Exception as e:
        logger.error(f"Error running test workflow: {str(e)}")
        return {"status": "error", "error": str(e)}

def save_company_profile():
    """Save the sample company profile to a JSON file."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    with open(data_dir / "company_profile.json", "w") as f:
        json.dump(SAMPLE_COMPANY_PROFILE, f, indent=2)
    
    logger.info("Saved sample company profile to data/company_profile.json")

def initialize_agents():
    """Initialize the agent system."""
    try:
        # Create data directories
        os.makedirs("data/feedback", exist_ok=True)
        
        # Save company profile
        save_company_profile()
        
        # Create agent manager with file-based configuration
        manager_config = {
            "company_profile": SAMPLE_COMPANY_PROFILE,
            "agent_configs": AGENT_CONFIGS
        }
        
        with open("data/agent_config.json", "w") as f:
            json.dump(manager_config, f, indent=2)
        
        logger.info("Saved agent configuration to data/agent_config.json")
        
        # Create agent manager
        manager = AgentManager(config_path="data/agent_config.json")
        
        # Create agents from configuration
        agents = AgentFactory.create_agents_from_config(AGENT_CONFIGS)
        
        # Register agents with the manager
        for agent_id, agent in agents.items():
            manager.register_agent(agent_id, agent)
        
        logger.info(f"Initialized {len(agents)} agents")
        
        # Run a test workflow
        asyncio.run(run_test_workflow(manager))
        
        logger.info("Agent system initialized successfully")
        
        return manager
    except Exception as e:
        logger.error(f"Error initializing agent system: {str(e)}")
        raise

if __name__ == "__main__":
    initialize_agents()