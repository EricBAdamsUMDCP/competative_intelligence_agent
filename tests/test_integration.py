# tests/test_integration.py
import unittest
import asyncio
import sys
import os
import json
import tempfile
import shutil
from unittest.mock import patch, MagicMock, AsyncMock

# Add project root to path to enable imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.agents.base_agent import BaseAgent
from core.agents.orchestrator import AgentOrchestrator
from core.agents.collection_agent import CollectionAgent
from core.agents.entity_analysis_agent import EntityAnalysisAgent
from core.agents.knowledge_graph_agent import KnowledgeGraphAgent
from core.agents.bid_analysis_agent import BidAnalysisAgent
from core.agents.feedback_learning_agent import FeedbackLearningAgent
from core.agents.factory import AgentFactory
from core.agents.workflows import Workflows
from core.agents.manager import AgentManager

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
        }
    ],
    "agency_relationships": [
        {
            "agency": "Department of Defense",
            "strength": 4,
            "contacts": 5
        }
    ],
    "technology_strengths": [
        "cybersecurity", "zero trust", "cloud"
    ],
    "min_contract_value": 500000,
    "target_contract_value": 5000000,
    "max_contract_value": 15000000,
    "strategic_agencies": [
        "Department of Defense", "Department of Homeland Security"
    ],
    "strategic_technologies": [
        "zero trust", "artificial intelligence", "DevSecOps"
    ]
}

# Sample mock data for testing
MOCK_SAM_GOV_DATA = [
    {
        'opportunityId': 'SAMGOV123456',
        'title': 'Cybersecurity Services for Department of Defense',
        'solicitationNumber': 'DOD-CYBER-2023-01',
        'agency': 'Department of Defense',
        'agencyId': 'DOD',
        'description': 'The Department of Defense is seeking cybersecurity services to enhance the security posture of critical infrastructure. Services include vulnerability assessment, penetration testing, and security monitoring with NIST compliance.',
        'award': {
            'awardee': 'TechDefense Solutions',
            'awardeeId': 'TDS12345',
            'amount': 5600000
        }
    }
]

# Mock SamGovCollector for testing
class MockSamGovCollector:
    """Mock SAM.gov collector for testing."""
    
    def __init__(self, source_name="sam.gov", config=None):
        self.source_name = source_name
        self.config = config or {}
        self.last_run = None
    
    async def collect(self):
        """Return mock data."""
        return MOCK_SAM_GOV_DATA
    
    async def run(self):
        """Process and return mock data."""
        return MOCK_SAM_GOV_DATA


# Create a mock Neo4j KnowledgeGraph for testing
class MockKnowledgeGraph:
    """Mock Knowledge Graph for testing."""
    
    def __init__(self, uri=None, user=None, password=None):
        self.uri = uri
        self.user = user
        self.password = password
        self.data = {
            "opportunities": {},
            "contractors": {},
            "agencies": {},
            "technologies": {},
            "regulations": {},
            "clearances": {}
        }
    
    def _initialize_schema(self):
        """Mock schema initialization."""
        pass
    
    def add_contract_award(self, award_data):
        """Mock adding a contract award."""
        opp_id = award_data.get("opportunity_id")
        if opp_id:
            self.data["opportunities"][opp_id] = award_data
        
        contractor_id = award_data.get("contractor_id")
        if contractor_id:
            if contractor_id not in self.data["contractors"]:
                self.data["contractors"][contractor_id] = []
            self.data["contractors"][contractor_id].append(opp_id)
        
        agency_id = award_data.get("agency_id")
        if agency_id:
            if agency_id not in self.data["agencies"]:
                self.data["agencies"][agency_id] = []
            self.data["agencies"][agency_id].append(opp_id)
    
    def search_opportunities(self, query, limit=10):
        """Mock searching opportunities."""
        results = []
        for opp_id, opp_data in self.data["opportunities"].items():
            if query.lower() in opp_data.get("title", "").lower() or query.lower() in opp_data.get("description", "").lower():
                results.append({
                    "id": opp_id,
                    "title": opp_data.get("title"),
                    "description": opp_data.get("description"),
                    "value": opp_data.get("value"),
                    "award_date": opp_data.get("award_date"),
                    "score": 1.0
                })
                if len(results) >= limit:
                    break
        return results
    
    def get_competitor_insights(self, competitor_id):
        """Mock getting competitor insights."""
        if competitor_id not in self.data["contractors"]:
            return []
        
        results = []
        for opp_id in self.data["contractors"][competitor_id]:
            opp_data = self.data["opportunities"].get(opp_id)
            if opp_data:
                agency_name = opp_data.get("agency_name")
                if agency_name:
                    # Check if we already have an entry for this agency
                    agency_entry = None
                    for entry in results:
                        if entry["agency_name"] == agency_name:
                            agency_entry = entry
                            break
                    
                    if agency_entry:
                        agency_entry["contract_count"] += 1
                        agency_entry["total_value"] += opp_data.get("value", 0)
                    else:
                        results.append({
                            "agency_name": agency_name,
                            "contract_count": 1,
                            "total_value": opp_data.get("value", 0)
                        })
        
        return results
    
    def get_opportunity_entities(self, opportunity_id):
        """Mock getting opportunity entities."""
        opp_data = self.data["opportunities"].get(opportunity_id)
        if not opp_data or "entity_summary" not in opp_data:
            return {
                "technologies": [],
                "regulations": [],
                "clearances": []
            }
        
        summary = opp_data["entity_summary"]
        return {
            "technologies": summary.get("tech_stack", []),
            "regulations": summary.get("regulatory_requirements", []),
            "clearances": summary.get("clearance_requirements", [])
        }
    
    def close(self):
        """Mock closing the database connection."""
        pass

# Patch the modules for testing
@patch("core.agents.knowledge_graph_agent.KnowledgeGraph", MockKnowledgeGraph)
@patch("core.collectors.sam_gov.SamGovCollector", MockSamGovCollector)
class TestAgentSystemIntegration(unittest.TestCase):
    """Integration tests for the agent system."""
    
    def setUp(self):
        # Create a temporary directory for feedback data
        self.temp_dir = tempfile.mkdtemp()
        
        # Create an orchestrator
        self.orchestrator = AgentOrchestrator()
        
        # Create the agents
        self.collection_agent = CollectionAgent(
            agent_id="collection_agent",
            name="Collection Agent",
            collector_type="core.collectors.sam_gov.SamGovCollector"
        )
        
        self.entity_agent = EntityAnalysisAgent(
            agent_id="entity_analysis_agent",
            name="Entity Analysis Agent"
        )
        
        self.kg_agent = KnowledgeGraphAgent(
            agent_id="knowledge_graph_agent",
            name="Knowledge Graph Agent"
        )
        
        self.bid_agent = BidAnalysisAgent(
            agent_id="bid_analysis_agent",
            name="Bid Analysis Agent",
            company_profile=SAMPLE_COMPANY_PROFILE
        )
        
        self.feedback_agent = FeedbackLearningAgent(
            agent_id="feedback_learning_agent",
            name="Feedback Learning Agent",
            storage_path=self.temp_dir
        )
        
        # Register the agents
        self.orchestrator.register_agent(self.collection_agent)
        self.orchestrator.register_agent(self.entity_agent)
        self.orchestrator.register_agent(self.kg_agent)
        self.orchestrator.register_agent(self.bid_agent)
        self.orchestrator.register_agent(self.feedback_agent)
        
        # Create the agent manager
        self.manager = AgentManager()
        
        # Register agents with the manager
        self.manager.register_agent("collection_agent", self.collection_agent)
        self.manager.register_agent("entity_analysis_agent", self.entity_agent)
        self.manager.register_agent("knowledge_graph_agent", self.kg_agent)
        self.manager.register_agent("bid_analysis_agent", self.bid_agent)
        self.manager.register_agent("feedback_learning_agent", self.feedback_agent)
    
    def tearDown(self):
        # Clean up the temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_data_collection_workflow(self):
        """Test the data collection workflow."""
        # Define the workflow
        workflow_def = Workflows.data_collection_workflow(
            collector_type="core.collectors.sam_gov.SamGovCollector"
        )
        
        # Execute the workflow
        result = asyncio.run(self.orchestrator.execute_workflow(
            workflow_id="test_data_collection",
            workflow_def=workflow_def
        ))
        
        # Check the result
        self.assertEqual(result["status"], "completed")
        self.assertIn("collection_result", result["results"])
        self.assertIn("entity_result", result["results"])
        self.assertIn("storage_result", result["results"])
        
        # Check that data was collected
        collection_result = result["results"]["collection_result"]
        self.assertIn("items", collection_result)
        self.assertGreater(len(collection_result["items"]), 0)
        
        # Check that entities were extracted
        entity_result = result["results"]["entity_result"]
        self.assertIn("processed_documents", entity_result)
        self.assertGreater(len(entity_result["processed_documents"]), 0)
        
        # Check that data was stored
        storage_result = result["results"]["storage_result"]
        self.assertIn("stored_count", storage_result)
        self.assertGreater(storage_result["stored_count"], 0)
    
    def test_opportunity_analysis_workflow(self):
        """Test the opportunity analysis workflow."""
        # Create a test opportunity
        opportunity_data = {
            "id": "test_opp",
            "title": "Test Opportunity",
            "agency": "Department of Defense",
            "value": 5000000,
            "description": "Test opportunity for cybersecurity services with NIST compliance and cloud security."
        }
        
        # Define the workflow
        workflow_def = Workflows.opportunity_analysis_workflow(
            opportunity_data=opportunity_data,
            company_profile=SAMPLE_COMPANY_PROFILE
        )
        
        # Execute the workflow
        result = asyncio.run(self.orchestrator.execute_workflow(
            workflow_id="test_opportunity_analysis",
            workflow_def=workflow_def
        ))
        
        # Check the result
        self.assertEqual(result["status"], "completed")
        self.assertIn("entity_result", result["results"])
        self.assertIn("storage_result", result["results"])
        self.assertIn("bid_result", result["results"])
        
        # Check that entities were extracted
        entity_result = result["results"]["entity_result"]
        self.assertIn("text_analysis", entity_result)
        
        # Check that a bid analysis was performed
        bid_result = result["results"]["bid_result"]
        self.assertIn("analyzed_opportunities", bid_result)
        self.assertGreater(len(bid_result["analyzed_opportunities"]), 0)
        
        # Check that the recommendation is present
        analysis = bid_result["analyzed_opportunities"][0]
        self.assertIn("recommendation", analysis)
        self.assertIn("overall_score", analysis)
    
    def test_bid_decision_feedback_workflow(self):
        """Test the bid decision feedback workflow."""
        # First, we need an opportunity in the knowledge graph
        opportunity_data = {
            "opportunity_id": "feedback_test_opp",
            "title": "Feedback Test Opportunity",
            "agency_name": "Department of Defense",
            "agency_id": "DOD",
            "value": 5000000,
            "description": "Test opportunity for feedback.",
            "entity_summary": {
                "tech_stack": ["cybersecurity", "cloud"],
                "regulatory_requirements": ["NIST"],
                "clearance_requirements": ["Secret"]
            }
        }
        
        # Add the opportunity to the knowledge graph
        self.kg_agent._get_graph().add_contract_award(opportunity_data)
        
        # Define the workflow
        workflow_def = Workflows.bid_decision_feedback_workflow(
            opportunity_id="feedback_test_opp",
            bid_decision=True,
            win_result=True,
            feedback_notes="Great opportunity that matched our capabilities."
        )
        
        # Execute the workflow
        result = asyncio.run(self.orchestrator.execute_workflow(
            workflow_id="test_bid_feedback",
            workflow_def=workflow_def
        ))
        
        # Check the result
        self.assertEqual(result["status"], "completed")
        self.assertIn("opportunity_data", result["results"])
        self.assertIn("feedback_result", result["results"])
        self.assertIn("model_update", result["results"])
        
        # Check that feedback was added
        feedback_result = result["results"]["feedback_result"]
        self.assertIn("feedback_id", feedback_result)
        self.assertEqual(feedback_result["status"], "success")
    
    def test_market_intelligence_workflow(self):
        """Test the market intelligence workflow."""
        # First, we need some data in the knowledge graph
        for i in range(5):
            opportunity_data = {
                "opportunity_id": f"market_test_opp_{i}",
                "title": f"Market Test Opportunity {i}",
                "agency_name": "Department of Defense" if i % 2 == 0 else "Department of Health and Human Services",
                "agency_id": "DOD" if i % 2 == 0 else "HHS",
                "value": 5000000 + i * 1000000,
                "description": f"Test opportunity {i} for market intelligence.",
                "entity_summary": {
                    "tech_stack": ["cybersecurity", "cloud"] if i % 2 == 0 else ["artificial intelligence", "machine learning"],
                    "regulatory_requirements": ["NIST", "CMMC"] if i % 2 == 0 else ["HIPAA", "FISMA"],
                    "clearance_requirements": ["Secret"] if i % 2 == 0 else ["Public Trust"]
                }
            }
            
            # Add the opportunity to the knowledge graph
            self.kg_agent._get_graph().add_contract_award(opportunity_data)
        
        # Define the workflow
        workflow_def = Workflows.market_intelligence_workflow(
            agency_id="DOD",
            technology="cybersecurity"
        )
        
        # Execute the workflow
        result = asyncio.run(self.orchestrator.execute_workflow(
            workflow_id="test_market_intelligence",
            workflow_def=workflow_def
        ))
        
        # Check the result
        self.assertEqual(result["status"], "completed")
        self.assertIn("agency_data", result["results"])
        self.assertIn("technology_data", result["results"])
        self.assertIn("agency_insights", result["results"])
        self.assertIn("technology_insights", result["results"])
        self.assertIn("success_factors", result["results"])
    
    def test_agent_manager_run_workflow(self):
        """Test running a workflow through the agent manager."""
        # Run a data collection workflow
        result = asyncio.run(self.manager.run_workflow(
            workflow_type="data_collection",
            params={
                "id": "manager_test",
                "collector_type": "core.collectors.sam_gov.SamGovCollector"
            }
        ))
        
        # Check the result
        self.assertIn("workflow_id", result)
        self.assertEqual(result["status"], "completed")
        self.assertIn("results", result)
        
        # Check that we can get the workflow state
        workflow_id = result["workflow_id"]
        workflow_state = self.manager.get_workflow_state(workflow_id)
        
        self.assertIsNotNone(workflow_state)
        self.assertEqual(workflow_state["id"], workflow_id)
        self.assertEqual(workflow_state["status"], "completed")

if __name__ == "__main__":
    unittest.main()