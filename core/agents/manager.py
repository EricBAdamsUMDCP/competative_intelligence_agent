# core/agents/manager.py
import logging
import asyncio
import os
import json
from typing import Dict, List, Any, Optional, Type

from core.agents.base_agent import BaseAgent
from core.agents.orchestrator import AgentOrchestrator
from core.agents.collection_agent import CollectionAgent
from core.agents.entity_analysis_agent import EntityAnalysisAgent
from core.agents.knowledge_graph_agent import KnowledgeGraphAgent
from core.agents.bid_analysis_agent import BidAnalysisAgent
from core.agents.feedback_learning_agent import FeedbackLearningAgent
from core.agents.workflows import Workflows

class AgentManager:
    """Manages agents and orchestrates workflows for the API."""
    
    def __init__(self, config_path: str = None):
        """Initialize the agent manager.
        
        Args:
            config_path: Path to agent configuration file
        """
        self.logger = logging.getLogger("agent.manager")
        self.orchestrator = AgentOrchestrator()
        self.logger.info("Initialized agent orchestrator")
        
        # Load configuration
        self.config = {}
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
                self.logger.info(f"Loaded agent configuration from {config_path}")
            except Exception as e:
                self.logger.error(f"Error loading agent configuration: {str(e)}")
        
        # Initialize the agent registry
        self.agent_registry = {}
        
        # Initialize default agents
        self._initialize_default_agents()
    
    def _initialize_default_agents(self):
        """Initialize the default set of agents."""
        # Collection agent
        collection_agent = CollectionAgent(
            name="collection_agent",
            collector_type="core.collectors.sam_gov.SamGovCollector"
        )
        self.register_agent("collection_agent", collection_agent)
        
        # Entity analysis agent
        entity_agent = EntityAnalysisAgent(
            name="entity_analysis_agent",
            model_name="en_core_web_lg"
        )
        self.register_agent("entity_analysis_agent", entity_agent)
        
        # Knowledge graph agent
        kg_agent = KnowledgeGraphAgent(
            name="knowledge_graph_agent",
            uri=os.environ.get("NEO4J_URI", "bolt://neo4j:7687"),
            user=os.environ.get("NEO4J_USER", "neo4j"),
            password=os.environ.get("NEO4J_PASSWORD", "password")
        )
        self.register_agent("knowledge_graph_agent", kg_agent)
        
        # Bid analysis agent
        company_profile = self.config.get("company_profile", {})
        bid_agent = BidAnalysisAgent(
            name="bid_analysis_agent",
            company_profile=company_profile
        )
        self.register_agent("bid_analysis_agent", bid_agent)
        
        # Feedback learning agent
        feedback_agent = FeedbackLearningAgent(
            name="feedback_learning_agent",
            storage_path=os.environ.get("FEEDBACK_STORAGE_PATH", "data/feedback")
        )
        self.register_agent("feedback_learning_agent", feedback_agent)
        
        self.logger.info("Initialized default agents")
    
    def register_agent(self, agent_id: str, agent: BaseAgent) -> str:
        """Register an agent with the manager and orchestrator.
        
        Args:
            agent_id: ID to assign to the agent
            agent: The agent to register
            
        Returns:
            The registered agent ID
        """
        # Update agent ID for consistency
        agent.agent_id = agent_id
        
        # Register with orchestrator
        self.orchestrator.register_agent(agent)
        
        # Add to local registry
        self.agent_registry[agent_id] = agent
        
        self.logger.info(f"Registered agent {agent.name} with ID {agent_id}")
        
        return agent_id
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get an agent by ID.
        
        Args:
            agent_id: ID of the agent to get
            
        Returns:
            The agent or None if not found
        """
        return self.agent_registry.get(agent_id)
    
    def get_all_agents(self) -> Dict[str, BaseAgent]:
        """Get all registered agents.
        
        Returns:
            Dictionary of agent IDs to agents
        """
        return self.agent_registry
    
    def get_agent_states(self) -> Dict[str, Dict[str, Any]]:
        """Get the current states of all agents.
        
        Returns:
            Dictionary of agent states
        """
        return self.orchestrator.get_all_agent_states()
    
    async def run_agent(self, agent_id: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run a specific agent.
        
        Args:
            agent_id: ID of the agent to run
            params: Parameters for the agent execution
            
        Returns:
            Results of the agent execution
        """
        return await self.orchestrator.run_agent(agent_id, params)
    
    async def run_workflow(self, workflow_type: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run a predefined workflow.
        
        Args:
            workflow_type: Type of workflow to run
            params: Parameters for the workflow
            
        Returns:
            Results of the workflow execution
        """
        workflow_id = f"{workflow_type}_{params.get('id', 'default')}"
        
        # Get workflow definition based on type
        if workflow_type == "data_collection":
            workflow_def = Workflows.data_collection_workflow(
                collector_type=params.get("collector_type"),
                collector_config=params.get("collector_config")
            )
        elif workflow_type == "opportunity_analysis":
            workflow_def = Workflows.opportunity_analysis_workflow(
                opportunity_data=params.get("opportunity_data", {}),
                company_profile=params.get("company_profile")
            )
        elif workflow_type == "competitor_analysis":
            workflow_def = Workflows.competitor_analysis_workflow(
                competitor_id=params.get("competitor_id")
            )
        elif workflow_type == "bid_decision_feedback":
            workflow_def = Workflows.bid_decision_feedback_workflow(
                opportunity_id=params.get("opportunity_id"),
                bid_decision=params.get("bid_decision"),
                win_result=params.get("win_result"),
                feedback_notes=params.get("feedback_notes", "")
            )
        elif workflow_type == "market_intelligence":
            workflow_def = Workflows.market_intelligence_workflow(
                agency_id=params.get("agency_id"),
                technology=params.get("technology"),
                time_period=params.get("time_period")
            )
        else:
            raise ValueError(f"Unknown workflow type: {workflow_type}")
        
        self.logger.info(f"Starting workflow: {workflow_type}")
        
        # Execute workflow
        result = await self.orchestrator.execute_workflow(
            workflow_id=workflow_id,
            workflow_def=workflow_def,
            workflow_params=params
        )
        
        return result
    
    def get_workflow_state(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get the current state of a workflow.
        
        Args:
            workflow_id: ID of the workflow to get
            
        Returns:
            The workflow state or None if not found
        """
        return self.orchestrator.get_workflow_state(workflow_id)