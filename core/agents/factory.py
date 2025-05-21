# core/agents/factory.py
import logging
import importlib
from typing import Dict, List, Any, Type, Optional

from core.agents.base_agent import BaseAgent

class AgentFactory:
    """Factory for creating agent instances dynamically."""
    
    # Registry of known agent types
    AGENT_TYPES = {
        "collection_agent": "core.agents.collection_agent.CollectionAgent",
        "entity_analysis_agent": "core.agents.entity_analysis_agent.EntityAnalysisAgent",
        "knowledge_graph_agent": "core.agents.knowledge_graph_agent.KnowledgeGraphAgent",
        "bid_analysis_agent": "core.agents.bid_analysis_agent.BidAnalysisAgent",
        "feedback_learning_agent": "core.agents.feedback_learning_agent.FeedbackLearningAgent"
    }
    
    @staticmethod
    def create_agent(agent_type: str, agent_id: str = None, name: str = None, 
                    config: Dict[str, Any] = None) -> BaseAgent:
        """Create an agent of the specified type.
        
        Args:
            agent_type: Type of agent to create (key in AGENT_TYPES or full class path)
            agent_id: ID to assign to the agent
            name: Name to assign to the agent
            config: Configuration for the agent
            
        Returns:
            The created agent
            
        Raises:
            ValueError: If agent_type is not recognized
            ImportError: If the agent class cannot be imported
        """
        logger = logging.getLogger("agent.factory")
        
        # Get the class path for the agent type
        if agent_type in AgentFactory.AGENT_TYPES:
            class_path = AgentFactory.AGENT_TYPES[agent_type]
        else:
            class_path = agent_type
        
        try:
            # Split the class path into module path and class name
            module_path, class_name = class_path.rsplit(".", 1)
            
            # Import the module
            module = importlib.import_module(module_path)
            
            # Get the class
            agent_class = getattr(module, class_name)
            
            # Create an instance of the agent
            kwargs = {}
            if agent_id:
                kwargs["agent_id"] = agent_id
            if name:
                kwargs["name"] = name
            if config:
                for key, value in config.items():
                    kwargs[key] = value
            
            agent = agent_class(**kwargs)
            
            logger.info(f"Created agent of type {agent_type} with name {agent.name}")
            
            return agent
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to create agent of type {agent_type}: {str(e)}")
            raise
    
    @staticmethod
    def create_agents_from_config(config: List[Dict[str, Any]]) -> Dict[str, BaseAgent]:
        """Create multiple agents from a configuration list.
        
        Args:
            config: List of agent configurations
                Each configuration must have:
                - type: Type of agent to create
                - id: ID to assign to the agent
                And may have:
                - name: Name to assign to the agent
                - config: Configuration for the agent
            
        Returns:
            Dictionary mapping agent IDs to agents
        """
        agents = {}
        
        for agent_config in config:
            agent_type = agent_config.get("type")
            agent_id = agent_config.get("id")
            
            if not agent_type or not agent_id:
                continue
            
            try:
                agent = AgentFactory.create_agent(
                    agent_type=agent_type,
                    agent_id=agent_id,
                    name=agent_config.get("name"),
                    config=agent_config.get("config")
                )
                
                agents[agent_id] = agent
            except Exception as e:
                logging.getLogger("agent.factory").error(
                    f"Failed to create agent {agent_id} of type {agent_type}: {str(e)}"
                )
        
        return agents