# core/agents/collection_agent.py
from typing import Dict, List, Any, Type, Optional
import logging
import asyncio
import importlib
import uuid

from core.agents.base_agent import BaseAgent
from core.collectors.base_collector import BaseCollector

class CollectionAgent(BaseAgent):
    """Agent for collecting data from various sources."""
    
    def __init__(self, agent_id: str = None, name: str = None, 
                 collector_type: str = None, collector_config: Dict[str, Any] = None):
        """Initialize a new collection agent.
        
        Args:
            agent_id: Unique identifier for the agent
            name: Human-readable name for the agent
            collector_type: Type of collector to use (module path)
            collector_config: Configuration for the collector
        """
        config = {
            "collector_type": collector_type,
            "collector_config": collector_config or {}
        }
        
        super().__init__(agent_id, name or f"{collector_type.split('.')[-1]}_agent", config)
        
        self.collector = None
        self._initialize_collector()
    
    def _initialize_collector(self):
        """Initialize the collector for this agent."""
        collector_type = self.config.get("collector_type")
        collector_config = self.config.get("collector_config", {})
        
        if not collector_type:
            raise ValueError("No collector_type specified")
        
        try:
            # Dynamic import of the collector module
            module_path, class_name = collector_type.rsplit(".", 1)
            module = importlib.import_module(module_path)
            collector_class = getattr(module, class_name)
            
            # Create an instance of the collector
            self.collector = collector_class(**collector_config)
            
            self.logger.info(f"Initialized collector of type {collector_type}")
        except (ImportError, AttributeError) as e:
            self.logger.error(f"Failed to initialize collector: {str(e)}")
            raise ValueError(f"Failed to initialize collector: {str(e)}")
    
    async def execute_task(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data collection task.
        
        Args:
            params: Parameters for the collection task
                - query: Optional query parameters for collection
                - time_range: Optional time range for collection
                
        Returns:
            Results of the collection task
        """
        if not self.collector:
            raise ValueError("Collector not initialized")
        
        # Extract parameters
        query = params.get("query", {})
        time_range = params.get("time_range", {})
        
        # Update collector config with task parameters if needed
        if query:
            self.collector.config.update(query)
        
        self.logger.info(f"Starting collection with params: {params}")
        
        # Run the collector
        start_time = self.collector.last_run
        results = await self.collector.run()
        end_time = self.collector.last_run
        
        return {
            "collector_type": self.config.get("collector_type"),
            "start_time": start_time.isoformat() if start_time else None,
            "end_time": end_time.isoformat() if end_time else None,
            "item_count": len(results),
            "items": results
        }