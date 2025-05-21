# core/agents/base_agent.py
from abc import ABC, abstractmethod
import logging
from typing import Dict, List, Any, Optional
import asyncio
import uuid

class BaseAgent(ABC):
    """Base class for all intelligent agents in the system."""
    
    def __init__(self, agent_id: str = None, name: str = None, config: Dict[str, Any] = None):
        """Initialize a new agent.
        
        Args:
            agent_id: Unique identifier for the agent
            name: Human-readable name for the agent
            config: Configuration parameters for the agent
        """
        self.agent_id = agent_id or str(uuid.uuid4())
        self.name = name or self.__class__.__name__
        self.config = config or {}
        self.logger = logging.getLogger(f"agent.{self.name}")
        self.state = "initialized"
        self.last_task_id = None
        self.results_cache = {}
        
    async def run(self, task_id: str = None, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run the agent's main task.
        
        Args:
            task_id: Unique identifier for this task execution
            params: Parameters for this task execution
            
        Returns:
            Results of the task execution
        """
        task_id = task_id or str(uuid.uuid4())
        self.last_task_id = task_id
        self.state = "running"
        
        self.logger.info(f"Agent {self.name} starting task {task_id}")
        
        try:
            # Execute the task
            results = await self.execute_task(params or {})
            
            # Store results in cache
            self.results_cache[task_id] = results
            
            self.state = "idle"
            self.logger.info(f"Agent {self.name} completed task {task_id}")
            
            return results
        except Exception as e:
            self.state = "error"
            self.logger.error(f"Agent {self.name} failed task {task_id}: {str(e)}")
            raise
    
    @abstractmethod
    async def execute_task(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent's specific task.
        
        Args:
            params: Parameters for this task execution
            
        Returns:
            Results of the task execution
        """
        pass
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the agent.
        
        Returns:
            Dictionary with the agent's current state
        """
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "state": self.state,
            "last_task_id": self.last_task_id
        }
    
    def get_results(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the results of a specific task.
        
        Args:
            task_id: ID of the task to get results for
            
        Returns:
            Results of the task or None if not found
        """
        return self.results_cache.get(task_id)