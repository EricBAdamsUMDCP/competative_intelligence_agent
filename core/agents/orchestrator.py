# core/agents/orchestrator.py
from typing import Dict, List, Any, Type, Optional
import logging
import asyncio
import uuid
from datetime import datetime

from core.agents.base_agent import BaseAgent

class AgentOrchestrator:
    """Orchestrator for coordinating multiple agents in the system."""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.logger = logging.getLogger("agent.orchestrator")
        self.workflows: Dict[str, Dict[str, Any]] = {}
        
    def register_agent(self, agent: BaseAgent) -> str:
        """Register an agent with the orchestrator.
        
        Args:
            agent: The agent to register
            
        Returns:
            The ID of the registered agent
        """
        agent_id = agent.agent_id
        if agent_id in self.agents:
            self.logger.warning(f"Agent with ID {agent_id} already registered, replacing")
        
        self.agents[agent_id] = agent
        self.logger.info(f"Registered agent {agent.name} with ID {agent_id}")
        
        return agent_id
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get an agent by ID.
        
        Args:
            agent_id: ID of the agent to get
            
        Returns:
            The agent or None if not found
        """
        return self.agents.get(agent_id)
    
    def get_all_agent_states(self) -> Dict[str, Dict[str, Any]]:
        """Get the current states of all registered agents.
        
        Returns:
            Dictionary mapping agent IDs to their current states
        """
        return {agent_id: agent.get_state() for agent_id, agent in self.agents.items()}
    
    async def run_agent(self, agent_id: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run a specific agent.
        
        Args:
            agent_id: ID of the agent to run
            params: Parameters for the agent execution
            
        Returns:
            Results of the agent execution
        """
        agent = self.get_agent(agent_id)
        if not agent:
            raise ValueError(f"No agent with ID {agent_id}")
        
        task_id = str(uuid.uuid4())
        results = await agent.run(task_id=task_id, params=params or {})
        
        return {
            "agent_id": agent_id,
            "task_id": task_id,
            "results": results
        }
    
    async def execute_workflow(self, workflow_id: str, workflow_def: Dict[str, Any], 
                              workflow_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a multi-agent workflow.
        
        Args:
            workflow_id: ID for this workflow execution
            workflow_def: Definition of the workflow to execute
            workflow_params: Parameters for the workflow execution
            
        Returns:
            Results of the workflow execution
        """
        workflow_id = workflow_id or str(uuid.uuid4())
        workflow_params = workflow_params or {}
        
        self.logger.info(f"Starting workflow {workflow_id}")
        
        workflow_state = {
            "id": workflow_id,
            "status": "running",
            "start_time": datetime.now().isoformat(),
            "steps": [],
            "results": {}
        }
        
        self.workflows[workflow_id] = workflow_state
        
        try:
            steps = workflow_def.get("steps", [])
            step_results = {}
            
            for i, step in enumerate(steps):
                step_id = f"{workflow_id}_step_{i}"
                agent_id = step.get("agent_id")
                step_params = step.get("params", {})
                
                # Replace parameter placeholders with values from previous steps
                for param_key, param_value in step_params.items():
                    if isinstance(param_value, str) and param_value.startswith("$"):
                        result_key = param_value[1:]
                        if result_key in step_results:
                            step_params[param_key] = step_results[result_key]
                
                # Add global workflow parameters
                step_params.update(workflow_params)
                
                self.logger.info(f"Executing workflow step {i+1}/{len(steps)}: {agent_id}")
                
                step_start_time = datetime.now()
                step_result = await self.run_agent(agent_id, step_params)
                step_end_time = datetime.now()
                
                step_info = {
                    "step_id": step_id,
                    "agent_id": agent_id,
                    "status": "completed",
                    "start_time": step_start_time.isoformat(),
                    "end_time": step_end_time.isoformat(),
                    "duration": (step_end_time - step_start_time).total_seconds()
                }
                
                workflow_state["steps"].append(step_info)
                
                # Store results for this step
                result_key = step.get("result_key", f"step_{i}_result")
                step_results[result_key] = step_result.get("results", {})
            
            workflow_state["results"] = step_results
            workflow_state["status"] = "completed"
            workflow_state["end_time"] = datetime.now().isoformat()
            
            self.logger.info(f"Workflow {workflow_id} completed successfully")
            
            return {
                "workflow_id": workflow_id,
                "status": "completed",
                "results": step_results
            }
            
        except Exception as e:
            workflow_state["status"] = "failed"
            workflow_state["error"] = str(e)
            workflow_state["end_time"] = datetime.now().isoformat()
            
            self.logger.error(f"Workflow {workflow_id} failed: {str(e)}")
            
            raise
    
    def get_workflow_state(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get the current state of a workflow.
        
        Args:
            workflow_id: ID of the workflow to get
            
        Returns:
            The workflow state or None if not found
        """
        return self.workflows.get(workflow_id)