# tests/test_agent_orchestrator.py
import unittest
import asyncio
import sys
import os
from unittest.mock import patch, MagicMock, AsyncMock

# Add project root to path to enable imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.agents.base_agent import BaseAgent
from core.agents.orchestrator import AgentOrchestrator

# Create a simple agent for testing
class MockAgent(BaseAgent):
    """Mock agent for testing the orchestrator."""
    
    def __init__(self, agent_id=None, name=None, config=None, should_fail=False):
        super().__init__(agent_id, name, config)
        self.should_fail = should_fail
        self.execution_count = 0
    
    async def execute_task(self, params):
        self.execution_count += 1
        
        if self.should_fail:
            raise Exception("Task execution failed as requested")
        
        return {
            "execution_count": self.execution_count,
            "agent_id": self.agent_id,
            "params": params
        }

class TestAgentOrchestrator(unittest.TestCase):
    """Tests for the AgentOrchestrator class."""
    
    def setUp(self):
        self.orchestrator = AgentOrchestrator()
        
        # Create some test agents
        self.agent1 = MockAgent(agent_id="agent1", name="Agent 1")
        self.agent2 = MockAgent(agent_id="agent2", name="Agent 2")
        self.failing_agent = MockAgent(agent_id="failing_agent", name="Failing Agent", should_fail=True)
        
        # Register agents
        self.orchestrator.register_agent(self.agent1)
        self.orchestrator.register_agent(self.agent2)
        self.orchestrator.register_agent(self.failing_agent)
    
    def test_initialization(self):
        """Test orchestrator initialization."""
        self.assertEqual(len(self.orchestrator.agents), 3)
        self.assertEqual(len(self.orchestrator.workflows), 0)
    
    def test_register_agent(self):
        """Test registering agents."""
        # Create a new agent
        new_agent = MockAgent(agent_id="new_agent", name="New Agent")
        
        # Register the agent
        agent_id = self.orchestrator.register_agent(new_agent)
        
        # Check that the agent was registered
        self.assertEqual(agent_id, "new_agent")
        self.assertIn("new_agent", self.orchestrator.agents)
        self.assertEqual(self.orchestrator.agents["new_agent"], new_agent)
        
        # Test replacing an agent
        replacement_agent = MockAgent(agent_id="agent1", name="Replacement Agent")
        agent_id = self.orchestrator.register_agent(replacement_agent)
        
        # Check that the agent was replaced
        self.assertEqual(agent_id, "agent1")
        self.assertEqual(self.orchestrator.agents["agent1"], replacement_agent)
    
    def test_get_agent(self):
        """Test getting agents by ID."""
        # Get an existing agent
        agent = self.orchestrator.get_agent("agent1")
        self.assertEqual(agent, self.agent1)
        
        # Get a non-existent agent
        agent = self.orchestrator.get_agent("nonexistent_agent")
        self.assertIsNone(agent)
    
    def test_get_all_agent_states(self):
        """Test getting all agent states."""
        states = self.orchestrator.get_all_agent_states()
        
        # Check that all agents are included
        self.assertEqual(len(states), 3)
        self.assertIn("agent1", states)
        self.assertIn("agent2", states)
        self.assertIn("failing_agent", states)
        
        # Check that state information is correct
        self.assertEqual(states["agent1"]["name"], "Agent 1")
        self.assertEqual(states["agent1"]["state"], "initialized")
    
    def test_run_agent(self):
        """Test running a specific agent."""
        # Run an agent
        result = asyncio.run(self.orchestrator.run_agent("agent1", {"test": "value"}))
        
        # Check the result
        self.assertEqual(result["agent_id"], "agent1")
        self.assertIn("task_id", result)
        self.assertIn("results", result)
        self.assertEqual(result["results"]["execution_count"], 1)
        self.assertEqual(result["results"]["params"], {"test": "value"})
        
        # Run the same agent again
        result = asyncio.run(self.orchestrator.run_agent("agent1", {"test": "again"}))
        
        # Check that the execution count incremented
        self.assertEqual(result["results"]["execution_count"], 2)
        
        # Try to run a non-existent agent
        with self.assertRaises(ValueError):
            asyncio.run(self.orchestrator.run_agent("nonexistent_agent", {}))
        
        # Run a failing agent
        with self.assertRaises(Exception):
            asyncio.run(self.orchestrator.run_agent("failing_agent", {}))
    
    def test_execute_workflow(self):
        """Test executing a workflow."""
        # Define a simple workflow
        workflow_def = {
            "name": "Test Workflow",
            "description": "A test workflow",
            "steps": [
                {
                    "name": "Step 1",
                    "agent_id": "agent1",
                    "result_key": "step1_result",
                    "params": {"step": 1}
                },
                {
                    "name": "Step 2",
                    "agent_id": "agent2",
                    "result_key": "step2_result",
                    "params": {"step": 2}
                }
            ]
        }
        
        # Execute the workflow
        result = asyncio.run(self.orchestrator.execute_workflow(
            workflow_id="test_workflow",
            workflow_def=workflow_def,
            workflow_params={"workflow_param": "value"}
        ))
        
        # Check the result
        self.assertEqual(result["workflow_id"], "test_workflow")
        self.assertEqual(result["status"], "completed")
        self.assertIn("results", result)
        self.assertIn("step1_result", result["results"])
        self.assertIn("step2_result", result["results"])
        
        # Check that the workflow was stored
        self.assertIn("test_workflow", self.orchestrator.workflows)
        self.assertEqual(self.orchestrator.workflows["test_workflow"]["status"], "completed")
        
        # Check that agents were executed with the correct parameters
        self.assertEqual(self.agent1.execution_count, 1)
        self.assertEqual(self.agent2.execution_count, 1)
    
    def test_workflow_with_parameter_passing(self):
        """Test a workflow with parameter passing between steps."""
        # Define a workflow that passes parameters between steps
        workflow_def = {
            "name": "Parameter Passing Workflow",
            "description": "A workflow that passes parameters between steps",
            "steps": [
                {
                    "name": "Step 1",
                    "agent_id": "agent1",
                    "result_key": "step1_result",
                    "params": {"step": 1, "generate": "value"}
                },
                {
                    "name": "Step 2",
                    "agent_id": "agent2",
                    "result_key": "step2_result",
                    "params": {"step": 2, "previous_result": "$step1_result"}
                }
            ]
        }
        
        # Execute the workflow
        result = asyncio.run(self.orchestrator.execute_workflow(
            workflow_id="param_passing_workflow",
            workflow_def=workflow_def
        ))
        
        # Check the result
        self.assertEqual(result["status"], "completed")
        
        # Get the workflow state
        workflow_state = self.orchestrator.get_workflow_state("param_passing_workflow")
        
        # Check that parameters were passed correctly
        step2_params = workflow_state["steps"][1].get("params", {})
        self.assertIn("previous_result", step2_params)
        
        # Check the results of both steps
        step1_result = result["results"]["step1_result"]
        step2_result = result["results"]["step2_result"]
        
        self.assertEqual(step1_result["execution_count"], 2)  # Second execution of agent1
        self.assertEqual(step2_result["execution_count"], 2)  # Second execution of agent2
    
    def test_workflow_with_failing_step(self):
        """Test a workflow with a failing step."""
        # Define a workflow with a failing step
        workflow_def = {
            "name": "Failing Workflow",
            "description": "A workflow with a failing step",
            "steps": [
                {
                    "name": "Step 1",
                    "agent_id": "agent1",
                    "result_key": "step1_result",
                    "params": {"step": 1}
                },
                {
                    "name": "Failing Step",
                    "agent_id": "failing_agent",
                    "result_key": "failing_step_result",
                    "params": {"step": "fail"}
                },
                {
                    "name": "Step 3",
                    "agent_id": "agent2",
                    "result_key": "step3_result",
                    "params": {"step": 3}
                }
            ]
        }
        
        # Execute the workflow
        with self.assertRaises(Exception):
            asyncio.run(self.orchestrator.execute_workflow(
                workflow_id="failing_workflow",
                workflow_def=workflow_def
            ))
        
        # Check that the workflow was stored with a failed status
        self.assertIn("failing_workflow", self.orchestrator.workflows)
        self.assertEqual(self.orchestrator.workflows["failing_workflow"]["status"], "failed")
        
        # Check that only the first step was executed
        self.assertEqual(self.agent1.execution_count, 3)  # Third execution of agent1
        self.assertEqual(self.agent2.execution_count, 2)  # Still only two executions of agent2
    
    def test_get_workflow_state(self):
        """Test getting the state of a workflow."""
        # Define and execute a simple workflow
        workflow_def = {
            "name": "State Test Workflow",
            "description": "A workflow for testing state retrieval",
            "steps": [
                {
                    "name": "Single Step",
                    "agent_id": "agent1",
                    "result_key": "step_result",
                    "params": {"state": "test"}
                }
            ]
        }
        
        # Execute the workflow
        asyncio.run(self.orchestrator.execute_workflow(
            workflow_id="state_test_workflow",
            workflow_def=workflow_def
        ))
        
        # Get the workflow state
        workflow_state = self.orchestrator.get_workflow_state("state_test_workflow")
        
        # Check the state
        self.assertEqual(workflow_state["id"], "state_test_workflow")
        self.assertEqual(workflow_state["status"], "completed")
        self.assertEqual(len(workflow_state["steps"]), 1)
        self.assertIn("results", workflow_state)
        self.assertIn("step_result", workflow_state["results"])
        
        # Get a non-existent workflow state
        workflow_state = self.orchestrator.get_workflow_state("nonexistent_workflow")
        self.assertIsNone(workflow_state)

if __name__ == "__main__":
    unittest.main()