# tests/test_base_agent.py
import unittest
import asyncio
import sys
import os
from unittest.mock import patch, MagicMock

# Add project root to path to enable imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.agents.base_agent import BaseAgent

# Create a concrete implementation of BaseAgent for testing
class TestAgent(BaseAgent):
    """Concrete implementation of BaseAgent for testing."""
    
    async def execute_task(self, params):
        if "fail" in params and params["fail"]:
            raise Exception("Task execution failed as requested")
        
        # Simulate processing time
        if "delay" in params:
            await asyncio.sleep(params["delay"])
        
        # Return test results
        return {
            "task_executed": True,
            "params_received": params,
            "agent_id": self.agent_id,
            "agent_name": self.name
        }

class TestBaseAgent(unittest.TestCase):
    """Tests for the BaseAgent class."""
    
    def setUp(self):
        self.agent = TestAgent(agent_id="test_agent", name="Test Agent")
    
    def test_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.agent_id, "test_agent")
        self.assertEqual(self.agent.name, "Test Agent")
        self.assertEqual(self.agent.state, "initialized")
        self.assertIsNone(self.agent.last_task_id)
        self.assertEqual(self.agent.results_cache, {})
    
    def test_get_state(self):
        """Test getting agent state."""
        state = self.agent.get_state()
        self.assertEqual(state["agent_id"], "test_agent")
        self.assertEqual(state["name"], "Test Agent")
        self.assertEqual(state["state"], "initialized")
        self.assertIsNone(state["last_task_id"])
    
    def test_run_success(self):
        """Test running an agent task successfully."""
        # Run the task
        result = asyncio.run(self.agent.run(task_id="task1", params={"test": "value"}))
        
        # Check the result
        self.assertTrue(result["task_executed"])
        self.assertEqual(result["params_received"], {"test": "value"})
        self.assertEqual(result["agent_id"], "test_agent")
        self.assertEqual(result["agent_name"], "Test Agent")
        
        # Check agent state after run
        self.assertEqual(self.agent.state, "idle")
        self.assertEqual(self.agent.last_task_id, "task1")
        self.assertIn("task1", self.agent.results_cache)
    
    def test_run_failure(self):
        """Test handling agent task failure."""
        with self.assertRaises(Exception) as context:
            asyncio.run(self.agent.run(task_id="task_fail", params={"fail": True}))
        
        self.assertEqual(str(context.exception), "Task execution failed as requested")
        self.assertEqual(self.agent.state, "error")
        self.assertEqual(self.agent.last_task_id, "task_fail")
        self.assertNotIn("task_fail", self.agent.results_cache)
    
    def test_get_results(self):
        """Test retrieving task results from cache."""
        # First run a task to cache results
        asyncio.run(self.agent.run(task_id="task_cache", params={"cache": "test"}))
        
        # Then retrieve the results
        results = self.agent.get_results("task_cache")
        self.assertIsNotNone(results)
        self.assertTrue(results["task_executed"])
        self.assertEqual(results["params_received"], {"cache": "test"})
        
        # Non-existent task should return None
        self.assertIsNone(self.agent.get_results("nonexistent_task"))
    
    def test_auto_task_id_generation(self):
        """Test automatic task ID generation."""
        # Run without providing a task ID
        result = asyncio.run(self.agent.run(params={"auto": "id"}))
        
        # The task ID should have been auto-generated
        self.assertIsNotNone(self.agent.last_task_id)
        self.assertTrue(result["task_executed"])
        self.assertEqual(result["params_received"], {"auto": "id"})
        
        # The result should be in the cache with the auto-generated ID
        self.assertIn(self.agent.last_task_id, self.agent.results_cache)
    
    def test_concurrency(self):
        """Test running multiple tasks concurrently."""
        async def run_multiple_tasks():
            # Create tasks with different delays to ensure they run concurrently
            task1 = self.agent.run(task_id="task_concurrent_1", params={"delay": 0.1, "task": 1})
            task2 = self.agent.run(task_id="task_concurrent_2", params={"delay": 0.05, "task": 2})
            
            # Run tasks concurrently
            results = await asyncio.gather(task1, task2)
            return results
        
        # Run the concurrent tasks
        results = asyncio.run(run_multiple_tasks())
        
        # Check that both tasks completed
        self.assertEqual(len(results), 2)
        self.assertTrue(results[0]["task_executed"])
        self.assertTrue(results[1]["task_executed"])
        
        # Because task2 should finish first (shorter delay), the last_task_id should be from task1
        self.assertEqual(self.agent.last_task_id, "task_concurrent_1")
        
        # Both results should be in the cache
        self.assertIn("task_concurrent_1", self.agent.results_cache)
        self.assertIn("task_concurrent_2", self.agent.results_cache)

if __name__ == "__main__":
    unittest.main()