# core/agents/agent_framework.py
import logging
import asyncio
from typing import Dict, List, Any, Optional, Type, Set, Callable
from datetime import datetime

class Agent:
    """Base class for all agents in the CI system"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.logger = logging.getLogger(f"agent.{agent_id}")
        self.dependencies = set()
        self.status = "initialized"
        self.start_time = None
        self.end_time = None
        self.result = None
    
    def add_dependency(self, agent_id: str):
        """Add a dependency on another agent"""
        self.dependencies.add(agent_id)
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent's task - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement execute()")
    
    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run the agent with timing and error handling"""
        self.logger.info(f"Starting agent {self.agent_id}")
        self.status = "running"
        self.start_time = datetime.now()
        
        try:
            # Execute the agent's task
            self.result = await self.execute(context)
            self.status = "completed"
            
            return self.result
        except Exception as e:
            self.logger.error(f"Agent {self.agent_id} failed: {str(e)}")
            self.status = "failed"
            self.result = {
                "status": "failed",
                "error": str(e)
            }
            return self.result
        finally:
            self.end_time = datetime.now()

class DataCollectionAgent(Agent):
    """Agent for collecting data from external sources"""
    
    def __init__(self, agent_id: str, collector_class: Type, collector_config: Optional[Dict[str, Any]] = None):
        super().__init__(agent_id)
        self.collector_class = collector_class
        self.collector_config = collector_config or {}
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the data collection task"""
        try:
            # Create collector instance
            collector = self.collector_class(config=self.collector_config)
            
            # Run collection
            results = await collector.run(
                save_results=context.get("save_results", True),
                process_results=context.get("process_results", True)
            )
            
            # Add results to context
            return {
                "status": "success",
                "results": results,
                "processed_count": len(results) if results else 0,
                "collector_status": collector.get_status() if hasattr(collector, "get_status") else {}
            }
        except Exception as e:
            self.logger.error(f"Data collection failed: {str(e)}")
            raise

class EntityExtractionAgent(Agent):
    """Agent for extracting entities from collected data"""
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the entity extraction task"""
        # Import here to avoid circular imports
        from core.processors.entity_extractor import EntityExtractor
        
        try:
            extractor = EntityExtractor()
            processed_items = []
            processed_count = 0
            
            # Process data from dependencies
            for dep_id in self.dependencies:
                dep_results = context.get("results", {}).get(dep_id, {}).get("results", [])
                if not dep_results:
                    self.logger.warning(f"No results found for dependency {dep_id}")
                    continue
                
                self.logger.info(f"Processing {len(dep_results)} items from {dep_id}")
                
                for item in dep_results:
                    try:
                        processed_item = extractor.process_document(item)
                        processed_items.append(processed_item)
                        processed_count += 1
                    except Exception as e:
                        self.logger.error(f"Error processing item: {str(e)}")
            
            self.logger.info(f"Extracted entities from {processed_count} items")
            
            return {
                "status": "success",
                "results": processed_items,
                "processed_count": processed_count
            }
        except Exception as e:
            self.logger.error(f"Entity extraction failed: {str(e)}")
            raise

class KnowledgeGraphAgent(Agent):
    """Agent for storing processed data in the knowledge graph"""
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the knowledge graph storage task"""
        # Import here to avoid circular imports
        from core.knowledge.graph_store import KnowledgeGraph
        
        try:
            graph = KnowledgeGraph()
            stored_count = 0
            
            # Get data from dependencies
            for dep_id in self.dependencies:
                dep_results = context.get("results", {}).get(dep_id, {}).get("results", [])
                if not dep_results:
                    self.logger.warning(f"No results found for dependency {dep_id}")
                    continue
                
                self.logger.info(f"Storing {len(dep_results)} items from {dep_id}")
                
                for item in dep_results:
                    try:
                        # Store based on item type
                        if 'award_data' in item:
                            graph.add_contract_award(item['award_data'])
                            stored_count += 1
                        else:
                            # Add as opportunity or other entity type
                            graph.add_opportunity(item)
                            stored_count += 1
                    except Exception as e:
                        self.logger.error(f"Error storing item: {str(e)}")
            
            self.logger.info(f"Stored {stored_count} items in knowledge graph")
            
            return {
                "status": "success",
                "stored_count": stored_count
            }
        except Exception as e:
            self.logger.error(f"Knowledge graph storage failed: {str(e)}")
            raise

class AgentOrchestrator:
    """Orchestrates the execution of agents in the correct order"""
    
    def __init__(self):
        self.agents = {}
        self.logger = logging.getLogger("agent_orchestrator")
    
    def register_agent(self, agent: Agent) -> None:
        """Register an agent with the orchestrator"""
        self.agents[agent.agent_id] = agent
        self.logger.info(f"Registered agent: {agent.agent_id}")
    
    async def run_pipeline(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run all agents in the correct order based on dependencies"""
        self.logger.info("Starting agent pipeline")
        start_time = datetime.now()
        
        # Initialize results
        results = {}
        context["results"] = results
        
        # Determine execution order
        execution_order = self._build_execution_order()
        self.logger.info(f"Execution order: {execution_order}")
        
        # Execute agents in order
        overall_status = "completed"
        errors = []
        
        for agent_id in execution_order:
            agent = self.agents[agent_id]
            self.logger.info(f"Running agent: {agent_id}")
            
            try:
                # Check if dependencies have been met
                for dep_id in agent.dependencies:
                    if dep_id not in results or results[dep_id].get("status") != "success":
                        self.logger.warning(f"Dependency {dep_id} has not completed successfully")
                
                # Run the agent
                agent_result = await agent.run(context)
                results[agent_id] = agent_result
                
                # Update context for next agents
                if "results" not in context:
                    context["results"] = {}
                context["results"][agent_id] = agent_result
                
                if agent_result.get("status") != "success":
                    overall_status = "partial_failure"
                    errors.append(f"Agent {agent_id} returned status {agent_result.get('status')}")
            except Exception as e:
                self.logger.error(f"Error executing agent {agent_id}: {str(e)}")
                results[agent_id] = {
                    "status": "failed",
                    "error": str(e)
                }
                overall_status = "partial_failure"
                errors.append(f"Agent {agent_id} failed: {str(e)}")
        
        # Calculate duration
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        return {
            "status": overall_status,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration": duration,
            "execution_order": execution_order,
            "results": results,
            "errors": errors
        }
    
    def _build_execution_order(self) -> List[str]:
        """Build execution order based on dependencies"""
        # Find roots (agents with no dependencies)
        execution_order = []
        visited = set()
        
        # Topological sort
        def visit(agent_id):
            if agent_id in visited:
                return
            
            visited.add(agent_id)
            
            # Process dependencies first
            agent = self.agents.get(agent_id)
            if agent:
                for dep_id in agent.dependencies:
                    if dep_id in self.agents:
                        visit(dep_id)
            
            execution_order.append(agent_id)
        
        # Start with agents that have no dependencies
        roots = [agent_id for agent_id, agent in self.agents.items() if not agent.dependencies]
        
        # If no roots, start with any agent
        if not roots and self.agents:
            roots = [next(iter(self.agents.keys()))]
        
        # Visit each root
        for root in roots:
            visit(root)
        
        # Reverse to get correct execution order
        execution_order.reverse()
        
        return execution_order