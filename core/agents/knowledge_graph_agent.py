# core/agents/knowledge_graph_agent.py
from typing import Dict, List, Any, Optional, Union
import logging
import asyncio
import uuid

from core.agents.base_agent import BaseAgent
from core.knowledge.graph_store import KnowledgeGraph

class KnowledgeGraphAgent(BaseAgent):
    """Agent for managing and querying the knowledge graph."""
    
    def __init__(self, agent_id: str = None, name: str = None, 
                 uri: str = None, user: str = None, password: str = None, 
                 config: Dict[str, Any] = None):
        """Initialize a new knowledge graph agent.
        
        Args:
            agent_id: Unique identifier for the agent
            name: Human-readable name for the agent
            uri: URI for the Neo4j database
            user: Username for the Neo4j database
            password: Password for the Neo4j database
            config: Additional configuration for the agent
        """
        agent_config = config or {}
        if uri:
            agent_config["uri"] = uri
        if user:
            agent_config["user"] = user
        if password:
            agent_config["password"] = password
        
        super().__init__(agent_id, name or "knowledge_graph_agent", agent_config)
        
        self.graph = None
        self.logger.info("Initialized knowledge graph agent")
    
    def _get_graph(self) -> KnowledgeGraph:
        """Get or create a KnowledgeGraph instance.
        
        Returns:
            A KnowledgeGraph instance
        """
        if not self.graph:
            uri = self.config.get("uri")
            user = self.config.get("user")
            password = self.config.get("password")
            
            self.graph = KnowledgeGraph(uri=uri, user=user, password=password)
            
        return self.graph
    
    async def execute_task(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute knowledge graph task.
        
        Args:
            params: Parameters for the graph task
                - operation: Operation to perform (store, query, analyze)
                - data: Data for the operation
                - query: Query for the operation
                
        Returns:
            Results of the graph task
        """
        operation = params.get("operation")
        if not operation:
            raise ValueError("No operation specified for knowledge graph task")
        
        self.logger.info(f"Starting knowledge graph operation: {operation}")
        
        graph = self._get_graph()
        
        if operation == "store":
            return await self._store_data(graph, params)
        elif operation == "query":
            return await self._query_data(graph, params)
        elif operation == "analyze":
            return await self._analyze_data(graph, params)
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    async def _store_data(self, graph: KnowledgeGraph, params: Dict[str, Any]) -> Dict[str, Any]:
        """Store data in the knowledge graph.
        
        Args:
            graph: KnowledgeGraph instance
            params: Parameters for the storage operation
                - data: Data to store
                - data_type: Type of data (awards, entities, relationships)
                
        Returns:
            Results of the storage operation
        """
        data = params.get("data", [])
        data_type = params.get("data_type", "awards")
        
        if not data:
            return {"stored_count": 0, "message": "No data provided"}
        
        stored_count = 0
        
        if data_type == "awards":
            for item in data:
                if isinstance(item, dict) and "award_data" in item:
                    award_data = item["award_data"]
                    graph.add_contract_award(award_data)
                    stored_count += 1
                elif isinstance(item, dict) and all(k in item for k in ["agency_id", "opportunity_id"]):
                    graph.add_contract_award(item)
                    stored_count += 1
        elif data_type == "entities":
            # Implementation for storing entities directly
            # This would depend on specifics of the graph_store implementation
            pass
        elif data_type == "relationships":
            # Implementation for storing relationships directly
            # This would depend on specifics of the graph_store implementation
            pass
        
        return {
            "operation": "store",
            "data_type": data_type,
            "total_items": len(data),
            "stored_count": stored_count
        }
    
    async def _query_data(self, graph: KnowledgeGraph, params: Dict[str, Any]) -> Dict[str, Any]:
        """Query data from the knowledge graph.
        
        Args:
            graph: KnowledgeGraph instance
            params: Parameters for the query operation
                - query_type: Type of query (search, competitor, technology, entity)
                - query: Query parameters
                
        Returns:
            Results of the query operation
        """
        query_type = params.get("query_type")
        query = params.get("query", {})
        
        if query_type == "search":
            # Search for opportunities
            search_term = query.get("term", "")
            limit = query.get("limit", 10)
            results = graph.search_opportunities(search_term, limit)
            return {
                "operation": "query",
                "query_type": "search",
                "count": len(results),
                "results": results
            }
        elif query_type == "competitor":
            # Get competitor insights
            competitor_id = query.get("competitor_id")
            if not competitor_id:
                raise ValueError("No competitor_id provided for competitor query")
            
            results = graph.get_competitor_insights(competitor_id)
            return {
                "operation": "query",
                "query_type": "competitor",
                "competitor_id": competitor_id,
                "count": len(results),
                "results": results
            }
        elif query_type == "technology":
            # Get technology landscape
            results = graph.get_technology_landscape()
            return {
                "operation": "query",
                "query_type": "technology",
                "count": len(results),
                "results": results
            }
        elif query_type == "entity":
            # Get entity statistics
            entity_type = query.get("entity_type")
            results = graph.get_entity_statistics(entity_type)
            return {
                "operation": "query",
                "query_type": "entity",
                "entity_type": entity_type,
                "results": results
            }
        elif query_type == "opportunity":
            # Get opportunity details
            opportunity_id = query.get("opportunity_id")
            if not opportunity_id:
                raise ValueError("No opportunity_id provided for opportunity query")
            
            entities = graph.get_opportunity_entities(opportunity_id)
            similar = query.get("find_similar", False)
            
            result = {
                "operation": "query",
                "query_type": "opportunity",
                "opportunity_id": opportunity_id,
                "entities": entities
            }
            
            if similar:
                similar_opps = graph.find_similar_opportunities(opportunity_id, query.get("limit", 5))
                result["similar_opportunities"] = similar_opps
            
            return result
        else:
            raise ValueError(f"Unknown query_type: {query_type}")
    
    async def _analyze_data(self, graph: KnowledgeGraph, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze data in the knowledge graph.
        
        Args:
            graph: KnowledgeGraph instance
            params: Parameters for the analysis operation
                - analysis_type: Type of analysis to perform
                - analysis_params: Parameters for the analysis
                
        Returns:
            Results of the analysis operation
        """
        analysis_type = params.get("analysis_type")
        analysis_params = params.get("analysis_params", {})
        
        if analysis_type == "competitor_comparison":
            # Compare multiple competitors
            competitor_ids = analysis_params.get("competitor_ids", [])
            if not competitor_ids:
                raise ValueError("No competitor_ids provided for competitor comparison")
            
            comparison = {}
            for comp_id in competitor_ids:
                insights = graph.get_competitor_insights(comp_id)
                comparison[comp_id] = insights
            
            return {
                "operation": "analyze",
                "analysis_type": "competitor_comparison",
                "competitor_ids": competitor_ids,
                "comparison": comparison
            }
        elif analysis_type == "market_share":
            # Analyze market share within agencies or technology areas
            segment_by = analysis_params.get("segment_by", "agency")
            # This would require custom Cypher queries not currently in the graph_store
            # Implementation would depend on specifics of the graph_store
            return {
                "operation": "analyze",
                "analysis_type": "market_share",
                "segment_by": segment_by,
                "message": "Market share analysis not yet implemented"
            }
        else:
            raise ValueError(f"Unknown analysis_type: {analysis_type}")