# core/agents/workflows.py
from typing import Dict, List, Any

# Define standard workflows for the agent orchestrator

class Workflows:
    """Standard workflow definitions for the agent system."""
    
    @staticmethod
    def data_collection_workflow(collector_type: str = "core.collectors.sam_gov.SamGovCollector",
                              collector_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a data collection workflow.
        
        Args:
            collector_type: Type of collector to use
            collector_config: Configuration for the collector
            
        Returns:
            Workflow definition
        """
        return {
            "name": "Data Collection Workflow",
            "description": "Collect data from a source, extract entities, and store in knowledge graph",
            "steps": [
                {
                    "name": "Collect Data",
                    "agent_id": "collection_agent",
                    "result_key": "collection_result",
                    "params": {
                        "collector_type": collector_type,
                        "collector_config": collector_config or {}
                    }
                },
                {
                    "name": "Extract Entities",
                    "agent_id": "entity_analysis_agent",
                    "result_key": "entity_result",
                    "params": {
                        "documents": "$collection_result.items",
                        "analysis_type": "full"
                    }
                },
                {
                    "name": "Store in Knowledge Graph",
                    "agent_id": "knowledge_graph_agent",
                    "result_key": "storage_result",
                    "params": {
                        "operation": "store",
                        "data_type": "awards",
                        "data": "$entity_result.processed_documents"
                    }
                }
            ]
        }
    
    @staticmethod
    def opportunity_analysis_workflow(opportunity_data: Dict[str, Any],
                                    company_profile: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create an opportunity analysis workflow.
        
        Args:
            opportunity_data: The opportunity to analyze
            company_profile: Profile of the company for matching
            
        Returns:
            Workflow definition
        """
        return {
            "name": "Opportunity Analysis Workflow",
            "description": "Analyze an opportunity for entity extraction, bid/no-bid decision",
            "steps": [
                {
                    "name": "Extract Entities",
                    "agent_id": "entity_analysis_agent",
                    "result_key": "entity_result",
                    "params": {
                        "text": opportunity_data.get("description", ""),
                        "analysis_type": "full"
                    }
                },
                {
                    "name": "Store Extracted Entities",
                    "agent_id": "knowledge_graph_agent",
                    "result_key": "storage_result",
                    "params": {
                        "operation": "store",
                        "data_type": "entities",
                        "data": [
                            {
                                "opportunity_id": opportunity_data.get("id") or opportunity_data.get("opportunity_id"),
                                "entity_summary": "$entity_result.text_analysis.summary",
                                "extracted_entities": "$entity_result.text_analysis.entities"
                            }
                        ]
                    }
                },
                {
                    "name": "Bid Analysis",
                    "agent_id": "bid_analysis_agent",
                    "result_key": "bid_result",
                    "params": {
                        "opportunities": [
                            {
                                "id": opportunity_data.get("id") or opportunity_data.get("opportunity_id"),
                                "title": opportunity_data.get("title", ""),
                                "agency": opportunity_data.get("agency", ""),
                                "value": opportunity_data.get("value", 0),
                                "description": opportunity_data.get("description", ""),
                                "entity_summary": "$entity_result.text_analysis.summary"
                            }
                        ],
                        "company_profile": company_profile
                    }
                }
            ]
        }
    
    @staticmethod
    def competitor_analysis_workflow(competitor_id: str) -> Dict[str, Any]:
        """Create a competitor analysis workflow.
        
        Args:
            competitor_id: ID of the competitor to analyze
            
        Returns:
            Workflow definition
        """
        return {
            "name": "Competitor Analysis Workflow",
            "description": "Analyze a competitor's contract history and capabilities",
            "steps": [
                {
                    "name": "Query Competitor Data",
                    "agent_id": "knowledge_graph_agent",
                    "result_key": "competitor_data",
                    "params": {
                        "operation": "query",
                        "query_type": "competitor",
                        "query": {
                            "competitor_id": competitor_id
                        }
                    }
                },
                {
                    "name": "Query Technology Landscape",
                    "agent_id": "knowledge_graph_agent",
                    "result_key": "technology_data",
                    "params": {
                        "operation": "query",
                        "query_type": "technology"
                    }
                },
                {
                    "name": "Generate Insights",
                    "agent_id": "feedback_learning_agent",
                    "result_key": "insights",
                    "params": {
                        "operation": "get_insights",
                        "insight_type": "summary",
                        "filters": {
                            "competitor_id": competitor_id
                        }
                    }
                }
            ]
        }
    
    @staticmethod
    def bid_decision_feedback_workflow(opportunity_id: str, 
                                     bid_decision: bool, 
                                     win_result: bool = None,
                                     feedback_notes: str = "") -> Dict[str, Any]:
        """Create a workflow for recording bid decision feedback.
        
        Args:
            opportunity_id: ID of the opportunity
            bid_decision: Whether the company bid on the opportunity
            win_result: Whether the company won the bid (if applicable)
            feedback_notes: Notes about the feedback
            
        Returns:
            Workflow definition
        """
        return {
            "name": "Bid Decision Feedback Workflow",
            "description": "Record feedback on bid decision and update learning model",
            "steps": [
                {
                    "name": "Query Opportunity Data",
                    "agent_id": "knowledge_graph_agent",
                    "result_key": "opportunity_data",
                    "params": {
                        "operation": "query",
                        "query_type": "opportunity",
                        "query": {
                            "opportunity_id": opportunity_id
                        }
                    }
                },
                {
                    "name": "Add Feedback",
                    "agent_id": "feedback_learning_agent",
                    "result_key": "feedback_result",
                    "params": {
                        "operation": "add_feedback",
                        "feedback": {
                            "opportunity_id": opportunity_id,
                            "bid_decision": bid_decision,
                            "win_result": win_result,
                            "feedback_notes": feedback_notes,
                            "opportunity_data": "$opportunity_data"
                        }
                    }
                },
                {
                    "name": "Update Model",
                    "agent_id": "feedback_learning_agent",
                    "result_key": "model_update",
                    "params": {
                        "operation": "update_model"
                    }
                }
            ]
        }
    
    @staticmethod
    def market_intelligence_workflow(agency_id: str = None, 
                                   technology: str = None,
                                   time_period: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a market intelligence workflow.
        
        Args:
            agency_id: Optional agency to focus on
            technology: Optional technology to focus on
            time_period: Optional time period to consider
            
        Returns:
            Workflow definition
        """
        query_params = {}
        if agency_id:
            query_params["agency_id"] = agency_id
        if technology:
            query_params["technology"] = technology
        if time_period:
            query_params["time_period"] = time_period
        
        return {
            "name": "Market Intelligence Workflow",
            "description": "Generate market intelligence for agencies and technologies",
            "steps": [
                {
                    "name": "Query Agency Data",
                    "agent_id": "knowledge_graph_agent",
                    "result_key": "agency_data",
                    "params": {
                        "operation": "query",
                        "query_type": "entity",
                        "query": {
                            "entity_type": "AGENCY",
                            **query_params
                        }
                    }
                },
                {
                    "name": "Query Technology Data",
                    "agent_id": "knowledge_graph_agent",
                    "result_key": "technology_data",
                    "params": {
                        "operation": "query",
                        "query_type": "entity",
                        "query": {
                            "entity_type": "TECHNOLOGY",
                            **query_params
                        }
                    }
                },
                {
                    "name": "Get Agency Insights",
                    "agent_id": "feedback_learning_agent",
                    "result_key": "agency_insights",
                    "params": {
                        "operation": "get_insights",
                        "insight_type": "agency",
                        "filters": query_params
                    }
                },
                {
                    "name": "Get Technology Insights",
                    "agent_id": "feedback_learning_agent",
                    "result_key": "technology_insights",
                    "params": {
                        "operation": "get_insights",
                        "insight_type": "technology",
                        "filters": query_params
                    }
                },
                {
                    "name": "Get Success Factors",
                    "agent_id": "feedback_learning_agent",
                    "result_key": "success_factors",
                    "params": {
                        "operation": "get_insights",
                        "insight_type": "success_factors",
                        "filters": query_params
                    }
                }
            ]
        }