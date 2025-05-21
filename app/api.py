# app/api.py
from fastapi import FastAPI, Depends, HTTPException, Security, BackgroundTasks
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import sys
import json
import asyncio
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("govcon.api")

# Add project root to path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.knowledge.graph_store import KnowledgeGraph
from core.collectors.sam_gov import SamGovCollector
from core.processors.entity_extractor import EntityExtractor
from core.agents.manager import AgentManager

# Initialize API
app = FastAPI(
    title="GovCon Intelligence API",
    description="API for the Government Contracting Competitive Intelligence System",
    version="0.2.0"
)

# Security
API_KEY = os.environ.get("API_KEY", "dev_key")
api_key_header = APIKeyHeader(name="X-API-Key")

# Initialize the entity extractor as a global variable
entity_extractor = None

# Initialize the agent manager as a global variable
agent_manager = None

def get_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

# Knowledge graph connection
def get_knowledge_graph():
    graph = KnowledgeGraph()
    try:
        yield graph
    finally:
        graph.close()

# Agent manager instance
def get_agent_manager():
    global agent_manager
    if not agent_manager:
        # Initialize agent manager with config file
        config_path = os.path.join("data", "agent_config.json")
        agent_manager = AgentManager(config_path=config_path if os.path.exists(config_path) else None)
    return agent_manager

# Models
class Query(BaseModel):
    query: str
    filters: Optional[Dict[str, Any]] = None

class SearchResult(BaseModel):
    id: str
    title: str
    type: str
    score: float
    data: Dict[str, Any]

class OpportunityAnalysisRequest(BaseModel):
    opportunity_id: Optional[str] = None
    title: str
    agency: str
    value: float
    description: str
    additional_info: Optional[str] = None

class CompetitorAnalysisRequest(BaseModel):
    competitor_id: str
    include_technologies: Optional[bool] = True
    include_agencies: Optional[bool] = True

class BidDecisionFeedbackRequest(BaseModel):
    opportunity_id: str
    bid_decision: bool
    win_result: Optional[bool] = None
    feedback_notes: Optional[str] = None

class WorkflowRequest(BaseModel):
    workflow_type: str
    params: Dict[str, Any]

class MarketIntelligenceRequest(BaseModel):
    agency_id: Optional[str] = None
    technology: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global entity_extractor, agent_manager
    
    logger.info("Initializing entity extractor...")
    entity_extractor = EntityExtractor()
    logger.info("Entity extractor initialized")
    
    logger.info("Initializing agent manager...")
    agent_manager = get_agent_manager()
    logger.info("Agent manager initialized")

# Routes
@app.get("/")
def read_root():
    return {
        "status": "operational",
        "system": "Government Contracting Competitive Intelligence System",
        "version": "0.2.0"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/search")
def search(query: Query, graph: KnowledgeGraph = Depends(get_knowledge_graph), api_key: str = Depends(get_api_key)):
    """Search the knowledge graph"""
    try:
        results = graph.search_opportunities(query.query)
        return results
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.get("/collect")
async def collect_data(api_key: str = Depends(get_api_key), manager: AgentManager = Depends(get_agent_manager)):
    """Manually trigger data collection using the agent system"""
    try:
        logger.info("Starting data collection workflow")
        
        # Run the data collection workflow
        result = await manager.run_workflow(
            workflow_type="data_collection",
            params={
                "id": f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "collector_type": "core.collectors.sam_gov.SamGovCollector"
            }
        )
        
        return {
            "status": "success",
            "workflow_id": result.get("workflow_id"),
            "results": result.get("results", {})
        }
    except Exception as e:
        logger.error(f"Collection error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Collection error: {str(e)}")

@app.post("/extract-entities")
async def extract_entities(request: Dict[str, Any], api_key: str = Depends(get_api_key)):
    """Extract entities from provided text"""
    global entity_extractor
    
    if not entity_extractor:
        entity_extractor = EntityExtractor()
    
    text = request.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")
    
    try:
        entities = entity_extractor.extract_entities(text)
        summary = entity_extractor._generate_entity_summary(entities)
        
        return {
            "entities": entities,
            "summary": summary
        }
    except Exception as e:
        logger.error(f"Entity extraction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Entity extraction error: {str(e)}")

@app.get("/competitors")
def list_competitors(graph: KnowledgeGraph = Depends(get_knowledge_graph), api_key: str = Depends(get_api_key)):
    """List all competitors in the knowledge graph"""
    try:
        # This would normally query the knowledge graph
        # Placeholder implementation
        return [
            {"id": "comp1", "name": "TechGov Solutions", "contract_count": 25},
            {"id": "comp2", "name": "Federal Systems Inc", "contract_count": 18},
            {"id": "comp3", "name": "Government IT Partners", "contract_count": 12}
        ]
    except Exception as e:
        logger.error(f"Competitor listing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Competitor listing error: {str(e)}")

@app.get("/agencies")
def list_agencies(graph: KnowledgeGraph = Depends(get_knowledge_graph), api_key: str = Depends(get_api_key)):
    """List all agencies in the knowledge graph"""
    try:
        # This would normally query the knowledge graph
        # Placeholder implementation
        return [
            {"id": "agency1", "name": "Department of Defense", "total_spend": 25000000000},
            {"id": "agency2", "name": "Department of Health & Human Services", "total_spend": 15000000000},
            {"id": "agency3", "name": "General Services Administration", "total_spend": 8000000000}
        ]
    except Exception as e:
        logger.error(f"Agency listing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Agency listing error: {str(e)}")

@app.get("/entity-stats")
def get_entity_statistics(graph: KnowledgeGraph = Depends(get_knowledge_graph), api_key: str = Depends(get_api_key)):
    """Get statistics on extracted entities"""
    try:
        # This would normally query the knowledge graph
        # Placeholder implementation
        return {
            "technology": {
                "total_count": 45,
                "top_entities": [
                    {"name": "cloud", "count": 12},
                    {"name": "cybersecurity", "count": 10},
                    {"name": "artificial intelligence", "count": 8}
                ]
            },
            "regulation": {
                "total_count": 30,
                "top_entities": [
                    {"name": "NIST", "count": 15},
                    {"name": "CMMC", "count": 8},
                    {"name": "FedRAMP", "count": 7}
                ]
            },
            "clearance": {
                "total_count": 20,
                "top_entities": [
                    {"name": "Top Secret", "count": 10},
                    {"name": "Secret", "count": 8},
                    {"name": "Public Trust", "count": 2}
                ]
            }
        }
    except Exception as e:
        logger.error(f"Entity statistics error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Entity statistics error: {str(e)}")

# New agent-based endpoints

@app.post("/analyze-opportunity")
async def analyze_opportunity(
    request: OpportunityAnalysisRequest, 
    api_key: str = Depends(get_api_key),
    manager: AgentManager = Depends(get_agent_manager)
):
    """Analyze an opportunity using the agent system"""
    try:
        logger.info(f"Starting opportunity analysis workflow for: {request.title}")
        
        # Prepare opportunity data
        opportunity_data = {
            "id": request.opportunity_id or f"opp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "title": request.title,
            "agency": request.agency,
            "value": request.value,
            "description": request.description,
            "additional_info": request.additional_info or ""
        }
        
        # Run the opportunity analysis workflow
        result = await manager.run_workflow(
            workflow_type="opportunity_analysis",
            params={
                "id": opportunity_data["id"],
                "opportunity_data": opportunity_data
            }
        )
        
        return {
            "status": "success",
            "opportunity_id": opportunity_data["id"],
            "workflow_id": result.get("workflow_id"),
            "analysis": result.get("results", {}).get("bid_result", {})
        }
    except Exception as e:
        logger.error(f"Opportunity analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Opportunity analysis error: {str(e)}")

@app.post("/analyze-competitor")
async def analyze_competitor(
    request: CompetitorAnalysisRequest, 
    api_key: str = Depends(get_api_key),
    manager: AgentManager = Depends(get_agent_manager)
):
    """Analyze a competitor using the agent system"""
    try:
        logger.info(f"Starting competitor analysis workflow for: {request.competitor_id}")
        
        # Run the competitor analysis workflow
        result = await manager.run_workflow(
            workflow_type="competitor_analysis",
            params={
                "id": request.competitor_id,
                "competitor_id": request.competitor_id,
                "include_technologies": request.include_technologies,
                "include_agencies": request.include_agencies
            }
        )
        
        return {
            "status": "success",
            "competitor_id": request.competitor_id,
            "workflow_id": result.get("workflow_id"),
            "analysis": {
                "competitor_data": result.get("results", {}).get("competitor_data", {}),
                "technology_data": result.get("results", {}).get("technology_data", {}) if request.include_technologies else None,
                "insights": result.get("results", {}).get("insights", {})
            }
        }
    except Exception as e:
        logger.error(f"Competitor analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Competitor analysis error: {str(e)}")

@app.post("/bid-feedback")
async def record_bid_feedback(
    request: BidDecisionFeedbackRequest, 
    api_key: str = Depends(get_api_key),
    manager: AgentManager = Depends(get_agent_manager)
):
    """Record feedback on a bid decision"""
    try:
        logger.info(f"Recording bid feedback for opportunity: {request.opportunity_id}")
        
        # Run the bid decision feedback workflow
        result = await manager.run_workflow(
            workflow_type="bid_decision_feedback",
            params={
                "id": f"feedback_{request.opportunity_id}",
                "opportunity_id": request.opportunity_id,
                "bid_decision": request.bid_decision,
                "win_result": request.win_result,
                "feedback_notes": request.feedback_notes or ""
            }
        )
        
        return {
            "status": "success",
            "opportunity_id": request.opportunity_id,
            "workflow_id": result.get("workflow_id"),
            "feedback_id": result.get("results", {}).get("feedback_result", {}).get("feedback_id")
        }
    except Exception as e:
        logger.error(f"Bid feedback error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Bid feedback error: {str(e)}")

@app.post("/market-intelligence")
async def get_market_intelligence(
    request: MarketIntelligenceRequest, 
    api_key: str = Depends(get_api_key),
    manager: AgentManager = Depends(get_agent_manager)
):
    """Get market intelligence using the agent system"""
    try:
        logger.info("Starting market intelligence workflow")
        
        # Prepare time period if provided
        time_period = None
        if request.start_date or request.end_date:
            time_period = {}
            if request.start_date:
                time_period["start_date"] = request.start_date
            if request.end_date:
                time_period["end_date"] = request.end_date
        
        # Run the market intelligence workflow
        result = await manager.run_workflow(
            workflow_type="market_intelligence",
            params={
                "id": f"market_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "agency_id": request.agency_id,
                "technology": request.technology,
                "time_period": time_period
            }
        )
        
        return {
            "status": "success",
            "workflow_id": result.get("workflow_id"),
            "intelligence": {
                "agency_data": result.get("results", {}).get("agency_data", {}),
                "technology_data": result.get("results", {}).get("technology_data", {}),
                "agency_insights": result.get("results", {}).get("agency_insights", {}),
                "technology_insights": result.get("results", {}).get("technology_insights", {}),
                "success_factors": result.get("results", {}).get("success_factors", {})
            }
        }
    except Exception as e:
        logger.error(f"Market intelligence error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Market intelligence error: {str(e)}")

@app.post("/run-workflow")
async def run_workflow(
    request: WorkflowRequest, 
    api_key: str = Depends(get_api_key),
    manager: AgentManager = Depends(get_agent_manager)
):
    """Run a custom workflow"""
    try:
        logger.info(f"Starting custom workflow: {request.workflow_type}")
        
        # Run the workflow
        result = await manager.run_workflow(
            workflow_type=request.workflow_type,
            params=request.params
        )
        
        return {
            "status": "success",
            "workflow_type": request.workflow_type,
            "workflow_id": result.get("workflow_id"),
            "results": result.get("results", {})
        }
    except Exception as e:
        logger.error(f"Workflow error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Workflow error: {str(e)}")

@app.get("/workflow-state/{workflow_id}")
def get_workflow_state(
    workflow_id: str, 
    api_key: str = Depends(get_api_key),
    manager: AgentManager = Depends(get_agent_manager)
):
    """Get the current state of a workflow"""
    try:
        logger.info(f"Getting state for workflow: {workflow_id}")
        
        state = manager.get_workflow_state(workflow_id)
        if not state:
            raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
        
        return {
            "status": "success",
            "workflow_id": workflow_id,
            "state": state
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Workflow state error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Workflow state error: {str(e)}")

@app.get("/agents")
def list_agents(
    api_key: str = Depends(get_api_key),
    manager: AgentManager = Depends(get_agent_manager)
):
    """List all agents in the system"""
    try:
        logger.info("Listing all agents")
        
        states = manager.get_agent_states()
        
        return {
            "status": "success",
            "agent_count": len(states),
            "agents": states
        }
    except Exception as e:
        logger.error(f"Agent listing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Agent listing error: {str(e)}")

@app.post("/agents/{agent_id}/run")
async def run_agent(
    agent_id: str,
    params: Dict[str, Any],
    api_key: str = Depends(get_api_key),
    manager: AgentManager = Depends(get_agent_manager)
):
    """Run a specific agent with given parameters"""
    try:
        logger.info(f"Running agent: {agent_id}")
        
        # Check if agent exists
        agent = manager.get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        # Run the agent
        result = await manager.run_agent(agent_id, params)
        
        return {
            "status": "success",
            "agent_id": agent_id,
            "task_id": result.get("task_id"),
            "results": result.get("results", {})
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Agent execution error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Agent execution error: {str(e)}")