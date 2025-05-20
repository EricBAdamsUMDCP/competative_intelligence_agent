# app/api.py
from fastapi import FastAPI, Depends, HTTPException, Security, Query as FastAPIQuery
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import sys
import asyncio
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("govcon.api")

# Add project root to path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.knowledge.graph_store import KnowledgeGraph
from core.collectors.sam_gov import SamGovCollector
from core.collectors.usa_spending import USASpendingCollector
from core.collectors.news_scraper import NewsCollector
from core.processors.entity_extractor import EntityExtractor

# Initialize API
app = FastAPI(
    title="GovCon Intelligence API",
    description="API for the Government Contracting Competitive Intelligence System",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
API_KEY = os.environ.get("API_KEY", "dev_key")
api_key_header = APIKeyHeader(name="X-API-Key")

# SAM.gov API key tracking
SAM_GOV_API_KEY = os.environ.get("SAM_GOV_API_KEY", "")
SAM_GOV_API_KEY_CREATED = os.environ.get("SAM_GOV_API_KEY_CREATED", "")

# Initialize the entity extractor as a global variable
entity_extractor = None

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

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global entity_extractor
    logger.info("Initializing entity extractor...")
    entity_extractor = EntityExtractor()
    logger.info("Entity extractor initialized")
    
    # Check API key age if environment variable exists
    if SAM_GOV_API_KEY_CREATED:
        try:
            created_date = datetime.fromisoformat(SAM_GOV_API_KEY_CREATED)
            days_old = (datetime.now() - created_date).days
            if days_old > 80:  # Warn when approaching 90 days
                logger.warning(f"SAM.gov API key is {days_old} days old. Keys must be rotated every 90 days.")
            elif days_old > 90:
                logger.error(f"SAM.gov API key is {days_old} days old and has expired. Please rotate your key immediately.")
        except ValueError:
            logger.warning("Could not parse SAM_GOV_API_KEY_CREATED date. Please use ISO format (YYYY-MM-DD).")

# Routes
@app.get("/")
def read_root():
    return {
        "status": "operational",
        "system": "Government Contracting Intelligence System",
        "version": "0.1.0"
    }

@app.get("/health")
def health_check():
    """System health check endpoint"""
    # Check API key status
    api_key_status = "unknown"
    api_key_days = None
    
    if SAM_GOV_API_KEY_CREATED:
        try:
            created_date = datetime.fromisoformat(SAM_GOV_API_KEY_CREATED)
            days_old = (datetime.now() - created_date).days
            api_key_days = days_old
            
            if days_old > 90:
                api_key_status = "expired"
            elif days_old > 80:
                api_key_status = "warning"
            else:
                api_key_status = "valid"
        except ValueError:
            api_key_status = "invalid_date"
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "api": "operational",
            "knowledge_graph": "operational",
            "entity_extractor": "operational" if entity_extractor is not None else "not_initialized"
        },
        "sam_gov_api_key": {
            "status": api_key_status,
            "days_old": api_key_days,
            "present": bool(SAM_GOV_API_KEY)
        }
    }

@app.post("/search")
def search(query: Query, graph: KnowledgeGraph = Depends(get_knowledge_graph), api_key: str = Depends(get_api_key)):
    """Search the knowledge graph"""
    try:
        results = graph.search_opportunities(query.query)
        return results
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

from typing import Annotated

@app.get("/collect")
async def collect_data(
    sources: Annotated[Optional[str], FastAPIQuery(default=None, description="Comma-separated list of sources to collect from")],
    api_key: str = Depends(get_api_key)
):
    """Manually trigger data collection from specified sources or all sources"""
    global entity_extractor
    
    # Determine which sources to collect from
    source_list = sources.split(",") if sources else ["sam.gov", "usaspending.gov", "industry_news"]
    
    results = {}
    all_items = []
    
    try:
        # Initialize collectors based on requested sources
        collectors = []
        if "sam.gov" in source_list:
            collectors.append(SamGovCollector())
        if "usaspending.gov" in source_list:
            collectors.append(USASpendingCollector())
        if "industry_news" in source_list:
            collectors.append(NewsCollector())
        
        # Run all collectors
        for collector in collectors:
            source_results = await collector.run()
            results[collector.source_name] = len(source_results)
            all_items.extend(source_results)
        
        logger.info(f"Collected a total of {len(all_items)} items from {len(collectors)} sources")
        
        # Process entities
        processed_results = []
        for item in all_items:
            if entity_extractor:
                item = entity_extractor.process_document(item)
            processed_results.append(item)
        
        # Store in knowledge graph
        graph = KnowledgeGraph()
        
        try:
            count = 0
            for item in processed_results:
                if 'award_data' in item:
                    # Add extracted entities to award data
                    if 'extracted_entities' in item:
                        item['award_data']['extracted_entities'] = item['extracted_entities']
                    if 'entity_summary' in item:
                        item['award_data']['entity_summary'] = item['entity_summary']
                    
                    graph.add_contract_award(item['award_data'])
                    count += 1
            
            return {
                "status": "success",
                "sources": results,
                "total_collected": len(all_items),
                "processed": len(processed_results),
                "stored": count
            }
        finally:
            graph.close()
            
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

@app.get("/sources")
def get_data_sources(api_key: str = Depends(get_api_key)):
    """Get information about available data sources"""
    return {
        "sources": [
            {
                "id": "sam.gov",
                "name": "SAM.gov",
                "description": "Federal contract opportunities and awards",
                "last_updated": None,
                "status": "active"
            },
            {
                "id": "usaspending.gov",
                "name": "USASpending.gov",
                "description": "Federal spending data and contract details",
                "last_updated": None,
                "status": "active"
            },
            {
                "id": "industry_news",
                "name": "Industry News",
                "description": "Government contracting news from industry sources",
                "last_updated": None,
                "status": "active"
            }
        ]
    }

@app.get("/source/{source_id}")
def get_source_data(source_id: str, api_key: str = Depends(get_api_key)):
    """Get detailed information about a specific data source"""
    sources = {
        "sam.gov": {
            "id": "sam.gov",
            "name": "SAM.gov",
            "description": "Federal contract opportunities and awards",
            "endpoint": "https://api.sam.gov/opportunities/v1/search",
            "data_types": ["opportunities", "awards", "entities"],
            "last_updated": None,
            "item_count": 0,
            "status": "active",
            "terms_of_use": "Data provided by SAM.gov is subject to specific terms of use. By using this system, you agree to only access data you are authorized to access, not use the data for unauthorized purposes, not share your API key with unauthorized parties, and update your API key every 90 days.",
            "api_key_status": {
                "present": bool(SAM_GOV_API_KEY),
                "created": SAM_GOV_API_KEY_CREATED,
                "requires_update": False
            }
        },
        "usaspending.gov": {
            "id": "usaspending.gov",
            "name": "USASpending.gov",
            "description": "Federal spending data and contract details",
            "endpoint": "https://api.usaspending.gov/api/v2/search/spending_by_award/",
            "data_types": ["awards", "agencies", "recipients"],
            "last_updated": None,
            "item_count": 0,
            "status": "active" 
        },
        "industry_news": {
            "id": "industry_news",
            "name": "Industry News",
            "description": "Government contracting news from industry sources",
            "sources": ["Washington Technology", "Federal News Network", "FCW"],
            "data_types": ["news", "trends", "announcements"],
            "last_updated": None,
            "item_count": 0,
            "status": "active"
        }
    }
    
    if source_id not in sources:
        raise HTTPException(status_code=404, detail=f"Source {source_id} not found")
    
    # Add API key status for SAM.gov
    if source_id == "sam.gov" and SAM_GOV_API_KEY_CREATED:
        try:
            created_date = datetime.fromisoformat(SAM_GOV_API_KEY_CREATED)
            days_old = (datetime.now() - created_date).days
            sources[source_id]["api_key_status"]["days_old"] = days_old
            sources[source_id]["api_key_status"]["requires_update"] = days_old > 80
        except ValueError:
            pass
    
    return sources[source_id]

@app.get("/api-key-status")
def check_api_key_status(api_key: str = Depends(get_api_key)):
    """Check the status of API keys"""
    result = {
        "sam_gov": {
            "key_present": bool(SAM_GOV_API_KEY),
            "created_date": None,
            "days_old": None,
            "status": "unknown",
            "requires_update": False
        }
    }
    
    if SAM_GOV_API_KEY_CREATED:
        try:
            created_date = datetime.fromisoformat(SAM_GOV_API_KEY_CREATED)
            days_old = (datetime.now() - created_date).days
            
            result["sam_gov"]["created_date"] = SAM_GOV_API_KEY_CREATED
            result["sam_gov"]["days_old"] = days_old
            
            if days_old > 90:
                result["sam_gov"]["status"] = "expired"
                result["sam_gov"]["requires_update"] = True
            elif days_old > 80:
                result["sam_gov"]["status"] = "warning"
                result["sam_gov"]["requires_update"] = True
            else:
                result["sam_gov"]["status"] = "valid"
                result["sam_gov"]["requires_update"] = False
        except ValueError:
            result["sam_gov"]["status"] = "invalid_date_format"
    
    return result