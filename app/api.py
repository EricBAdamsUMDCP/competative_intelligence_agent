# Updated API code to integrate new collectors
# app/api.py

from fastapi import FastAPI, Depends, HTTPException, Security, BackgroundTasks, Query
from fastapi.security import APIKeyHeader
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
from core.collectors.usaspending_gov import USASpendingCollector
from core.processors.entity_extractor import EntityExtractor
from core.processors.data_normalizer import DataNormalizer

# Initialize API
app = FastAPI(
    title="GovCon Intelligence API",
    description="API for the Government Contracting Competitive Intelligence System",
    version="1.0.0"
)

# Security
API_KEY = os.environ.get("API_KEY", "dev_key")
api_key_header = APIKeyHeader(name="X-API-Key")

# Initialize global components
entity_extractor = None
data_normalizer = None

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

class CollectionParams(BaseModel):
    sources: List[str] = ["sam.gov", "usaspending.gov"]
    days: int = 30
    limit: int = 100

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
    global entity_extractor, data_normalizer
    
    logger.info("Initializing entity extractor...")
    entity_extractor = EntityExtractor()
    logger.info("Entity extractor initialized")
    
    logger.info("Initializing data normalizer...")
    data_normalizer = DataNormalizer()
    logger.info("Data normalizer initialized")

# Routes
@app.get("/")
def read_root():
    return {
        "status": "operational",
        "system": "Government Contracting Intelligence System",
        "version": "1.0.0"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/search")
def search(query: Query, graph: KnowledgeGraph = Depends(get_knowledge_graph), api_key: str = Depends(get_api_key)):
    """Search the knowledge graph"""
    try:
        # Apply filters if provided
        filter_params = {}
        if query.filters:
            if 'agency' in query.filters:
                filter_params['agency'] = query.filters['agency']
            if 'min_value' in query.filters:
                filter_params['min_value'] = query.filters['min_value']
            if 'max_value' in query.filters:
                filter_params['max_value'] = query.filters['max_value']
            if 'date_from' in query.filters:
                filter_params['date_from'] = query.filters['date_from']
            if 'date_to' in query.filters:
                filter_params['date_to'] = query.filters['date_to']
        
        results = graph.search_opportunities(query.query, filters=filter_params)
        return results
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.post("/collect")
async def collect_data(
    params: Optional[CollectionParams] = None,
    background_tasks: BackgroundTasks = None,
    api_key: str = Depends(get_api_key)
):
    """Manually trigger data collection"""
    global entity_extractor, data_normalizer
    
    # Use default params if not provided
    if not params:
        params = CollectionParams()
    
    # Start date for collection
    start_date = datetime.now() - timedelta(days=params.days)
    
    # If background tasks is provided, run collection in background
    if background_tasks:
        background_tasks.add_task(
            _run_data_collection, 
            params.sources, 
            start_date, 
            params.limit,
            entity_extractor,
            data_normalizer
        )
        return {
            "status": "initiated",
            "message": f"Data collection started in background for sources: {', '.join(params.sources)}",
            "params": {
                "days": params.days,
                "limit": params.limit
            }
        }
    
    # Otherwise run synchronously
    try:
        result = await _run_data_collection(
            params.sources, 
            start_date, 
            params.limit,
            entity_extractor,
            data_normalizer
        )
        return result
    except Exception as e:
        logger.error(f"Collection error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Collection error: {str(e)}")

async def _run_data_collection(
    sources: List[str], 
    start_date: datetime,
    limit: int,
    entity_extractor,
    data_normalizer
) -> Dict[str, Any]:
    """Run data collection from specified sources"""
    all_results = []
    processed_results = []
    collection_stats = {}
    
    # Create date strings in different formats for different APIs
    iso_date = start_date.isoformat()
    sam_date = start_date.strftime("%m/%d/%Y")
    
    # Collect from each source
    for source in sources:
        try:
            if source.lower() == "sam.gov":
                # Initialize SAM.gov collector with date range
                collector = SamGovCollector(config={
                    'published_since': iso_date,
                    'posted_date_start': sam_date,
                    'posted_date_end': datetime.now().strftime("%m/%d/%Y"),
                    'limit': limit
                })
                
                # Run collection
                source_results = await collector.run()
                
                collection_stats["sam.gov"] = {
                    "collected": len(source_results)
                }
                
                all_results.extend(source_results)
                
            elif source.lower() == "usaspending.gov":
                # Initialize USASpending collector with date range
                collector = USASpendingCollector(config={
                    'time_period': {
                        'start_date': start_date.strftime('%Y-%m-%d'),
                        'end_date': datetime.now().strftime('%Y-%m-%d')
                    },
                    'limit': limit
                })
                
                # Run collection
                source_results = await collector.run()
                
                collection_stats["usaspending.gov"] = {
                    "collected": len(source_results)
                }
                
                all_results.extend(source_results)
            
            # Add additional collectors here
            
        except Exception as e:
            logger.error(f"Error collecting from {source}: {str(e)}")
            collection_stats[source] = {
                "error": str(e)
            }
    
    logger.info(f"Collected {len(all_results)} total items from all sources")
    
    # Process entities and normalize data
    for item in all_results:
        try:
            # Extract entities
            if entity_extractor:
                item = entity_extractor.process_document(item)
            
            # Normalize data
            if data_normalizer and 'source' in item:
                item = data_normalizer.normalize(item, item['source'])
            
            processed_results.append(item)
        except Exception as e:
            logger.error(f"Error processing item: {str(e)}")
    
    # Store in knowledge graph
    graph = KnowledgeGraph()
    
    try:
        stored_count = 0
        for item in processed_results:
            try:
                if 'award_data' in item:
                    # Add extracted entities to award data if present
                    if 'extracted_entities' in item:
                        item['award_data']['extracted_entities'] = item['extracted_entities']
                    if 'entity_summary' in item:
                        item['award_data']['entity_summary'] = item['entity_summary']
                    
                    graph.add_contract_award(item['award_data'])
                    stored_count += 1
            except Exception as e:
                logger.error(f"Error storing item in knowledge graph: {str(e)}")
        
        return {
            "status": "success",
            "collected": len(all_results),
            "processed": len(processed_results),
            "stored": stored_count,
            "source_stats": collection_stats
        }
    finally:
        graph.close()

@app.get("/collection/status")
async def collection_status(task_id: str, api_key: str = Depends(get_api_key)):
    """Get status of a background collection task"""
    # In a production system, this would check a task queue or database
    # For now, return a placeholder
    return {
        "task_id": task_id,
        "status": "completed",
        "message": "Collection task completed successfully"
    }

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
        competitors = graph.get_competitors()
        return competitors
    except Exception as e:
        logger.error(f"Competitor listing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Competitor listing error: {str(e)}")

@app.get("/agencies")
def list_agencies(graph: KnowledgeGraph = Depends(get_knowledge_graph), api_key: str = Depends(get_api_key)):
    """List all agencies in the knowledge graph"""
    try:
        agencies = graph.get_agencies()
        return agencies
    except Exception as e:
        logger.error(f"Agency listing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Agency listing error: {str(e)}")

@app.get("/entity-stats")
def get_entity_statistics(
    entity_type: Optional[str] = None,
    graph: KnowledgeGraph = Depends(get_knowledge_graph),
    api_key: str = Depends(get_api_key)
):
    """Get statistics on extracted entities"""
    try:
        stats = graph.get_entity_statistics(entity_type)
        return stats
    except Exception as e:
        logger.error(f"Entity statistics error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Entity statistics error: {str(e)}")

@app.get("/sources")
def list_data_sources(api_key: str = Depends(get_api_key)):
    """List available data sources"""
    return {
        "sources": [
            {
                "id": "sam.gov",
                "name": "SAM.gov",
                "description": "Federal contract opportunities and awards",
                "enabled": True
            },
            {
                "id": "usaspending.gov",
                "name": "USASpending.gov",
                "description": "Federal spending data including contract transactions",
                "enabled": True
            }
            # Add more sources as they are implemented
        ]
    }

@app.get("/data-insights")
def get_data_insights(
    days: int = Query(90, description="Number of days to analyze"),
    graph: KnowledgeGraph = Depends(get_knowledge_graph),
    api_key: str = Depends(get_api_key)
):
    """Get insights from collected data"""
    try:
        # This would call a more advanced analysis method in the knowledge graph
        # For now, return a placeholder
        return {
            "time_period": f"Last {days} days",
            "top_agencies": [
                {"name": "Department of Defense", "contract_count": 42, "total_value": 4200000000},
                {"name": "Department of Health and Human Services", "contract_count": 28, "total_value": 1800000000},
                {"name": "General Services Administration", "contract_count": 15, "total_value": 950000000}
            ],
            "top_contractors": [
                {"name": "TechDefense Solutions", "contract_count": 12, "total_value": 750000000},
                {"name": "Federal Systems Inc", "contract_count": 8, "total_value": 520000000},
                {"name": "CloudTech Services", "contract_count": 7, "total_value": 480000000}
            ],
            "top_technologies": [
                {"name": "cloud", "contract_count": 35},
                {"name": "cybersecurity", "contract_count": 28},
                {"name": "artificial intelligence", "contract_count": 15}
            ],
            "trends": {
                "fastest_growing_agencies": [
                    {"name": "Department of Energy", "growth_rate": 0.28},
                    {"name": "Department of Homeland Security", "growth_rate": 0.22}
                ],
                "fastest_growing_technologies": [
                    {"name": "zero trust", "growth_rate": 0.45},
                    {"name": "quantum", "growth_rate": 0.35}
                ]
            }
        }
    except Exception as e:
        logger.error(f"Data insights error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Data insights error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)