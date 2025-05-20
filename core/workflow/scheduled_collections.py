"""
Scheduled data collection workflow for government contracting intelligence system.
This script orchestrates the regular collection, processing, and storage of contract data.
"""

import asyncio
import logging
import schedule
import time
import sys
import os
from datetime import datetime, timedelta
import json
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.agents.agent_framework import (
    AgentOrchestrator, 
    DataCollectionAgent, 
    EntityExtractionAgent,
    KnowledgeGraphAgent
)
from core.collectors.sam_gov import SamGovCollector
from core.collectors.usaspending_gov import USASpendingCollector
from core.collectors.industry_news import IndustryNewsCollector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ci_collection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("scheduled_collection")

async def run_collection_pipeline(days_back=7, use_mock=False, save_results=True):
    """Run the complete data collection pipeline"""
    logger.info(f"Starting data collection pipeline (days_back={days_back}, use_mock={use_mock})")
    
    # Initialize orchestrator
    orchestrator = AgentOrchestrator()
    
    # Configure and register data collection agents
    sam_gov_config = {
        'use_mock': use_mock,
        'published_since': (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d'),
        'limit': 25 if use_mock else 100
    }
    
    usaspending_config = {
        'use_mock': use_mock,
        'time_period_start': (datetime.now() - timedelta(days=days_back*4)).strftime('%Y-%m-%d'),
        'time_period_end': datetime.now().strftime('%Y-%m-%d'),
        'keywords': ['cybersecurity', 'cloud', 'artificial intelligence', 'data analytics', 'machine learning'],
        'naics_codes': ['541512', '518210', '541511', '541330', '541715'],
        'limit': 25 if use_mock else 100
    }

    news_config = {
        'use_mock': use_mock,
        'days_back': days_back
    }
    
    # Create and register data collection agents
    sam_agent = DataCollectionAgent("sam_gov_collector", SamGovCollector, sam_gov_config)
    usaspending_agent = DataCollectionAgent("usaspending_collector", USASpendingCollector, usaspending_config)
    news_agent = DataCollectionAgent("industry_news_collector", IndustryNewsCollector, news_config)
    
    orchestrator.register_agent(sam_agent)
    orchestrator.register_agent(usaspending_agent)
    orchestrator.register_agent(news_agent)
    
    # Entity extraction agent
    entity_agent = EntityExtractionAgent("entity_extractor")
    entity_agent.add_dependency("sam_gov_collector")
    entity_agent.add_dependency("usaspending_collector")
    entity_agent.add_dependency("industry_news_collector")
    
    orchestrator.register_agent(entity_agent)
    
    # Knowledge graph agent
    graph_agent = KnowledgeGraphAgent("knowledge_graph")
    graph_agent.add_dependency("entity_extractor")
    
    orchestrator.register_agent(graph_agent)
    
    # Run the pipeline
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    context = {
        "run_id": run_id,
        "scheduled": True,
        "days_back": days_back,
        "use_mock": use_mock
    }
    
    results = await orchestrator.run_pipeline(context)
    
    # Log results
    if results["status"] == "completed":
        logger.info(f"Pipeline completed successfully (run_id={run_id})")
        for agent_name, agent_result in results["results"].items():
            if agent_result["status"] == "success":
                processed = agent_result.get('processed_count', 0)
                if processed:
                    logger.info(f"Agent {agent_name}: {processed} items processed")
                else:
                    logger.info(f"Agent {agent_name}: completed successfully")
            else:
                logger.warning(f"Agent {agent_name}: {agent_result.get('status')}")
    else:
        logger.error(f"Pipeline failed: {results.get('message', 'Unknown error')}")
    
    # Save results to file if requested
    if save_results:
        output_dir = os.path.join(project_root, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, f"pipeline_results_{run_id}.json")
        with open(output_file, 'w') as f:
            # Convert datetimes to strings for JSON serialization
            json_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    json_results[key] = {}
                    for k, v in value.items():
                        if isinstance(v, (datetime, timedelta)):
                            json_results[key][k] = str(v)
                        else:
                            json_results[key][k] = v
                else:
                    json_results[key] = value
            
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
    
    return results

def run_scheduled_job(days_back=1, use_mock=False):
    """Run the async collection job synchronously for the scheduler"""
    logger.info(f"Running scheduled collection job (days_back={days_back})")
    try:
        asyncio.run(run_collection_pipeline(days_back=days_back, use_mock=use_mock))
        logger.info("Scheduled job completed successfully")
    except Exception as e:
        logger.error(f"Error in scheduled job: {str(e)}")

def main():
    """Set up scheduled collection"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run scheduled collection for government contracting intelligence.')
    parser.add_argument('--days-back', type=int, default=7, help='Number of days to look back for data')
    parser.add_argument('--use-mock', action='store_true', help='Use mock data instead of real API calls')
    parser.add_argument('--run-now', action='store_true', help='Run collection immediately')
    parser.add_argument('--schedule-time', type=str, default="02:00", help='Time to run daily collection (HH:MM)')
    parser.add_argument('--no-schedule', action='store_true', help='Do not set up scheduled runs')
    
    args = parser.parse_args()
    
    logger.info(f"Starting scheduled collection service with args: {args}")
    
    # Run immediately if requested
    if args.run_now:
        logger.info("Running initial collection...")
        run_scheduled_job(days_back=args.days_back, use_mock=args.use_mock)
    
    # Set up scheduled run
    if not args.no_schedule:
        # Schedule daily collection
        schedule.every().day.at(args.schedule_time).do(
            run_scheduled_job, days_back=1, use_mock=args.use_mock
        )
        
        # Also schedule a weekly more comprehensive collection
        schedule.every().monday.at("03:00").do(
            run_scheduled_job, days_back=7, use_mock=args.use_mock
        )
        
        logger.info(f"Scheduled daily collection at {args.schedule_time}")
        logger.info("Scheduled weekly collection on Mondays at 03:00")
        
        # Keep the scheduler running
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    else:
        logger.info("No scheduled collection set up (--no-schedule was specified)")

if __name__ == "__main__":
    main()