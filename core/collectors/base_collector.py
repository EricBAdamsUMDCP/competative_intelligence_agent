# core/collectors/base_collector.py
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Any
import logging

class BaseCollector(ABC):
    """Base class for all data collectors"""
    
    def __init__(self, source_name: str, config: Dict[str, Any]):
        self.source_name = source_name
        self.config = config
        self.last_run = None
        self.logger = logging.getLogger(f"collector.{source_name}")
    
    @abstractmethod
    async def collect(self) -> List[Dict[str, Any]]:
        """Collect data from the source"""
        pass
    
    def process_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process and normalize collected data"""
        processed = []
        for item in results:
            # Add metadata
            item['source'] = self.source_name
            item['collection_time'] = datetime.now().isoformat()
            processed.append(item)
        return processed
    
    async def run(self) -> List[Dict[str, Any]]:
        """Run the collector and process results"""
        self.logger.info(f"Starting collection from {self.source_name}")
        results = await self.collect()
        processed = self.process_results(results)
        self.last_run = datetime.now()
        self.logger.info(f"Completed collection from {self.source_name}, found {len(processed)} items")
        return processed