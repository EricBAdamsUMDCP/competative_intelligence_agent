# core/collectors/base_collector.py
import logging
import os
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

class BaseCollector:
    """Base class for all data collectors in the government contracting intelligence system"""
    
    def __init__(self, source_name: str, config: Dict[str, Any] = None):
        """
        Initialize the base collector
        
        Args:
            source_name: Name of the data source
            config: Configuration dictionary for the collector
        """
        self.source_name = source_name
        self.config = config or {}
        
        # Configure logging
        self.logger = logging.getLogger(f"collector.{source_name}")
        
        # Initialize tracking variables
        self.last_run = None
        self.run_duration = None
        self.items_collected = 0
        self.last_modified_date = None
        self.errors = []
    
    async def run(self, save_results: bool = True, process_results: bool = True) -> List[Dict[str, Any]]:
        """
        Run the data collection process
        
        Args:
            save_results: Whether to save results to a file
            process_results: Whether to process results before returning
            
        Returns:
            List of collected items (processed if process_results is True)
        """
        self.logger.info(f"Starting collection from {self.source_name}")
        self.last_run = datetime.now().isoformat()
        start_time = datetime.now()
        
        try:
            # Collect data
            results = await self.collect()
            
            self.items_collected = len(results) if results else 0
            self.logger.info(f"Collected {self.items_collected} items from {self.source_name}")
            
            # Process results if requested
            if process_results and results:
                results = self.process_results(results)
                self.logger.info(f"Processed {len(results)} items from {self.source_name}")
            
            # Save results if requested
            if save_results and results:
                output_path = self.save_results(results)
                self.logger.info(f"Results saved to {output_path}")
            
            # Calculate duration
            end_time = datetime.now()
            self.run_duration = (end_time - start_time).total_seconds()
            
            return results
            
        except Exception as e:
            self.errors.append(str(e))
            self.logger.error(f"Error running collector: {str(e)}")
            
            # Calculate duration even if there's an error
            end_time = datetime.now()
            self.run_duration = (end_time - start_time).total_seconds()
            
            # Re-raise the exception
            raise
    
    async def collect(self) -> List[Dict[str, Any]]:
        """
        Abstract method to collect data from the source
        
        Returns:
            List of collected items
        """
        raise NotImplementedError("Subclasses must implement collect()")
    
    def process_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process and normalize collected results
        
        Args:
            results: Raw results from the collect method
            
        Returns:
            Processed results
        """
        # Base implementation returns results unchanged
        # Subclasses should override this method to add more processing
        return results
    
    def save_results(self, results: List[Dict[str, Any]], filename: Optional[str] = None) -> str:
        """
        Save collection results to a JSON file
        
        Args:
            results: The results to save
            filename: Optional filename, if not provided a timestamp-based name will be used
            
        Returns:
            Path to the saved file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.source_name}_{timestamp}.json"
        
        # Ensure data directory exists
        data_dir = self.config.get('data_dir', 'data')
        os.makedirs(data_dir, exist_ok=True)
        file_path = os.path.join(data_dir, filename)
        
        # Handle common non-serializable types
        def json_serial(obj):
            if isinstance(obj, (datetime, timedelta)):
                return obj.isoformat()
            raise TypeError(f"Type {type(obj)} not serializable")
        
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=2, default=json_serial)
        
        return file_path
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the collector
        
        Returns:
            Dictionary with status information
        """
        return {
            "source_name": self.source_name,
            "last_run": self.last_run,
            "run_duration": self.run_duration,
            "items_collected": self.items_collected,
            "last_modified_date": self.last_modified_date,
            "errors": self.errors
        }