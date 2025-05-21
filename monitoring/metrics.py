# core/monitoring/metrics.py
import time
import logging
import os
import json
import threading
import queue
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import uuid
import concurrent.futures

class MetricsCollector:
    """Collects and processes metrics from the agent system."""
    
    def __init__(self, storage_path: str = None, flush_interval: int = 60):
        """Initialize the metrics collector.
        
        Args:
            storage_path: Path to store metrics data
            flush_interval: Time in seconds between writing metrics to disk
        """
        self.logger = logging.getLogger("metrics.collector")
        self.storage_path = storage_path or "data/metrics"
        self.flush_interval = flush_interval
        
        # Create metrics storage directory if it doesn't exist
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Queue for metrics data
        self.metrics_queue = queue.Queue()
        
        # Cache for aggregating metrics
        self.metrics_cache = {
            "agent_execution": {},
            "workflow_execution": {},
            "api_requests": {},
            "error_counts": {}
        }
        
        # Start the background processing thread
        self._start_processor()
    
    def _start_processor(self):
        """Start the background metrics processing thread."""
        self.processor_thread = threading.Thread(target=self._process_metrics, daemon=True)
        self.processor_thread.start()
        self.logger.info("Started metrics processor thread")
    
    def _process_metrics(self):
        """Process metrics in the background."""
        last_flush_time = time.time()
        
        while True:
            try:
                # Get metrics from the queue with a timeout
                try:
                    metric = self.metrics_queue.get(timeout=1.0)
                    self._process_metric(metric)
                    self.metrics_queue.task_done()
                except queue.Empty:
                    # No metrics available, check if it's time to flush
                    pass
                
                # Check if it's time to flush metrics to disk
                current_time = time.time()
                if current_time - last_flush_time >= self.flush_interval:
                    self._flush_metrics()
                    last_flush_time = current_time
            
            except Exception as e:
                self.logger.error(f"Error processing metrics: {str(e)}")
    
    def _process_metric(self, metric: Dict[str, Any]):
        """Process a single metric entry.
        
        Args:
            metric: The metric data to process
        """
        metric_type = metric.get("type")
        timestamp = metric.get("timestamp", datetime.now().isoformat())
        
        if metric_type == "agent_execution":
            agent_id = metric.get("agent_id")
            task_id = metric.get("task_id")
            duration = metric.get("duration", 0)
            status = metric.get("status", "unknown")
            
            # Add to cache
            if agent_id not in self.metrics_cache["agent_execution"]:
                self.metrics_cache["agent_execution"][agent_id] = {
                    "executions": 0,
                    "total_duration": 0,
                    "status_counts": {"success": 0, "error": 0, "other": 0},
                    "recent_tasks": []
                }
            
            agent_metrics = self.metrics_cache["agent_execution"][agent_id]
            agent_metrics["executions"] += 1
            agent_metrics["total_duration"] += duration
            
            if status == "success":
                agent_metrics["status_counts"]["success"] += 1
            elif status == "error":
                agent_metrics["status_counts"]["error"] += 1
            else:
                agent_metrics["status_counts"]["other"] += 1
            
            # Add to recent tasks
            agent_metrics["recent_tasks"].append({
                "task_id": task_id,
                "timestamp": timestamp,
                "duration": duration,
                "status": status
            })
            
            # Keep only the 100 most recent tasks
            if len(agent_metrics["recent_tasks"]) > 100:
                agent_metrics["recent_tasks"] = agent_metrics["recent_tasks"][-100:]
        
        elif metric_type == "workflow_execution":
            workflow_id = metric.get("workflow_id")
            workflow_type = metric.get("workflow_type")
            duration = metric.get("duration", 0)
            steps_count = metric.get("steps_count", 0)
            status = metric.get("status", "unknown")
            
            # Add to cache
            if workflow_type not in self.metrics_cache["workflow_execution"]:
                self.metrics_cache["workflow_execution"][workflow_type] = {
                    "executions": 0,
                    "total_duration": 0,
                    "status_counts": {"completed": 0, "failed": 0, "other": 0},
                    "recent_workflows": []
                }
            
            workflow_metrics = self.metrics_cache["workflow_execution"][workflow_type]
            workflow_metrics["executions"] += 1
            workflow_metrics["total_duration"] += duration
            
            if status == "completed":
                workflow_metrics["status_counts"]["completed"] += 1
            elif status == "failed":
                workflow_metrics["status_counts"]["failed"] += 1
            else:
                workflow_metrics["status_counts"]["other"] += 1
            
            # Add to recent workflows
            workflow_metrics["recent_workflows"].append({
                "workflow_id": workflow_id,
                "timestamp": timestamp,
                "duration": duration,
                "steps_count": steps_count,
                "status": status
            })
            
            # Keep only the 100 most recent workflows
            if len(workflow_metrics["recent_workflows"]) > 100:
                workflow_metrics["recent_workflows"] = workflow_metrics["recent_workflows"][-100:]
        
        elif metric_type == "api_request":
            endpoint = metric.get("endpoint")
            method = metric.get("method", "GET")
            duration = metric.get("duration", 0)
            status_code = metric.get("status_code", 0)
            
            # Add to cache
            endpoint_key = f"{method}:{endpoint}"
            if endpoint_key not in self.metrics_cache["api_requests"]:
                self.metrics_cache["api_requests"][endpoint_key] = {
                    "count": 0,
                    "total_duration": 0,
                    "status_counts": {},
                    "min_duration": float('inf'),
                    "max_duration": 0
                }
            
            request_metrics = self.metrics_cache["api_requests"][endpoint_key]
            request_metrics["count"] += 1
            request_metrics["total_duration"] += duration
            
            status_code_str = str(status_code)
            if status_code_str not in request_metrics["status_counts"]:
                request_metrics["status_counts"][status_code_str] = 0
            request_metrics["status_counts"][status_code_str] += 1
            
            request_metrics["min_duration"] = min(request_metrics["min_duration"], duration)
            request_metrics["max_duration"] = max(request_metrics["max_duration"], duration)
        
        elif metric_type == "error":
            error_source = metric.get("source")
            error_type = metric.get("error_type")
            
            # Add to cache
            error_key = f"{error_source}:{error_type}"
            if error_key not in self.metrics_cache["error_counts"]:
                self.metrics_cache["error_counts"][error_key] = {
                    "count": 0,
                    "first_seen": timestamp,
                    "last_seen": timestamp,
                    "examples": []
                }
            
            error_metrics = self.metrics_cache["error_counts"][error_key]
            error_metrics["count"] += 1
            error_metrics["last_seen"] = timestamp
            
            # Add error details
            error_message = metric.get("message", "")
            if error_message and len(error_metrics["examples"]) < 10:
                error_metrics["examples"].append({
                    "timestamp": timestamp,
                    "message": error_message,
                    "details": metric.get("details", {})
                })
    
    def _flush_metrics(self):
        """Flush cached metrics to disk."""
        self.logger.info("Flushing metrics to disk")
        
        try:
            # Create timestamp for this metrics snapshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create the metrics file
            metrics_file = os.path.join(self.storage_path, f"metrics_{timestamp}.json")
            
            # Add top-level metrics
            metrics_data = {
                "timestamp": datetime.now().isoformat(),
                "agent_execution": self._summarize_agent_metrics(),
                "workflow_execution": self._summarize_workflow_metrics(),
                "api_requests": self._summarize_api_metrics(),
                "error_counts": self._summarize_error_metrics(),
                "raw_cache": self.metrics_cache
            }
            
            # Write to file
            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            
            self.logger.info(f"Wrote metrics to {metrics_file}")
            
            # Clean up old metrics files
            self._cleanup_old_metrics()
        
        except Exception as e:
            self.logger.error(f"Error flushing metrics: {str(e)}")
    
    def _summarize_agent_metrics(self) -> Dict[str, Any]:
        """Summarize agent execution metrics.
        
        Returns:
            Summary of agent execution metrics
        """
        summary = {
            "total_executions": 0,
            "total_duration": 0,
            "success_rate": 0,
            "agents": []
        }
        
        # Calculate totals
        total_success = 0
        total_executions = 0
        
        for agent_id, metrics in self.metrics_cache["agent_execution"].items():
            total_executions += metrics["executions"]
            total_success += metrics["status_counts"]["success"]
            summary["total_duration"] += metrics["total_duration"]
            
            # Calculate agent-specific metrics
            success_rate = 0
            if metrics["executions"] > 0:
                success_rate = (metrics["status_counts"]["success"] / metrics["executions"]) * 100
                
            avg_duration = 0
            if metrics["executions"] > 0:
                avg_duration = metrics["total_duration"] / metrics["executions"]
            
            summary["agents"].append({
                "agent_id": agent_id,
                "executions": metrics["executions"],
                "avg_duration": avg_duration,
                "success_rate": success_rate
            })
        
        summary["total_executions"] = total_executions
        
        # Calculate overall success rate
        if total_executions > 0:
            summary["success_rate"] = (total_success / total_executions) * 100
        
        return summary
    
    def _summarize_workflow_metrics(self) -> Dict[str, Any]:
        """Summarize workflow execution metrics.
        
        Returns:
            Summary of workflow execution metrics
        """
        summary = {
            "total_executions": 0,
            "total_duration": 0,
            "completion_rate": 0,
            "workflow_types": []
        }
        
        # Calculate totals
        total_completed = 0
        total_executions = 0
        
        for workflow_type, metrics in self.metrics_cache["workflow_execution"].items():
            total_executions += metrics["executions"]
            total_completed += metrics["status_counts"]["completed"]
            summary["total_duration"] += metrics["total_duration"]
            
            # Calculate workflow-specific metrics
            completion_rate = 0
            if metrics["executions"] > 0:
                completion_rate = (metrics["status_counts"]["completed"] / metrics["executions"]) * 100
                
            avg_duration = 0
            if metrics["executions"] > 0:
                avg_duration = metrics["total_duration"] / metrics["executions"]
            
            summary["workflow_types"].append({
                "workflow_type": workflow_type,
                "executions": metrics["executions"],
                "avg_duration": avg_duration,
                "completion_rate": completion_rate
            })
        
        summary["total_executions"] = total_executions
        
        # Calculate overall completion rate
        if total_executions > 0:
            summary["completion_rate"] = (total_completed / total_executions) * 100
        
        return summary
    
    def _summarize_api_metrics(self) -> Dict[str, Any]:
        """Summarize API request metrics.
        
        Returns:
            Summary of API request metrics
        """
        summary = {
            "total_requests": 0,
            "total_duration": 0,
            "avg_duration": 0,
            "success_rate": 0,
            "endpoints": []
        }
        
        # Calculate totals
        total_requests = 0
        total_success = 0
        
        for endpoint_key, metrics in self.metrics_cache["api_requests"].items():
            total_requests += metrics["count"]
            summary["total_duration"] += metrics["total_duration"]
            
            # Count successful requests (2xx status codes)
            for status_code, count in metrics["status_counts"].items():
                if status_code.startswith("2"):
                    total_success += count
            
            # Calculate endpoint-specific metrics
            success_count = 0
            for status_code, count in metrics["status_counts"].items():
                if status_code.startswith("2"):
                    success_count += count
            
            success_rate = 0
            if metrics["count"] > 0:
                success_rate = (success_count / metrics["count"]) * 100
                
            avg_duration = 0
            if metrics["count"] > 0:
                avg_duration = metrics["total_duration"] / metrics["count"]
            
            method, endpoint = endpoint_key.split(":", 1)
            
            summary["endpoints"].append({
                "method": method,
                "endpoint": endpoint,
                "count": metrics["count"],
                "avg_duration": avg_duration,
                "min_duration": metrics["min_duration"] if metrics["min_duration"] != float('inf') else 0,
                "max_duration": metrics["max_duration"],
                "success_rate": success_rate
            })
        
        summary["total_requests"] = total_requests
        
        # Calculate overall average duration
        if total_requests > 0:
            summary["avg_duration"] = summary["total_duration"] / total_requests
        
        # Calculate overall success rate
        if total_requests > 0:
            summary["success_rate"] = (total_success / total_requests) * 100
        
        return summary
    
    def _summarize_error_metrics(self) -> Dict[str, Any]:
        """Summarize error metrics.
        
        Returns:
            Summary of error metrics
        """
        summary = {
            "total_errors": 0,
            "sources": {}
        }
        
        # Calculate totals
        for error_key, metrics in self.metrics_cache["error_counts"].items():
            summary["total_errors"] += metrics["count"]
            
            # Split error key into source and type
            source, error_type = error_key.split(":", 1)
            
            if source not in summary["sources"]:
                summary["sources"][source] = {
                    "total": 0,
                    "types": {}
                }
            
            summary["sources"][source]["total"] += metrics["count"]
            
            if error_type not in summary["sources"][source]["types"]:
                summary["sources"][source]["types"][error_type] = 0
            
            summary["sources"][source]["types"][error_type] += metrics["count"]
        
        return summary
    
    def _cleanup_old_metrics(self):
        """Clean up old metrics files."""
        # Keep only the last 24 hours of metrics (assuming 1 file per minute)
        max_files = 24 * 60
        
        try:
            # Get all metrics files
            files = [f for f in os.listdir(self.storage_path) if f.startswith("metrics_") and f.endswith(".json")]
            
            # Sort by timestamp (which is part of the filename)
            files.sort()
            
            # Remove old files
            if len(files) > max_files:
                files_to_remove = files[:-max_files]
                for file in files_to_remove:
                    os.remove(os.path.join(self.storage_path, file))
                
                self.logger.info(f"Removed {len(files_to_remove)} old metrics files")
        
        except Exception as e:
            self.logger.error(f"Error cleaning up old metrics files: {str(e)}")
    
    def record_agent_execution(self, agent_id: str, task_id: str, duration: float, status: str):
        """Record a metric for agent execution.
        
        Args:
            agent_id: ID of the agent
            task_id: ID of the task
            duration: Duration of the execution in seconds
            status: Status of the execution (success, error, etc.)
        """
        self.metrics_queue.put({
            "type": "agent_execution",
            "timestamp": datetime.now().isoformat(),
            "agent_id": agent_id,
            "task_id": task_id,
            "duration": duration,
            "status": status
        })
    
    def record_workflow_execution(self, workflow_id: str, workflow_type: str, duration: float, 
                                 steps_count: int, status: str):
        """Record a metric for workflow execution.
        
        Args:
            workflow_id: ID of the workflow
            workflow_type: Type of workflow
            duration: Duration of the execution in seconds
            steps_count: Number of steps in the workflow
            status: Status of the execution (completed, failed, etc.)
        """
        self.metrics_queue.put({
            "type": "workflow_execution",
            "timestamp": datetime.now().isoformat(),
            "workflow_id": workflow_id,
            "workflow_type": workflow_type,
            "duration": duration,
            "steps_count": steps_count,
            "status": status
        })
    
    def record_api_request(self, endpoint: str, method: str, duration: float, status_code: int):
        """Record a metric for an API request.
        
        Args:
            endpoint: API endpoint
            method: HTTP method
            duration: Duration of the request in seconds
            status_code: HTTP status code
        """
        self.metrics_queue.put({
            "type": "api_request",
            "timestamp": datetime.now().isoformat(),
            "endpoint": endpoint,
            "method": method,
            "duration": duration,
            "status_code": status_code
        })
    
    def record_error(self, source: str, error_type: str, message: str, details: Dict[str, Any] = None):
        """Record a metric for an error.
        
        Args:
            source: Source of the error (agent, workflow, api, etc.)
            error_type: Type of error
            message: Error message
            details: Additional error details
        """
        self.metrics_queue.put({
            "type": "error",
            "timestamp": datetime.now().isoformat(),
            "source": source,
            "error_type": error_type,
            "message": message,
            "details": details or {}
        })
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get the current metrics.
        
        Returns:
            Current metrics data
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "agent_execution": self._summarize_agent_metrics(),
            "workflow_execution": self._summarize_workflow_metrics(),
            "api_requests": self._summarize_api_metrics(),
            "error_counts": self._summarize_error_metrics()
        }


# Create a global metrics collector instance
_metrics_collector = None

def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance.
    
    Returns:
        Global metrics collector instance
    """
    global _metrics_collector
    if _metrics_collector is None:
        storage_path = os.environ.get("METRICS_STORAGE_PATH", "data/metrics")
        flush_interval = int(os.environ.get("METRICS_FLUSH_INTERVAL", "60"))
        _metrics_collector = MetricsCollector(storage_path=storage_path, flush_interval=flush_interval)
    
    return _metrics_collector


# Decorator for measuring agent execution time
def measure_agent_execution(func):
    """Decorator to measure agent execution time and record metrics."""
    async def wrapper(self, *args, **kwargs):
        task_id = kwargs.get("task_id") or str(uuid.uuid4())
        
        start_time = time.time()
        try:
            result = await func(self, *args, **kwargs)
            duration = time.time() - start_time
            
            # Record success metric
            metrics = get_metrics_collector()
            metrics.record_agent_execution(
                agent_id=self.agent_id,
                task_id=task_id,
                duration=duration,
                status="success"
            )
            
            return result
        except Exception as e:
            duration = time.time() - start_time
            
            # Record error metric
            metrics = get_metrics_collector()
            metrics.record_agent_execution(
                agent_id=self.agent_id,
                task_id=task_id,
                duration=duration,
                status="error"
            )
            
            # Record detailed error
            metrics.record_error(
                source=f"agent:{self.agent_id}",
                error_type=type(e).__name__,
                message=str(e),
                details={
                    "task_id": task_id,
                    "args": str(args),
                    "kwargs": {k: v for k, v in kwargs.items() if k != "password"}
                }
            )
            
            raise
    
    return wrapper


# Decorator for measuring workflow execution time
def measure_workflow_execution(func):
    """Decorator to measure workflow execution time and record metrics."""
    async def wrapper(self, *args, **kwargs):
        workflow_id = kwargs.get("workflow_id") or str(uuid.uuid4())
        workflow_def = kwargs.get("workflow_def", {})
        workflow_type = workflow_def.get("name", "unknown")
        
        start_time = time.time()
        try:
            result = await func(self, *args, **kwargs)
            duration = time.time() - start_time
            
            # Record success metric
            metrics = get_metrics_collector()
            metrics.record_workflow_execution(
                workflow_id=workflow_id,
                workflow_type=workflow_type,
                duration=duration,
                steps_count=len(workflow_def.get("steps", [])),
                status=result.get("status", "completed")
            )
            
            return result
        except Exception as e:
            duration = time.time() - start_time
            
            # Record error metric
            metrics = get_metrics_collector()
            metrics.record_workflow_execution(
                workflow_id=workflow_id,
                workflow_type=workflow_type,
                duration=duration,
                steps_count=len(workflow_def.get("steps", [])),
                status="failed"
            )
            
            # Record detailed error
            metrics.record_error(
                source=f"workflow:{workflow_type}",
                error_type=type(e).__name__,
                message=str(e),
                details={
                    "workflow_id": workflow_id,
                    "args": str(args),
                    "kwargs": {k: v for k, v in kwargs.items() if k != "password"}
                }
            )
            
            raise
    
    return wrapper


# Decorator for measuring API request time
def measure_api_request(func):
    """Decorator to measure API request time and record metrics."""
    async def wrapper(request, *args, **kwargs):
        start_time = time.time()
        try:
            response = await func(request, *args, **kwargs)
            duration = time.time() - start_time
            
            # Record metric
            metrics = get_metrics_collector()
            metrics.record_api_request(
                endpoint=str(request.url.path),
                method=request.method,
                duration=duration,
                status_code=response.status_code
            )
            
            return response
        except Exception as e:
            duration = time.time() - start_time
            
            # Record error metric
            metrics = get_metrics_collector()
            metrics.record_api_request(
                endpoint=str(request.url.path),
                method=request.method,
                duration=duration,
                status_code=500
            )
            
            # Record detailed error
            metrics.record_error(
                source="api",
                error_type=type(e).__name__,
                message=str(e),
                details={
                    "endpoint": str(request.url.path),
                    "method": request.method,
                    "query_params": str(request.query_params),
                    "client_host": request.client.host if request.client else "unknown"
                }
            )
            
            raise
    
    return wrapper