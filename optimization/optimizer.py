# core/optimization/optimizer.py
import time
import logging
import os
import json
import asyncio
import concurrent.futures
from functools import wraps
from typing import Dict, List, Any, Optional, Union, Callable, Awaitable
from datetime import datetime, timedelta
import threading
import queue
from collections import deque
import traceback

class AgentOptimizer:
    """Optimizes agent performance through caching, batching, and throttling."""
    
    def __init__(self):
        """Initialize the agent optimizer."""
        self.logger = logging.getLogger("agent.optimizer")
        
        # Result cache for avoiding repeated work
        self.result_cache = {}
        
        # Cache TTL for different operation types (in seconds)
        self.cache_ttl = {
            "default": 300,  # 5 minutes
            "search": 60,  # 1 minute
            "collect": 3600,  # 1 hour
            "analyze": 1800,  # 30 minutes
            "query": 120  # 2 minutes
        }
        
        # Task queue for batching
        self.task_queues = {}
        
        # Task queue locks
        self.queue_locks = {}
        
        # Batch processing intervals (in seconds)
        self.batch_intervals = {
            "default": 1.0,
            "collect": 5.0,
            "entity_extraction": 2.0,
            "knowledge_graph": 1.0
        }
        
        # Rate limiting configuration
        self.rate_limits = {
            "default": (10, 1),  # 10 calls per second
            "sam_gov_api": (5, 60),  # 5 calls per minute
            "elasticsearch": (20, 1),  # 20 calls per second
            "neo4j": (100, 1)  # 100 calls per second
        }
        
        # Rate limit tracking
        self.call_history = {}
        
        # Start background processing threads for each queue
        self._start_batch_processors()
    
    def _start_batch_processors(self):
        """Start background threads for processing task batches."""
        for queue_name, interval in self.batch_intervals.items():
            self.task_queues[queue_name] = queue.Queue()
            self.queue_locks[queue_name] = threading.Lock()
            
            processor_thread = threading.Thread(
                target=self._process_queue,
                args=(queue_name, interval),
                daemon=True
            )
            processor_thread.start()
            
            self.logger.info(f"Started batch processor for {queue_name} with interval {interval}s")
    
    def _process_queue(self, queue_name: str, interval: float):
        """Process tasks from a specific queue at regular intervals.
        
        Args:
            queue_name: Name of the queue to process
            interval: Interval in seconds between batch processing
        """
        while True:
            time.sleep(interval)
            
            # Skip if queue is empty
            if self.task_queues[queue_name].empty():
                continue
            
            with self.queue_locks[queue_name]:
                # Get all tasks from the queue
                tasks = []
                try:
                    while not self.task_queues[queue_name].empty():
                        tasks.append(self.task_queues[queue_name].get_nowait())
                except queue.Empty:
                    pass
                
                if not tasks:
                    continue
                
                # Process the batch
                self.logger.info(f"Processing batch of {len(tasks)} tasks for {queue_name}")
                
                # Group tasks by function
                grouped_tasks = {}
                for task in tasks:
                    func = task["func"]
                    if func not in grouped_tasks:
                        grouped_tasks[func] = []
                    grouped_tasks[func].append(task)
                
                # Process each group
                for func, func_tasks in grouped_tasks.items():
                    try:
                        # Call the batch processing function
                        batch_args = [task["args"] for task in func_tasks]
                        batch_kwargs = [task["kwargs"] for task in func_tasks]
                        
                        # Check if the function has a batch processing method
                        if hasattr(func, "_batch_process"):
                            results = func._batch_process(batch_args, batch_kwargs)
                        else:
                            # Process individually
                            results = []
                            for args, kwargs in zip(batch_args, batch_kwargs):
                                result = func(*args, **kwargs)
                                results.append(result)
                        
                        # Set results for each task
                        for task, result in zip(func_tasks, results):
                            task["future"].set_result(result)
                            self.task_queues[queue_name].task_done()
                    
                    except Exception as e:
                        self.logger.error(f"Error processing batch for {func.__name__}: {str(e)}")
                        
                        # Set exception for each task
                        for task in func_tasks:
                            task["future"].set_exception(e)
                            self.task_queues[queue_name].task_done()
    
    def cache_result(self, func=None, ttl=None, key_fn=None, 
                    operation_type="default", is_async=False):
        """Decorator to cache function results.
        
        Args:
            func: The function to cache
            ttl: Time-to-live for cache entries (in seconds)
            key_fn: Function to generate cache keys from args/kwargs
            operation_type: Type of operation for TTL lookup
            is_async: Whether the function is async
            
        Returns:
            Decorated function
        """
        def decorator(f):
            @wraps(f)
            def sync_wrapper(*args, **kwargs):
                # Generate cache key
                if key_fn:
                    cache_key = key_fn(*args, **kwargs)
                else:
                    # Default key is based on function name, args, and kwargs
                    kwarg_str = json.dumps(
                        {k: v for k, v in kwargs.items() if k != "password"},
                        sort_keys=True,
                        default=str
                    )
                    cache_key = f"{f.__module__}.{f.__name__}:{str(args)}:{kwarg_str}"
                
                # Check if result is in cache and not expired
                if cache_key in self.result_cache:
                    entry = self.result_cache[cache_key]
                    if entry["expiry"] > time.time():
                        self.logger.debug(f"Cache hit for {f.__name__}")
                        return entry["result"]
                
                # Execute function
                result = f(*args, **kwargs)
                
                # Store in cache
                cache_ttl = ttl or self.cache_ttl.get(operation_type, self.cache_ttl["default"])
                self.result_cache[cache_key] = {
                    "result": result,
                    "expiry": time.time() + cache_ttl
                }
                
                return result
            
            @wraps(f)
            async def async_wrapper(*args, **kwargs):
                # Generate cache key
                if key_fn:
                    cache_key = key_fn(*args, **kwargs)
                else:
                    # Default key is based on function name, args, and kwargs
                    kwarg_str = json.dumps(
                        {k: v for k, v in kwargs.items() if k != "password"},
                        sort_keys=True,
                        default=str
                    )
                    cache_key = f"{f.__module__}.{f.__name__}:{str(args)}:{kwarg_str}"
                
                # Check if result is in cache and not expired
                if cache_key in self.result_cache:
                    entry = self.result_cache[cache_key]
                    if entry["expiry"] > time.time():
                        self.logger.debug(f"Cache hit for {f.__name__}")
                        return entry["result"]
                
                # Execute function
                result = await f(*args, **kwargs)
                
                # Store in cache
                cache_ttl = ttl or self.cache_ttl.get(operation_type, self.cache_ttl["default"])
                self.result_cache[cache_key] = {
                    "result": result,
                    "expiry": time.time() + cache_ttl
                }
                
                return result
            
            if is_async:
                return async_wrapper
            else:
                return sync_wrapper
        
        if func:
            return decorator(func)
        return decorator
    
    def batch_operations(self, queue_name="default", is_async=False):
        """Decorator to batch operations.
        
        Args:
            queue_name: Name of the queue to use for batching
            is_async: Whether the function is async
            
        Returns:
            Decorated function
        """
        def decorator(func):
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Create a future for this task
                loop = asyncio.get_event_loop()
                future = loop.create_future()
                
                # Add to the task queue
                self.task_queues[queue_name].put({
                    "func": func,
                    "args": args,
                    "kwargs": kwargs,
                    "future": future
                })
                
                # Wait for the result
                return asyncio.run(future)
            
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Create a future for this task
                loop = asyncio.get_event_loop()
                future = loop.create_future()
                
                # Add to the task queue
                self.task_queues[queue_name].put({
                    "func": func,
                    "args": args,
                    "kwargs": kwargs,
                    "future": future
                })
                
                # Wait for the result
                return await future
            
            if is_async:
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    def rate_limit(self, limit_type="default", retry=True, is_async=False):
        """Decorator to apply rate limiting.
        
        Args:
            limit_type: Type of rate limit to apply
            retry: Whether to retry if rate limited
            is_async: Whether the function is async
            
        Returns:
            Decorated function
        """
        def decorator(func):
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Get rate limit configuration
                rate_limit = self.rate_limits.get(limit_type, self.rate_limits["default"])
                max_calls, period = rate_limit
                
                # Check if we're within the rate limit
                current_time = time.time()
                
                # Initialize call history if needed
                if limit_type not in self.call_history:
                    self.call_history[limit_type] = deque()
                
                # Remove expired entries
                while (self.call_history[limit_type] and 
                      self.call_history[limit_type][0] < current_time - period):
                    self.call_history[limit_type].popleft()
                
                # Check if we've reached the limit
                if len(self.call_history[limit_type]) >= max_calls:
                    if retry:
                        # Wait until we can make another call
                        sleep_time = period - (current_time - self.call_history[limit_type][0])
                        self.logger.info(f"Rate limit reached for {limit_type}, waiting {sleep_time:.2f}s")
                        time.sleep(max(0, sleep_time))
                    else:
                        raise Exception(f"Rate limit exceeded for {limit_type}")
                
                # Record this call
                self.call_history[limit_type].append(current_time)
                
                # Execute function
                return func(*args, **kwargs)
            
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Get rate limit configuration
                rate_limit = self.rate_limits.get(limit_type, self.rate_limits["default"])
                max_calls, period = rate_limit
                
                # Check if we're within the rate limit
                current_time = time.time()
                
                # Initialize call history if needed
                if limit_type not in self.call_history:
                    self.call_history[limit_type] = deque()
                
                # Remove expired entries
                while (self.call_history[limit_type] and 
                      self.call_history[limit_type][0] < current_time - period):
                    self.call_history[limit_type].popleft()
                
                # Check if we've reached the limit
                if len(self.call_history[limit_type]) >= max_calls:
                    if retry:
                        # Wait until we can make another call
                        sleep_time = period - (current_time - self.call_history[limit_type][0])
                        self.logger.info(f"Rate limit reached for {limit_type}, waiting {sleep_time:.2f}s")
                        await asyncio.sleep(max(0, sleep_time))
                    else:
                        raise Exception(f"Rate limit exceeded for {limit_type}")
                
                # Record this call
                self.call_history[limit_type].append(current_time)
                
                # Execute function
                return await func(*args, **kwargs)
            
            if is_async:
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    def timeout(self, seconds=30, is_async=False):
        """Decorator to apply a timeout to a function.
        
        Args:
            seconds: Timeout in seconds
            is_async: Whether the function is async
            
        Returns:
            Decorated function
        """
        def decorator(func):
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Create a future for this task
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(func, *args, **kwargs)
                    try:
                        return future.result(timeout=seconds)
                    except concurrent.futures.TimeoutError:
                        future.cancel()
                        raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")
            
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    # Execute with timeout
                    return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
                except asyncio.TimeoutError:
                    raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")
            
            if is_async:
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    def retry(self, max_retries=3, retry_delay=1.0, 
             exceptions=(Exception,), backoff_factor=2.0, is_async=False):
        """Decorator to retry a function on failure.
        
        Args:
            max_retries: Maximum number of retries
            retry_delay: Initial delay between retries (in seconds)
            exceptions: Exceptions to retry on
            backoff_factor: Factor to multiply delay by after each retry
            is_async: Whether the function is async
            
        Returns:
            Decorated function
        """
        def decorator(func):
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                last_exception = None
                delay = retry_delay
                
                for attempt in range(max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        if attempt < max_retries:
                            self.logger.warning(
                                f"Attempt {attempt + 1}/{max_retries + 1} for {func.__name__} failed: {str(e)}, "
                                f"retrying in {delay:.2f}s"
                            )
                            time.sleep(delay)
                            delay *= backoff_factor
                        else:
                            self.logger.error(
                                f"All {max_retries + 1} attempts for {func.__name__} failed, "
                                f"last error: {str(e)}"
                            )
                
                raise last_exception
            
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                last_exception = None
                delay = retry_delay
                
                for attempt in range(max_retries + 1):
                    try:
                        return await func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        if attempt < max_retries:
                            self.logger.warning(
                                f"Attempt {attempt + 1}/{max_retries + 1} for {func.__name__} failed: {str(e)}, "
                                f"retrying in {delay:.2f}s"
                            )
                            await asyncio.sleep(delay)
                            delay *= backoff_factor
                        else:
                            self.logger.error(
                                f"All {max_retries + 1} attempts for {func.__name__} failed, "
                                f"last error: {str(e)}"
                            )
                
                raise last_exception
            
            if is_async:
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    def circuit_breaker(self, failure_threshold=5, recovery_timeout=60, 
                       half_open_timeout=30, is_async=False):
        """Decorator to implement the circuit breaker pattern.
        
        Args:
            failure_threshold: Number of failures before opening the circuit
            recovery_timeout: Time to wait before closing the circuit (in seconds)
            half_open_timeout: Time to wait before half-opening the circuit (in seconds)
            is_async: Whether the function is async
            
        Returns:
            Decorated function
        """
        # Circuit state
        class CircuitState:
            CLOSED = "closed"
            OPEN = "open"
            HALF_OPEN = "half_open"
        
        class CircuitBreaker:
            def __init__(self):
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.last_failure_time = 0
                self.half_open_time = 0
        
        # Create a circuit breaker instance for this decorator
        circuit = CircuitBreaker()
        
        def decorator(func):
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                current_time = time.time()
                
                # Check circuit state
                if circuit.state == CircuitState.OPEN:
                    # Check if recovery timeout has elapsed
                    if current_time - circuit.last_failure_time >= half_open_timeout:
                        # Half-open the circuit
                        circuit.state = CircuitState.HALF_OPEN
                        circuit.half_open_time = current_time
                        self.logger.info(f"Circuit half-opened for {func.__name__}")
                    else:
                        raise Exception(f"Circuit is open for {func.__name__}, request rejected")
                
                # Try to execute the function
                try:
                    result = func(*args, **kwargs)
                    
                    # If we're in half-open state and the call succeeded, close the circuit
                    if circuit.state == CircuitState.HALF_OPEN:
                        circuit.state = CircuitState.CLOSED
                        circuit.failure_count = 0
                        self.logger.info(f"Circuit closed for {func.__name__}")
                    
                    return result
                
                except Exception as e:
                    # Record the failure
                    circuit.failure_count += 1
                    circuit.last_failure_time = current_time
                    
                    # Check if we need to open the circuit
                    if circuit.state == CircuitState.CLOSED and circuit.failure_count >= failure_threshold:
                        circuit.state = CircuitState.OPEN
                        self.logger.warning(f"Circuit opened for {func.__name__} after {failure_threshold} failures")
                    elif circuit.state == CircuitState.HALF_OPEN:
                        circuit.state = CircuitState.OPEN
                        self.logger.warning(f"Circuit re-opened for {func.__name__} after failure in half-open state")
                    
                    raise
            
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                current_time = time.time()
                
                # Check circuit state
                if circuit.state == CircuitState.OPEN:
                    # Check if recovery timeout has elapsed
                    if current_time - circuit.last_failure_time >= half_open_timeout:
                        # Half-open the circuit
                        circuit.state = CircuitState.HALF_OPEN
                        circuit.half_open_time = current_time
                        self.logger.info(f"Circuit half-opened for {func.__name__}")
                    else:
                        raise Exception(f"Circuit is open for {func.__name__}, request rejected")
                
                # Try to execute the function
                try:
                    result = await func(*args, **kwargs)
                    
                    # If we're in half-open state and the call succeeded, close the circuit
                    if circuit.state == CircuitState.HALF_OPEN:
                        circuit.state = CircuitState.CLOSED
                        circuit.failure_count = 0
                        self.logger.info(f"Circuit closed for {func.__name__}")
                    
                    return result
                
                except Exception as e:
                    # Record the failure
                    circuit.failure_count += 1
                    circuit.last_failure_time = current_time
                    
                    # Check if we need to open the circuit
                    if circuit.state == CircuitState.CLOSED and circuit.failure_count >= failure_threshold:
                        circuit.state = CircuitState.OPEN
                        self.logger.warning(f"Circuit opened for {func.__name__} after {failure_threshold} failures")
                    elif circuit.state == CircuitState.HALF_OPEN:
                        circuit.state = CircuitState.OPEN
                        self.logger.warning(f"Circuit re-opened for {func.__name__} after failure in half-open state")
                    
                    raise
            
            if is_async:
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    def clear_cache(self, operation_type=None):
        """Clear the result cache.
        
        Args:
            operation_type: Optional operation type to clear cache for
        """
        if operation_type:
            # Clear only entries with matching TTL
            ttl = self.cache_ttl.get(operation_type, self.cache_ttl["default"])
            to_remove = []
            
            for key, entry in self.result_cache.items():
                cache_ttl = entry["expiry"] - (entry.get("creation_time", 0) or time.time())
                if abs(cache_ttl - ttl) < 1:  # Allow 1 second difference
                    to_remove.append(key)
            
            for key in to_remove:
                del self.result_cache[key]
            
            self.logger.info(f"Cleared cache for operation type {operation_type} ({len(to_remove)} entries)")
        else:
            # Clear all cache
            self.result_cache = {}
            self.logger.info("Cleared entire result cache")
    
    def update_rate_limit(self, limit_type: str, max_calls: int, period: int):
        """Update a rate limit configuration.
        
        Args:
            limit_type: Type of rate limit to update
            max_calls: Maximum number of calls allowed
            period: Time period in seconds
        """
        self.rate_limits[limit_type] = (max_calls, period)
        self.logger.info(f"Updated rate limit for {limit_type}: {max_calls} calls per {period}s")


# Create a global optimizer instance
_optimizer = None

def get_optimizer() -> AgentOptimizer:
    """Get the global optimizer instance.
    
    Returns:
        Global optimizer instance
    """
    global _optimizer
    if _optimizer is None:
        _optimizer = AgentOptimizer()
    
    return _optimizer