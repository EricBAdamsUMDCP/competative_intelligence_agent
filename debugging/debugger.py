# core/debugging/debugger.py
import time
import logging
import os
import json
import asyncio
import inspect
import traceback
import cProfile
import pstats
import io
from typing import Dict, List, Any, Optional, Union, Callable, Awaitable
from datetime import datetime, timedelta
import threading
import uuid
from functools import wraps
import tempfile
import sys

class AgentDebugger:
    """Debug and profile agents and workflows."""
    
    def __init__(self, debug_dir: str = None, enabled: bool = True, 
                profiling_enabled: bool = False):
        """Initialize the agent debugger.
        
        Args:
            debug_dir: Directory to store debug logs and profiles
            enabled: Whether debugging is enabled
            profiling_enabled: Whether profiling is enabled
        """
        self.logger = logging.getLogger("agent.debugger")
        self.debug_dir = debug_dir or "data/debug"
        self.enabled = enabled
        self.profiling_enabled = profiling_enabled
        
        # Create debug directory if it doesn't exist
        if self.enabled:
            os.makedirs(self.debug_dir, exist_ok=True)
            os.makedirs(os.path.join(self.debug_dir, "traces"), exist_ok=True)
            os.makedirs(os.path.join(self.debug_dir, "profiles"), exist_ok=True)
        
        # Trace storage
        self.traces = {}
        
        # Current trace stack by thread
        self.trace_stack = {}
        
        # Profiling data
        self.profiles = {}
    
    def start_trace(self, trace_id: str = None, trace_name: str = None, 
                   parent_id: str = None) -> str:
        """Start a new execution trace.
        
        Args:
            trace_id: Optional ID for the trace
            trace_name: Name for the trace
            parent_id: Optional parent trace ID
            
        Returns:
            ID of the new trace
        """
        if not self.enabled:
            return "disabled"
        
        trace_id = trace_id or str(uuid.uuid4())
        trace_name = trace_name or f"Trace {trace_id}"
        
        # Create trace
        trace = {
            "id": trace_id,
            "name": trace_name,
            "parent_id": parent_id,
            "start_time": time.time(),
            "end_time": None,
            "duration": None,
            "thread_id": threading.get_ident(),
            "events": [],
            "children": [],
            "status": "running",
            "result": None,
            "exception": None
        }
        
        self.traces[trace_id] = trace
        
        # Add to parent's children if parent exists
        if parent_id and parent_id in self.traces:
            self.traces[parent_id]["children"].append(trace_id)
        
        # Add to the trace stack for this thread
        thread_id = threading.get_ident()
        if thread_id not in self.trace_stack:
            self.trace_stack[thread_id] = []
        
        self.trace_stack[thread_id].append(trace_id)
        
        return trace_id
    
    def add_event(self, trace_id: str, event_type: str, event_data: Dict[str, Any] = None):
        """Add an event to a trace.
        
        Args:
            trace_id: ID of the trace
            event_type: Type of event
            event_data: Additional event data
        """
        if not self.enabled or trace_id == "disabled" or trace_id not in self.traces:
            return
        
        event = {
            "id": str(uuid.uuid4()),
            "type": event_type,
            "time": time.time(),
            "data": event_data or {}
        }
        
        self.traces[trace_id]["events"].append(event)
    
    def end_trace(self, trace_id: str, status: str = "completed", 
                 result: Any = None, exception: Exception = None):
        """End a trace.
        
        Args:
            trace_id: ID of the trace
            status: Status of the trace
            result: Result of the trace
            exception: Exception that occurred during the trace
        """
        if not self.enabled or trace_id == "disabled" or trace_id not in self.traces:
            return
        
        # Update trace
        trace = self.traces[trace_id]
        trace["end_time"] = time.time()
        trace["duration"] = trace["end_time"] - trace["start_time"]
        trace["status"] = status
        trace["result"] = self._sanitize_result(result)
        
        if exception:
            trace["exception"] = {
                "type": type(exception).__name__,
                "message": str(exception),
                "traceback": traceback.format_exc()
            }
        
        # Remove from trace stack
        thread_id = threading.get_ident()
        if thread_id in self.trace_stack and self.trace_stack[thread_id]:
            if self.trace_stack[thread_id][-1] == trace_id:
                self.trace_stack[thread_id].pop()
            else:
                # This is unexpected, but try to find and remove it
                try:
                    self.trace_stack[thread_id].remove(trace_id)
                except ValueError:
                    pass
        
        # Save the trace to disk if it's a top-level trace
        if not trace["parent_id"]:
            self._save_trace(trace_id)
    
    def _sanitize_result(self, result: Any) -> Any:
        """Sanitize a result for storage in a trace.
        
        Args:
            result: Result to sanitize
            
        Returns:
            Sanitized result
        """
        try:
            # Try to convert to JSON to ensure it's serializable
            json.dumps(result)
            return result
        except (TypeError, ValueError, OverflowError):
            # If not serializable, convert to string
            try:
                return str(result)
            except:
                return "Unserializable result"
    
    def _save_trace(self, trace_id: str):
        """Save a trace to disk.
        
        Args:
            trace_id: ID of the trace
        """
        if trace_id not in self.traces:
            return
        
        trace = self.traces[trace_id]
        
        # Create a complete trace tree for saving
        trace_tree = self._build_trace_tree(trace_id)
        
        # Generate filename based on trace name and time
        timestamp = datetime.fromtimestamp(trace["start_time"]).strftime("%Y%m%d_%H%M%S")
        name_part = trace["name"].replace(" ", "_").lower()
        filename = f"{timestamp}_{name_part}_{trace_id}.json"
        
        # Save to file
        try:
            with open(os.path.join(self.debug_dir, "traces", filename), 'w') as f:
                json.dump(trace_tree, f, indent=2)
            
            self.logger.debug(f"Saved trace {trace_id} to {filename}")
        except Exception as e:
            self.logger.error(f"Error saving trace {trace_id}: {str(e)}")
    
    def _build_trace_tree(self, trace_id: str) -> Dict[str, Any]:
        """Build a complete trace tree from a trace ID.
        
        Args:
            trace_id: ID of the trace
            
        Returns:
            Complete trace tree
        """
        if trace_id not in self.traces:
            return {}
        
        trace = self.traces[trace_id].copy()
        
        # Replace child IDs with actual child traces
        child_traces = []
        for child_id in trace["children"]:
            child_trace = self._build_trace_tree(child_id)
            if child_trace:
                child_traces.append(child_trace)
        
        trace["children"] = child_traces
        
        return trace
    
    def get_current_trace_id(self) -> Optional[str]:
        """Get the ID of the current trace for this thread.
        
        Returns:
            Current trace ID or None
        """
        if not self.enabled:
            return None
        
        thread_id = threading.get_ident()
        
        if thread_id in self.trace_stack and self.trace_stack[thread_id]:
            return self.trace_stack[thread_id][-1]
        
        return None
    
    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get a trace by ID.
        
        Args:
            trace_id: ID of the trace
            
        Returns:
            Trace or None
        """
        if not self.enabled or trace_id not in self.traces:
            return None
        
        return self.traces[trace_id].copy()
    
    def start_profiling(self, profile_id: str = None, profile_name: str = None) -> str:
        """Start profiling.
        
        Args:
            profile_id: Optional ID for the profile
            profile_name: Name for the profile
            
        Returns:
            ID of the new profile
        """
        if not self.enabled or not self.profiling_enabled:
            return "disabled"
        
        profile_id = profile_id or str(uuid.uuid4())
        profile_name = profile_name or f"Profile {profile_id}"
        
        # Create profile
        profile = {
            "id": profile_id,
            "name": profile_name,
            "start_time": time.time(),
            "end_time": None,
            "duration": None,
            "thread_id": threading.get_ident(),
            "profiler": cProfile.Profile(),
            "stats": None,
            "file_path": None
        }
        
        self.profiles[profile_id] = profile
        
        # Start profiling
        profile["profiler"].enable()
        
        return profile_id
    
    def end_profiling(self, profile_id: str):
        """End profiling.
        
        Args:
            profile_id: ID of the profile
        """
        if not self.enabled or not self.profiling_enabled or profile_id == "disabled" or profile_id not in self.profiles:
            return
        
        # End profiling
        profile = self.profiles[profile_id]
        profile["profiler"].disable()
        profile["end_time"] = time.time()
        profile["duration"] = profile["end_time"] - profile["start_time"]
        
        # Create a StringIO to capture text output
        s = io.StringIO()
        ps = pstats.Stats(profile["profiler"], stream=s).sort_stats('cumulative')
        ps.print_stats()
        
        profile["stats"] = s.getvalue()
        
        # Generate filename based on profile name and time
        timestamp = datetime.fromtimestamp(profile["start_time"]).strftime("%Y%m%d_%H%M%S")
        name_part = profile["name"].replace(" ", "_").lower()
        filename = f"{timestamp}_{name_part}_{profile_id}.prof"
        file_path = os.path.join(self.debug_dir, "profiles", filename)
        
        # Save to file
        try:
            profile["profiler"].dump_stats(file_path)
            profile["file_path"] = file_path
            
            self.logger.debug(f"Saved profile {profile_id} to {filename}")
        except Exception as e:
            self.logger.error(f"Error saving profile {profile_id}: {str(e)}")
    
    def get_profile(self, profile_id: str) -> Optional[Dict[str, Any]]:
        """Get a profile by ID.
        
        Args:
            profile_id: ID of the profile
            
        Returns:
            Profile or None
        """
        if not self.enabled or not self.profiling_enabled or profile_id not in self.profiles:
            return None
        
        profile = self.profiles[profile_id].copy()
        
        # Remove the actual profiler object
        if "profiler" in profile:
            del profile["profiler"]
        
        return profile
    
    def agent_trace(self, func=None, trace_name=None):
        """Decorator to trace agent execution.
        
        Args:
            func: The function to trace
            trace_name: Name for the trace
            
        Returns:
            Decorated function
        """
        def decorator(f):
            @wraps(f)
            async def wrapper(self, *args, **kwargs):
                # Use the agent's ID as part of the trace name
                agent_id = getattr(self, "agent_id", "unknown")
                function_name = f.__name__
                
                # Create trace name
                name = trace_name or f"Agent {agent_id}.{function_name}"
                
                # Get the current trace ID (if any) to use as parent
                debugger = get_debugger()
                parent_id = debugger.get_current_trace_id()
                
                # Start a new trace
                trace_id = debugger.start_trace(trace_name=name, parent_id=parent_id)
                
                # Log basic info
                debugger.add_event(trace_id, "start", {
                    "agent_id": agent_id,
                    "function": function_name,
                    "args": str(args),
                    "kwargs": {k: v for k, v in kwargs.items() if k != "password"}
                })
                
                # Start profiling if enabled
                profile_id = None
                if debugger.profiling_enabled:
                    profile_id = debugger.start_profiling(profile_name=f"Profile {name}")
                
                try:
                    # Execute the function
                    result = await f(self, *args, **kwargs)
                    
                    # Log the result
                    debugger.add_event(trace_id, "result", {
                        "result_type": type(result).__name__,
                        "result_summary": str(result)[:1000]
                    })
                    
                    # End the trace
                    debugger.end_trace(trace_id, status="completed", result=result)
                    
                    return result
                
                except Exception as e:
                    # Log the exception
                    debugger.add_event(trace_id, "exception", {
                        "exception_type": type(e).__name__,
                        "exception_message": str(e),
                        "exception_traceback": traceback.format_exc()
                    })
                    
                    # End the trace with error
                    debugger.end_trace(trace_id, status="error", exception=e)
                    
                    raise
                
                finally:
                    # End profiling if enabled
                    if debugger.profiling_enabled and profile_id:
                        debugger.end_profiling(profile_id)
            
            return wrapper
        
        if func:
            return decorator(func)
        return decorator
    
    def workflow_trace(self, func=None, trace_name=None):
        """Decorator to trace workflow execution.
        
        Args:
            func: The function to trace
            trace_name: Name for the trace
            
        Returns:
            Decorated function
        """
        def decorator(f):
            @wraps(f)
            async def wrapper(self, *args, **kwargs):
                # Extract workflow info from arguments
                workflow_id = kwargs.get("workflow_id", "unknown")
                workflow_def = kwargs.get("workflow_def", {})
                workflow_name = workflow_def.get("name", "Unknown Workflow")
                
                # Create trace name
                name = trace_name or f"Workflow {workflow_name} ({workflow_id})"
                
                # Start a new trace (top-level)
                debugger = get_debugger()
                trace_id = debugger.start_trace(trace_name=name)
                
                # Log basic info
                debugger.add_event(trace_id, "start", {
                    "workflow_id": workflow_id,
                    "workflow_name": workflow_name,
                    "workflow_def": workflow_def,
                    "params": kwargs.get("workflow_params", {})
                })
                
                # Start profiling if enabled
                profile_id = None
                if debugger.profiling_enabled:
                    profile_id = debugger.start_profiling(profile_name=f"Profile {name}")
                
                try:
                    # Execute the function
                    result = await f(self, *args, **kwargs)
                    
                    # Log the result
                    debugger.add_event(trace_id, "result", {
                        "status": result.get("status", "unknown"),
                        "result_summary": str(result)[:1000]
                    })
                    
                    # End the trace
                    debugger.end_trace(trace_id, status="completed", result=result)
                    
                    return result
                
                except Exception as e:
                    # Log the exception
                    debugger.add_event(trace_id, "exception", {
                        "exception_type": type(e).__name__,
                        "exception_message": str(e),
                        "exception_traceback": traceback.format_exc()
                    })
                    
                    # End the trace with error
                    debugger.end_trace(trace_id, status="error", exception=e)
                    
                    raise
                
                finally:
                    # End profiling if enabled
                    if debugger.profiling_enabled and profile_id:
                        debugger.end_profiling(profile_id)
            
            return wrapper
        
        if func:
            return decorator(func)
        return decorator
    
    def step_trace(self, func=None, trace_name=None):
        """Decorator to trace workflow step execution.
        
        Args:
            func: The function to trace
            trace_name: Name for the trace
            
        Returns:
            Decorated function
        """
        def decorator(f):
            @wraps(f)
            async def wrapper(self, *args, **kwargs):
                # Extract step info from arguments
                step = args[0] if args else {}
                step_name = step.get("name", "Unknown Step")
                agent_id = step.get("agent_id", "unknown")
                
                # Create trace name
                name = trace_name or f"Step {step_name} ({agent_id})"
                
                # Get the current trace ID (if any) to use as parent
                debugger = get_debugger()
                parent_id = debugger.get_current_trace_id()
                
                # Start a new trace
                trace_id = debugger.start_trace(trace_name=name, parent_id=parent_id)
                
                # Log basic info
                debugger.add_event(trace_id, "start", {
                    "step_name": step_name,
                    "agent_id": agent_id,
                    "params": step.get("params", {})
                })
                
                # Start profiling if enabled
                profile_id = None
                if debugger.profiling_enabled:
                    profile_id = debugger.start_profiling(profile_name=f"Profile {name}")
                
                try:
                    # Execute the function
                    result = await f(self, *args, **kwargs)
                    
                    # Log the result
                    debugger.add_event(trace_id, "result", {
                        "result_type": type(result).__name__,
                        "result_summary": str(result)[:1000]
                    })
                    
                    # End the trace
                    debugger.end_trace(trace_id, status="completed", result=result)
                    
                    return result
                
                except Exception as e:
                    # Log the exception
                    debugger.add_event(trace_id, "exception", {
                        "exception_type": type(e).__name__,
                        "exception_message": str(e),
                        "exception_traceback": traceback.format_exc()
                    })
                    
                    # End the trace with error
                    debugger.end_trace(trace_id, status="error", exception=e)
                    
                    raise
                
                finally:
                    # End profiling if enabled
                    if debugger.profiling_enabled and profile_id:
                        debugger.end_profiling(profile_id)
            
            return wrapper
        
        if func:
            return decorator(func)
        return decorator
    
    def function_trace(self, func=None, trace_name=None):
        """Decorator to trace function execution.
        
        Args:
            func: The function to trace
            trace_name: Name for the trace
            
        Returns:
            Decorated function
        """
        def decorator(f):
            if asyncio.iscoroutinefunction(f):
                @wraps(f)
                async def async_wrapper(*args, **kwargs):
                    # Create trace name
                    name = trace_name or f"Function {f.__module__}.{f.__name__}"
                    
                    # Get the current trace ID (if any) to use as parent
                    debugger = get_debugger()
                    parent_id = debugger.get_current_trace_id()
                    
                    # Start a new trace
                    trace_id = debugger.start_trace(trace_name=name, parent_id=parent_id)
                    
                    # Log basic info
                    debugger.add_event(trace_id, "start", {
                        "function": f.__name__,
                        "module": f.__module__,
                        "args": str(args),
                        "kwargs": {k: v for k, v in kwargs.items() if k != "password"}
                    })
                    
                    # Start profiling if enabled
                    profile_id = None
                    if debugger.profiling_enabled:
                        profile_id = debugger.start_profiling(profile_name=f"Profile {name}")
                    
                    try:
                        # Execute the function
                        result = await f(*args, **kwargs)
                        
                        # Log the result
                        debugger.add_event(trace_id, "result", {
                            "result_type": type(result).__name__,
                            "result_summary": str(result)[:1000]
                        })
                        
                        # End the trace
                        debugger.end_trace(trace_id, status="completed", result=result)
                        
                        return result
                    
                    except Exception as e:
                        # Log the exception
                        debugger.add_event(trace_id, "exception", {
                            "exception_type": type(e).__name__,
                            "exception_message": str(e),
                            "exception_traceback": traceback.format_exc()
                        })
                        
                        # End the trace with error
                        debugger.end_trace(trace_id, status="error", exception=e)
                        
                        raise
                    
                    finally:
                        # End profiling if enabled
                        if debugger.profiling_enabled and profile_id:
                            debugger.end_profiling(profile_id)
                
                return async_wrapper
            else:
                @wraps(f)
                def sync_wrapper(*args, **kwargs):
                    # Create trace name
                    name = trace_name or f"Function {f.__module__}.{f.__name__}"
                    
                    # Get the current trace ID (if any) to use as parent
                    debugger = get_debugger()
                    parent_id = debugger.get_current_trace_id()
                    
                    # Start a new trace
                    trace_id = debugger.start_trace(trace_name=name, parent_id=parent_id)
                    
                    # Log basic info
                    debugger.add_event(trace_id, "start", {
                        "function": f.__name__,
                        "module": f.__module__,
                        "args": str(args),
                        "kwargs": {k: v for k, v in kwargs.items() if k != "password"}
                    })
                    
                    # Start profiling if enabled
                    profile_id = None
                    if debugger.profiling_enabled:
                        profile_id = debugger.start_profiling(profile_name=f"Profile {name}")
                    
                    try:
                        # Execute the function
                        result = f(*args, **kwargs)
                        
                        # Log the result
                        debugger.add_event(trace_id, "result", {
                            "result_type": type(result).__name__,
                            "result_summary": str(result)[:1000]
                        })
                        
                        # End the trace
                        debugger.end_trace(trace_id, status="completed", result=result)
                        
                        return result
                    
                    except Exception as e:
                        # Log the exception
                        debugger.add_event(trace_id, "exception", {
                            "exception_type": type(e).__name__,
                            "exception_message": str(e),
                            "exception_traceback": traceback.format_exc()
                        })
                        
                        # End the trace with error
                        debugger.end_trace(trace_id, status="error", exception=e)
                        
                        raise
                    
                    finally:
                        # End profiling if enabled
                        if debugger.profiling_enabled and profile_id:
                            debugger.end_profiling(profile_id)
                
                return sync_wrapper
        
        if func:
            return decorator(func)
        return decorator


# Create a global debugger instance
_debugger = None

def get_debugger() -> AgentDebugger:
    """Get the global debugger instance.
    
    Returns:
        Global debugger instance
    """
    global _debugger
    if _debugger is None:
        debug_dir = os.environ.get("DEBUG_DIR", "data/debug")
        enabled = os.environ.get("DEBUG_ENABLED", "true").lower() == "true"
        profiling_enabled = os.environ.get("PROFILING_ENABLED", "false").lower() == "true"
        
        _debugger = AgentDebugger(
            debug_dir=debug_dir,
            enabled=enabled,
            profiling_enabled=profiling_enabled
        )
    
    return _debugger


# Create a command-line interface for the debugger
def main():
    """Command-line interface for the debugger."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Agent debugger")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # List traces
    list_traces_parser = subparsers.add_parser("list-traces", help="List traces")
    list_traces_parser.add_argument("--debug-dir", help="Debug directory")
    
    # View trace
    view_trace_parser = subparsers.add_parser("view-trace", help="View a trace")
    view_trace_parser.add_argument("trace_file", help="Trace file to view")
    view_trace_parser.add_argument("--debug-dir", help="Debug directory")
    
    # List profiles
    list_profiles_parser = subparsers.add_parser("list-profiles", help="List profiles")
    list_profiles_parser.add_argument("--debug-dir", help="Debug directory")
    
    # View profile
    view_profile_parser = subparsers.add_parser("view-profile", help="View a profile")
    view_profile_parser.add_argument("profile_file", help="Profile file to view")
    view_profile_parser.add_argument("--debug-dir", help="Debug directory")
    view_profile_parser.add_argument("--sort", default="cumulative", 
                                   help="Sort order (cumulative, time, calls)")
    view_profile_parser.add_argument("--limit", type=int, default=20, 
                                   help="Number of entries to show")
    
    args = parser.parse_args()
    
    if args.command == "list-traces":
        debug_dir = args.debug_dir or "data/debug"
        traces_dir = os.path.join(debug_dir, "traces")
        
        if not os.path.exists(traces_dir):
            print(f"Traces directory not found: {traces_dir}")
            return
        
        trace_files = os.listdir(traces_dir)
        trace_files.sort(reverse=True)  # Newest first
        
        print(f"Found {len(trace_files)} traces:")
        for i, file in enumerate(trace_files[:20], 1):  # Show at most 20
            print(f"{i}. {file}")
        
        if len(trace_files) > 20:
            print(f"...and {len(trace_files) - 20} more")
    
    elif args.command == "view-trace":
        debug_dir = args.debug_dir or "data/debug"
        traces_dir = os.path.join(debug_dir, "traces")
        
        trace_file = args.trace_file
        if not os.path.exists(os.path.join(traces_dir, trace_file)):
            # Check if it's just a prefix
            matching_files = [f for f in os.listdir(traces_dir) if f.endswith(f"{trace_file}.json")]
            if matching_files:
                trace_file = matching_files[0]
            else:
                print(f"Trace file not found: {trace_file}")
                return
        
        try:
            with open(os.path.join(traces_dir, trace_file)) as f:
                trace = json.load(f)
            
            print(f"Trace: {trace['name']} ({trace['id']})")
            print(f"Status: {trace['status']}")
            print(f"Duration: {trace['duration']:.3f}s")
            print(f"Start time: {datetime.fromtimestamp(trace['start_time']).strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"End time: {datetime.fromtimestamp(trace['end_time']).strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Events: {len(trace['events'])}")
            print(f"Children: {len(trace['children'])}")
            
            if trace.get("exception"):
                print("\nException:")
                print(f"Type: {trace['exception']['type']}")
                print(f"Message: {trace['exception']['message']}")
                print("\nTraceback:")
                print(trace['exception']['traceback'])
            
            print("\nEvents:")
            for i, event in enumerate(trace['events'], 1):
                event_time = datetime.fromtimestamp(event['time']).strftime('%H:%M:%S')
                print(f"{i}. [{event_time}] {event['type']}")
                
                if event['type'] == "exception":
                    print(f"    Exception: {event['data'].get('exception_type')}: {event['data'].get('exception_message')}")
            
            print("\nChildren:")
            for i, child in enumerate(trace['children'], 1):
                print(f"{i}. {child['name']} ({child['status']}) - {child['duration']:.3f}s")
        
        except Exception as e:
            print(f"Error loading trace: {str(e)}")
    
    elif args.command == "list-profiles":
        debug_dir = args.debug_dir or "data/debug"
        profiles_dir = os.path.join(debug_dir, "profiles")
        
        if not os.path.exists(profiles_dir):
            print(f"Profiles directory not found: {profiles_dir}")
            return
        
        profile_files = os.listdir(profiles_dir)
        profile_files.sort(reverse=True)  # Newest first
        
        print(f"Found {len(profile_files)} profiles:")
        for i, file in enumerate(profile_files[:20], 1):  # Show at most 20
            print(f"{i}. {file}")
        
        if len(profile_files) > 20:
            print(f"...and {len(profile_files) - 20} more")
    
    elif args.command == "view-profile":
        debug_dir = args.debug_dir or "data/debug"
        profiles_dir = os.path.join(debug_dir, "profiles")
        
        profile_file = args.profile_file
        if not os.path.exists(os.path.join(profiles_dir, profile_file)):
            # Check if it's just a prefix
            matching_files = [f for f in os.listdir(profiles_dir) if f.endswith(f"{profile_file}.prof")]
            if matching_files:
                profile_file = matching_files[0]
            else:
                print(f"Profile file not found: {profile_file}")
                return
        
        try:
            # Load the profile
            profile_path = os.path.join(profiles_dir, profile_file)
            
            # Use pstats to analyze the profile
            p = pstats.Stats(profile_path)
            
            # Display general stats
            print(f"Profile: {profile_file}")
            print(f"Total calls: {p.total_calls}")
            print(f"Total time: {p.total_tt:.3f}s")
            
            # Print the top functions
            print(f"\nTop {args.limit} functions by {args.sort} time:")
            p.sort_stats(args.sort)
            p.print_stats(args.limit)
        
        except Exception as e:
            print(f"Error loading profile: {str(e)}")


if __name__ == "__main__":
    main()