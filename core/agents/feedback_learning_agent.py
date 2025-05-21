# core/agents/feedback_learning_agent.py
from typing import Dict, List, Any, Optional, Union
import logging
import asyncio
import uuid
import json
import os
from datetime import datetime

from core.agents.base_agent import BaseAgent

class FeedbackLearningAgent(BaseAgent):
    """Agent for collecting feedback and improving system performance over time."""
    
    def __init__(self, agent_id: str = None, name: str = None, 
                 storage_path: str = None, config: Dict[str, Any] = None):
        """Initialize a new feedback learning agent.
        
        Args:
            agent_id: Unique identifier for the agent
            name: Human-readable name for the agent
            storage_path: Path to store feedback data
            config: Additional configuration for the agent
        """
        agent_config = config or {}
        if storage_path:
            agent_config["storage_path"] = storage_path
        else:
            agent_config["storage_path"] = "data/feedback"
        
        super().__init__(agent_id, name or "feedback_learning_agent", agent_config)
        
        # Initialize feedback storage
        os.makedirs(agent_config["storage_path"], exist_ok=True)
        
        self.feedback_file = os.path.join(agent_config["storage_path"], "feedback_data.json")
        self.model_file = os.path.join(agent_config["storage_path"], "feedback_model.json")
        
        # Load existing feedback data if available
        self.feedback_data = self._load_feedback_data()
        self.model = self._load_model()
        
        self.logger.info(f"Initialized feedback learning agent with {len(self.feedback_data)} feedback items")
    
    def _load_feedback_data(self) -> List[Dict[str, Any]]:
        """Load feedback data from storage.
        
        Returns:
            List of feedback data items
        """
        if os.path.exists(self.feedback_file):
            try:
                with open(self.feedback_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading feedback data: {str(e)}")
        
        return []
    
    def _save_feedback_data(self):
        """Save feedback data to storage."""
        try:
            with open(self.feedback_file, 'w') as f:
                json.dump(self.feedback_data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving feedback data: {str(e)}")
    
    def _load_model(self) -> Dict[str, Any]:
        """Load feedback model from storage.
        
        Returns:
            Feedback model data
        """
        if os.path.exists(self.model_file):
            try:
                with open(self.model_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading model: {str(e)}")
        
        # Default model structure
        return {
            "version": 1,
            "last_updated": datetime.now().isoformat(),
            "weights": {
                "technical_fit": 0.3,
                "past_performance": 0.25,
                "competition": 0.15,
                "financial": 0.2,
                "strategic": 0.1
            },
            "agency_insights": {},
            "technology_insights": {},
            "success_factors": {}
        }
    
    def _save_model(self):
        """Save feedback model to storage."""
        try:
            self.model["last_updated"] = datetime.now().isoformat()
            with open(self.model_file, 'w') as f:
                json.dump(self.model, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
    
    async def execute_task(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute feedback learning task.
        
        Args:
            params: Parameters for the task
                - operation: Operation to perform (add_feedback, get_insights, update_model)
                - feedback: Feedback data to add
                - analysis_type: Type of analysis to perform
                
        Returns:
            Results of the task
        """
        operation = params.get("operation")
        if not operation:
            raise ValueError("No operation specified for feedback learning task")
        
        self.logger.info(f"Starting feedback learning operation: {operation}")
        
        if operation == "add_feedback":
            return await self._add_feedback(params)
        elif operation == "get_insights":
            return await self._get_insights(params)
        elif operation == "update_model":
            return await self._update_model(params)
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    async def _add_feedback(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Add new feedback data.
        
        Args:
            params: Parameters for adding feedback
                - feedback: Feedback data to add
                  - opportunity_id: ID of the opportunity
                  - bid_decision: Whether the company bid on the opportunity
                  - win_result: Whether the company won the bid
                  - feedback_notes: Notes about the feedback
                  - scores: Original scores for the opportunity
                
        Returns:
            Results of adding feedback
        """
        feedback = params.get("feedback")
        if not feedback:
            raise ValueError("No feedback data provided")
        
        # Add timestamp to feedback
        feedback["timestamp"] = datetime.now().isoformat()
        
        # Generate unique ID for feedback
        feedback_id = str(uuid.uuid4())
        feedback["feedback_id"] = feedback_id
        
        # Add to feedback data
        self.feedback_data.append(feedback)
        
        # Save updated feedback data
        self._save_feedback_data()
        
        # Check if we should update the model
        if len(self.feedback_data) % 10 == 0:
            await self._update_model({"auto_triggered": True})
        
        return {
            "operation": "add_feedback",
            "feedback_id": feedback_id,
            "timestamp": feedback["timestamp"],
            "status": "success"
        }
    
    async def _get_insights(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get insights from feedback data.
        
        Args:
            params: Parameters for getting insights
                - insight_type: Type of insights to get
                - filters: Filters to apply to feedback data
                
        Returns:
            Requested insights
        """
        insight_type = params.get("insight_type", "summary")
        filters = params.get("filters", {})
        
        # Filter feedback data based on provided filters
        filtered_data = self.feedback_data
        for key, value in filters.items():
            if key in ["agency", "technologies", "bid_decision", "win_result"]:
                if isinstance(value, list):
                    filtered_data = [
                        item for item in filtered_data
                        if any(v in item.get(key, []) for v in value)
                    ]
                else:
                    filtered_data = [
                        item for item in filtered_data
                        if item.get(key) == value
                    ]
        
        if insight_type == "summary":
            return self._generate_summary_insights(filtered_data)
        elif insight_type == "agency":
            return self._generate_agency_insights(filtered_data)
        elif insight_type == "technology":
            return self._generate_technology_insights(filtered_data)
        elif insight_type == "success_factors":
            return self._generate_success_factor_insights(filtered_data)
        else:
            raise ValueError(f"Unknown insight_type: {insight_type}")
    
    async def _update_model(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Update the feedback model based on accumulated feedback.
        
        Args:
            params: Parameters for updating the model
                - force: Force update even if not enough new data
                
        Returns:
            Results of model update
        """
        force = params.get("force", False)
        auto_triggered = params.get("auto_triggered", False)
        
        # Check if we have enough data to update the model
        if len(self.feedback_data) < 5 and not force:
            return {
                "operation": "update_model",
                "status": "skipped",
                "reason": "Not enough feedback data"
            }
        
        # Update weights based on success patterns
        new_weights = self._calculate_optimal_weights()
        if new_weights:
            self.model["weights"] = new_weights
        
        # Update agency insights
        agency_insights = self._calculate_agency_insights()
        if agency_insights:
            self.model["agency_insights"] = agency_insights
        
        # Update technology insights
        technology_insights = self._calculate_technology_insights()
        if technology_insights:
            self.model["technology_insights"] = technology_insights
        
        # Update success factors
        success_factors = self._calculate_success_factors()
        if success_factors:
            self.model["success_factors"] = success_factors
        
        # Save updated model
        self._save_model()
        
        return {
            "operation": "update_model",
            "status": "success",
            "auto_triggered": auto_triggered,
            "updated_components": ["weights", "agency_insights", "technology_insights", "success_factors"],
            "updated_at": datetime.now().isoformat()
        }
    
    def _generate_summary_insights(self, feedback_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary insights from feedback data.
        
        Args:
            feedback_data: Filtered feedback data
            
        Returns:
            Summary insights
        """
        if not feedback_data:
            return {
                "insight_type": "summary",
                "message": "No feedback data available for summary"
            }
        
        # Calculate bid rate and win rate
        total_opportunities = len(feedback_data)
        bid_opportunities = sum(1 for item in feedback_data if item.get("bid_decision") == True)
        win_opportunities = sum(1 for item in feedback_data 
                              if item.get("bid_decision") == True and item.get("win_result") == True)
        
        bid_rate = (bid_opportunities / total_opportunities) * 100 if total_opportunities > 0 else 0
        win_rate = (win_opportunities / bid_opportunities) * 100 if bid_opportunities > 0 else 0
        
        # Calculate average scores
        avg_scores = {
            "technical_fit": 0,
            "past_performance": 0,
            "competition": 0,
            "financial": 0,
            "strategic": 0,
            "overall": 0
        }
        
        score_count = 0
        for item in feedback_data:
            if "scores" in item:
                scores = item["scores"]
                for key in avg_scores:
                    if key in scores:
                        avg_scores[key] += scores[key]
                score_count += 1
        
        if score_count > 0:
            for key in avg_scores:
                avg_scores[key] = round(avg_scores[key] / score_count, 1)
        
        # Calculate score differences between won and lost bids
        won_scores = {key: 0 for key in avg_scores}
        lost_scores = {key: 0 for key in avg_scores}
        
        won_count = 0
        lost_count = 0
        
        for item in feedback_data:
            if item.get("bid_decision") == True and "scores" in item:
                if item.get("win_result") == True:
                    for key, value in item["scores"].items():
                        if key in won_scores:
                            won_scores[key] += value
                    won_count += 1
                else:
                    for key, value in item["scores"].items():
                        if key in lost_scores:
                            lost_scores[key] += value
                    lost_count += 1
        
        if won_count > 0:
            for key in won_scores:
                won_scores[key] = round(won_scores[key] / won_count, 1)
        
        if lost_count > 0:
            for key in lost_scores:
                lost_scores[key] = round(lost_scores[key] / lost_count, 1)
        
        # Calculate score differences
        score_differences = {}
        if won_count > 0 and lost_count > 0:
            for key in won_scores:
                score_differences[key] = round(won_scores[key] - lost_scores[key], 1)
        
        return {
            "insight_type": "summary",
            "opportunity_count": total_opportunities,
            "bid_count": bid_opportunities,
            "win_count": win_opportunities,
            "bid_rate": round(bid_rate, 1),
            "win_rate": round(win_rate, 1),
            "average_scores": avg_scores,
            "won_bid_scores": won_scores if won_count > 0 else None,
            "lost_bid_scores": lost_scores if lost_count > 0 else None,
            "score_differences": score_differences if won_count > 0 and lost_count > 0 else None
        }
    
    def _generate_agency_insights(self, feedback_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate agency-specific insights from feedback data.
        
        Args:
            feedback_data: Filtered feedback data
            
        Returns:
            Agency insights
        """
        if not feedback_data:
            return {
                "insight_type": "agency",
                "message": "No feedback data available for agency insights"
            }
        
        # Group feedback by agency
        agency_data = {}
        
        for item in feedback_data:
            agency = item.get("agency")
            if not agency:
                continue
            
            if agency not in agency_data:
                agency_data[agency] = {
                    "opportunity_count": 0,
                    "bid_count": 0,
                    "win_count": 0
                }
            
            agency_data[agency]["opportunity_count"] += 1
            
            if item.get("bid_decision") == True:
                agency_data[agency]["bid_count"] += 1
                
                if item.get("win_result") == True:
                    agency_data[agency]["win_count"] += 1
        
        # Calculate rates for each agency
        for agency, data in agency_data.items():
            data["bid_rate"] = (data["bid_count"] / data["opportunity_count"]) * 100 if data["opportunity_count"] > 0 else 0
            data["win_rate"] = (data["win_count"] / data["bid_count"]) * 100 if data["bid_count"] > 0 else 0
            
            # Round for readability
            data["bid_rate"] = round(data["bid_rate"], 1)
            data["win_rate"] = round(data["win_rate"], 1)
        
        return {
            "insight_type": "agency",
            "agency_insights": agency_data
        }
    
    def _generate_technology_insights(self, feedback_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate technology-specific insights from feedback data.
        
        Args:
            feedback_data: Filtered feedback data
            
        Returns:
            Technology insights
        """
        if not feedback_data:
            return {
                "insight_type": "technology",
                "message": "No feedback data available for technology insights"
            }
        
        # Group feedback by technology
        tech_data = {}
        
        for item in feedback_data:
            technologies = item.get("technologies", [])
            for tech in technologies:
                if not tech:
                    continue
                
                if tech not in tech_data:
                    tech_data[tech] = {
                        "opportunity_count": 0,
                        "bid_count": 0,
                        "win_count": 0
                    }
                
                tech_data[tech]["opportunity_count"] += 1
                
                if item.get("bid_decision") == True:
                    tech_data[tech]["bid_count"] += 1
                    
                    if item.get("win_result") == True:
                        tech_data[tech]["win_count"] += 1
        
        # Calculate rates for each technology
        for tech, data in tech_data.items():
            data["bid_rate"] = (data["bid_count"] / data["opportunity_count"]) * 100 if data["opportunity_count"] > 0 else 0
            data["win_rate"] = (data["win_count"] / data["bid_count"]) * 100 if data["bid_count"] > 0 else 0
            
            # Round for readability
            data["bid_rate"] = round(data["bid_rate"], 1)
            data["win_rate"] = round(data["win_rate"], 1)
        
        return {
            "insight_type": "technology",
            "technology_insights": tech_data
        }
    
    def _generate_success_factor_insights(self, feedback_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate success factor insights from feedback data.
        
        Args:
            feedback_data: Filtered feedback data
            
        Returns:
            Success factor insights
        """
        if not feedback_data:
            return {
                "insight_type": "success_factors",
                "message": "No feedback data available for success factor insights"
            }
        
        # Only look at bids that were submitted
        bid_data = [item for item in feedback_data if item.get("bid_decision") == True]
        
        if not bid_data:
            return {
                "insight_type": "success_factors",
                "message": "No bid data available for success factor insights"
            }
        
        # Group by win/loss
        won_bids = [item for item in bid_data if item.get("win_result") == True]
        lost_bids = [item for item in bid_data if item.get("win_result") == False]
        
        # Calculate average scores for won vs lost bids
        won_scores = {
            "technical_fit": 0,
            "past_performance": 0,
            "competition": 0,
            "financial": 0,
            "strategic": 0,
            "overall": 0
        }
        
        lost_scores = {key: 0 for key in won_scores}
        
        won_count = len(won_bids)
        lost_count = len(lost_bids)
        
        for item in won_bids:
            if "scores" in item:
                for key, value in item["scores"].items():
                    if key in won_scores:
                        won_scores[key] += value
        
        for item in lost_bids:
            if "scores" in item:
                for key, value in item["scores"].items():
                    if key in lost_scores:
                        lost_scores[key] += value
        
        if won_count > 0:
            for key in won_scores:
                won_scores[key] = round(won_scores[key] / won_count, 1)
        
        if lost_count > 0:
            for key in lost_scores:
                lost_scores[key] = round(lost_scores[key] / lost_count, 1)
        
        # Calculate differences
        score_differences = {}
        if won_count > 0 and lost_count > 0:
            for key in won_scores:
                score_differences[key] = round(won_scores[key] - lost_scores[key], 1)
        
        # Identify key success factors
        key_factors = []
        if score_differences:
            # Sort factors by difference
            sorted_factors = sorted(
                score_differences.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            
            # Top 3 factors with significant differences
            for factor, diff in sorted_factors:
                if abs(diff) >= 5:  # Only include if difference is significant
                    key_factors.append({
                        "factor": factor,
                        "difference": diff,
                        "won_average": won_scores[factor],
                        "lost_average": lost_scores[factor]
                    })
                
                if len(key_factors) >= 3:
                    break
        
        return {
            "insight_type": "success_factors",
            "total_bids": len(bid_data),
            "won_bids": won_count,
            "lost_bids": lost_count,
            "win_rate": round((won_count / len(bid_data)) * 100, 1) if len(bid_data) > 0 else 0,
            "won_scores": won_scores if won_count > 0 else None,
            "lost_scores": lost_scores if lost_count > 0 else None,
            "score_differences": score_differences if won_count > 0 and lost_count > 0 else None,
            "key_success_factors": key_factors
        }
    
    def _calculate_optimal_weights(self) -> Dict[str, float]:
        """Calculate optimal weights based on feedback data.
        
        Returns:
            Optimized weights
        """
        # Only consider opportunities with bid decisions and results
        relevant_data = [
            item for item in self.feedback_data
            if item.get("bid_decision") is not None and 
               item.get("win_result") is not None and
               "scores" in item
        ]
        
        if len(relevant_data) < 5:
            return None
        
        # Simple approach: weight factors based on their predictive power
        factor_scores = {
            "technical_fit": [],
            "past_performance": [],
            "competition": [],
            "financial": [],
            "strategic": []
        }
        
        # Collect scores for won vs lost opportunities
        won_scores = {key: [] for key in factor_scores}
        lost_scores = {key: [] for key in factor_scores}
        
        for item in relevant_data:
            if not item.get("bid_decision"):
                continue
                
            scores = item.get("scores", {})
            for factor in factor_scores:
                if factor in scores:
                    if item.get("win_result"):
                        won_scores[factor].append(scores[factor])
                    else:
                        lost_scores[factor].append(scores[factor])
        
        # Calculate average score difference between won and lost bids
        score_diffs = {}
        for factor in factor_scores:
            if won_scores[factor] and lost_scores[factor]:
                won_avg = sum(won_scores[factor]) / len(won_scores[factor])
                lost_avg = sum(lost_scores[factor]) / len(lost_scores[factor])
                score_diffs[factor] = abs(won_avg - lost_avg)
            else:
                score_diffs[factor] = 0
        
        # Normalize differences to weights (sum to 1)
        total_diff = sum(score_diffs.values())
        weights = {}
        
        if total_diff > 0:
            for factor, diff in score_diffs.items():
                weights[factor] = round(diff / total_diff, 2)
                
            # Ensure weights sum to 1
            sum_weights = sum(weights.values())
            if sum_weights != 1:
                adjustment = 1 / sum_weights
                for factor in weights:
                    weights[factor] = round(weights[factor] * adjustment, 2)
                
                # Fix any rounding issues
                remaining = 1 - sum(weights.values())
                if remaining != 0:
                    # Add remainder to the factor with highest weight
                    max_factor = max(weights, key=weights.get)
                    weights[max_factor] += remaining
        else:
            # Default weights if no meaningful differences
            weights = {
                "technical_fit": 0.3,
                "past_performance": 0.25,
                "competition": 0.15,
                "financial": 0.2,
                "strategic": 0.1
            }
        
        return weights
    
    def _calculate_agency_insights(self) -> Dict[str, Dict[str, Any]]:
        """Calculate agency-specific insights from feedback data.
        
        Returns:
            Agency insights
        """
        insights = {}
        
        # Group feedback by agency
        agency_data = {}
        
        for item in self.feedback_data:
            agency = item.get("agency")
            if not agency:
                continue
            
            if agency not in agency_data:
                agency_data[agency] = {
                    "opportunities": [],
                    "bids": [],
                    "wins": []
                }
            
            agency_data[agency]["opportunities"].append(item)
            
            if item.get("bid_decision") == True:
                agency_data[agency]["bids"].append(item)
                
                if item.get("win_result") == True:
                    agency_data[agency]["wins"].append(item)
        
        # Calculate insights for each agency
        for agency, data in agency_data.items():
            opp_count = len(data["opportunities"])
            bid_count = len(data["bids"])
            win_count = len(data["wins"])
            
            if opp_count < 3:
                continue  # Skip agencies with too little data
            
            bid_rate = (bid_count / opp_count) * 100 if opp_count > 0 else 0
            win_rate = (win_count / bid_count) * 100 if bid_count > 0 else 0
            
            # Calculate average scores for this agency
            avg_scores = {
                "technical_fit": 0,
                "past_performance": 0,
                "competition": 0,
                "financial": 0,
                "strategic": 0,
                "overall": 0
            }
            
            score_count = 0
            for item in data["opportunities"]:
                if "scores" in item:
                    for key, value in item["scores"].items():
                        if key in avg_scores:
                            avg_scores[key] += value
                    score_count += 1
            
            if score_count > 0:
                for key in avg_scores:
                    avg_scores[key] = round(avg_scores[key] / score_count, 1)
            
            # Calculate winning scores for this agency
            win_scores = {key: 0 for key in avg_scores}
            win_score_count = 0
            
            for item in data["wins"]:
                if "scores" in item:
                    for key, value in item["scores"].items():
                        if key in win_scores:
                            win_scores[key] += value
                    win_score_count += 1
            
            if win_score_count > 0:
                for key in win_scores:
                    win_scores[key] = round(win_scores[key] / win_score_count, 1)
            
            # Store insights
            insights[agency] = {
                "opportunity_count": opp_count,
                "bid_count": bid_count,
                "win_count": win_count,
                "bid_rate": round(bid_rate, 1),
                "win_rate": round(win_rate, 1),
                "average_scores": avg_scores,
                "winning_scores": win_scores if win_score_count > 0 else None,
                "last_updated": datetime.now().isoformat()
            }
        
        return insights
    
    def _calculate_technology_insights(self) -> Dict[str, Dict[str, Any]]:
        """Calculate technology-specific insights from feedback data.
        
        Returns:
            Technology insights
        """
        insights = {}
        
        # Group feedback by technology
        tech_data = {}
        
        for item in self.feedback_data:
            technologies = item.get("technologies", [])
            for tech in technologies:
                if not tech:
                    continue
                
                if tech not in tech_data:
                    tech_data[tech] = {
                        "opportunities": [],
                        "bids": [],
                        "wins": []
                    }
                
                tech_data[tech]["opportunities"].append(item)
                
                if item.get("bid_decision") == True:
                    tech_data[tech]["bids"].append(item)
                    
                    if item.get("win_result") == True:
                        tech_data[tech]["wins"].append(item)
        
        # Calculate insights for each technology
        for tech, data in tech_data.items():
            opp_count = len(data["opportunities"])
            bid_count = len(data["bids"])
            win_count = len(data["wins"])
            
            if opp_count < 3:
                continue  # Skip technologies with too little data
            
            bid_rate = (bid_count / opp_count) * 100 if opp_count > 0 else 0
            win_rate = (win_count / bid_count) * 100 if bid_count > 0 else 0
            
            # Calculate average technical score for this technology
            tech_scores = []
            for item in data["opportunities"]:
                if "scores" in item and "technical_fit" in item["scores"]:
                    tech_scores.append(item["scores"]["technical_fit"])
            
            avg_tech_score = sum(tech_scores) / len(tech_scores) if tech_scores else 0
            
            # Store insights
            insights[tech] = {
                "opportunity_count": opp_count,
                "bid_count": bid_count,
                "win_count": win_count,
                "bid_rate": round(bid_rate, 1),
                "win_rate": round(win_rate, 1),
                "average_technical_score": round(avg_tech_score, 1),
                "last_updated": datetime.now().isoformat()
            }
        
        return insights
    
    def _calculate_success_factors(self) -> Dict[str, Any]:
        """Calculate success factors from feedback data.
        
        Returns:
            Success factors
        """
        # Only look at bids that were submitted
        bid_data = [item for item in self.feedback_data if item.get("bid_decision") == True]
        
        if len(bid_data) < 5:
            return None
        
        # Group by win/loss
        won_bids = [item for item in bid_data if item.get("win_result") == True]
        lost_bids = [item for item in bid_data if item.get("win_result") == False]
        
        if len(won_bids) < 2 or len(lost_bids) < 2:
            return None
        
        # Calculate average scores for won vs lost bids
        won_scores = {
            "technical_fit": 0,
            "past_performance": 0,
            "competition": 0,
            "financial": 0,
            "strategic": 0,
            "overall": 0
        }
        
        lost_scores = {key: 0 for key in won_scores}
        
        won_count = len(won_bids)
        lost_count = len(lost_bids)
        
        for item in won_bids:
            if "scores" in item:
                for key, value in item["scores"].items():
                    if key in won_scores:
                        won_scores[key] += value
        
        for item in lost_bids:
            if "scores" in item:
                for key, value in item["scores"].items():
                    if key in lost_scores:
                        lost_scores[key] += value
        
        if won_count > 0:
            for key in won_scores:
                won_scores[key] = round(won_scores[key] / won_count, 1)
        
        if lost_count > 0:
            for key in lost_scores:
                lost_scores[key] = round(lost_scores[key] / lost_count, 1)
        
        # Calculate differences
        score_differences = {}
        for key in won_scores:
            score_differences[key] = round(won_scores[key] - lost_scores[key], 1)
        
        # Identify key success factors
        key_factors = []
        sorted_factors = sorted(
            score_differences.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Top factors with significant differences
        for factor, diff in sorted_factors:
            if abs(diff) >= 5:  # Only include if difference is significant
                key_factors.append({
                    "factor": factor,
                    "difference": diff,
                    "won_average": won_scores[factor],
                    "lost_average": lost_scores[factor]
                })
        
        return {
            "total_bids": len(bid_data),
            "won_bids": won_count,
            "lost_bids": lost_count,
            "win_rate": round((won_count / len(bid_data)) * 100, 1),
            "won_scores": won_scores,
            "lost_scores": lost_scores,
            "score_differences": score_differences,
            "key_success_factors": key_factors,
            "last_updated": datetime.now().isoformat()
        }