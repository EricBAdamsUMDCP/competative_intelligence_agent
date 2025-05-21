# core/agents/bid_analysis_agent.py
from typing import Dict, List, Any, Optional
import logging
import asyncio
import uuid
import json
import math

from core.agents.base_agent import BaseAgent

class BidAnalysisAgent(BaseAgent):
    """Agent for analyzing and scoring bid opportunities."""
    
    def __init__(self, agent_id: str = None, name: str = None, 
                 company_profile: Dict[str, Any] = None, config: Dict[str, Any] = None):
        """Initialize a new bid analysis agent.
        
        Args:
            agent_id: Unique identifier for the agent
            name: Human-readable name for the agent
            company_profile: Profile of the company for matching
            config: Additional configuration for the agent
        """
        agent_config = config or {}
        agent_config["company_profile"] = company_profile or {}
        
        super().__init__(agent_id, name or "bid_analysis_agent", agent_config)
        
        self.logger.info("Initialized bid analysis agent")
        
        # Load default weights if none provided
        if "weights" not in agent_config:
            agent_config["weights"] = {
                "technical_fit": 0.3,
                "past_performance": 0.25,
                "competition": 0.15,
                "financial": 0.2,
                "strategic": 0.1
            }
    
    async def execute_task(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute bid analysis task.
        
        Args:
            params: Parameters for the analysis task
                - opportunities: List of opportunities to analyze
                - company_profile: Company profile for matching (optional, override init)
                - weights: Weighting for different factors (optional, override init)
                
        Returns:
            Results of the analysis task
        """
        opportunities = params.get("opportunities", [])
        
        if not opportunities:
            raise ValueError("No opportunities provided for analysis")
        
        # Use provided profile or the one from init
        company_profile = params.get("company_profile", self.config.get("company_profile", {}))
        weights = params.get("weights", self.config.get("weights", {}))
        
        self.logger.info(f"Starting bid analysis of {len(opportunities)} opportunities")
        
        results = {
            "opportunity_count": len(opportunities),
            "analyzed_opportunities": [],
            "recommended_opportunities": []
        }
        
        # Analyze each opportunity
        for opportunity in opportunities:
            analysis = self._analyze_opportunity(opportunity, company_profile, weights)
            results["analyzed_opportunities"].append(analysis)
        
        # Sort by overall score
        results["analyzed_opportunities"].sort(key=lambda x: x["overall_score"], reverse=True)
        
        # Select recommended opportunities (score >= 70)
        results["recommended_opportunities"] = [
            opp for opp in results["analyzed_opportunities"] 
            if opp["overall_score"] >= 70
        ]
        
        return results
    
    def _analyze_opportunity(self, opportunity: Dict[str, Any], 
                            company_profile: Dict[str, Any],
                            weights: Dict[str, float]) -> Dict[str, Any]:
        """Analyze a single opportunity.
        
        Args:
            opportunity: The opportunity to analyze
            company_profile: Profile of the company for matching
            weights: Weighting for different factors
            
        Returns:
            Analysis results for the opportunity
        """
        # Extract relevant opportunity data
        opp_id = opportunity.get("id") or opportunity.get("opportunity_id")
        opp_title = opportunity.get("title", "Unknown")
        opp_value = opportunity.get("value", 0)
        opp_agency = opportunity.get("agency_name") or opportunity.get("agency", "Unknown")
        opp_description = opportunity.get("description", "")
        
        # Extract entities if available
        tech_stack = []
        regulatory_requirements = []
        clearance_requirements = []
        
        if "entity_summary" in opportunity:
            summary = opportunity["entity_summary"]
            tech_stack = summary.get("tech_stack", [])
            regulatory_requirements = summary.get("regulatory_requirements", [])
            clearance_requirements = summary.get("clearance_requirements", [])
        
        # Technical fit score - match company capabilities with opportunity requirements
        technical_score = self._calculate_technical_fit(
            tech_stack, 
            regulatory_requirements,
            clearance_requirements,
            company_profile
        )
        
        # Past performance score - how well company has performed on similar contracts
        past_performance_score = self._calculate_past_performance(
            opp_agency,
            tech_stack,
            company_profile
        )
        
        # Competition score - assessment of competitive landscape
        competition_score = self._calculate_competition_score(
            opp_agency,
            tech_stack,
            opp_value,
            company_profile
        )
        
        # Financial score - financial attractiveness of the opportunity
        financial_score = self._calculate_financial_score(
            opp_value,
            company_profile
        )
        
        # Strategic score - alignment with company strategy
        strategic_score = self._calculate_strategic_score(
            opp_agency,
            tech_stack,
            company_profile
        )
        
        # Calculate overall score using weights
        overall_score = (
            weights.get("technical_fit", 0.3) * technical_score +
            weights.get("past_performance", 0.25) * past_performance_score +
            weights.get("competition", 0.15) * competition_score +
            weights.get("financial", 0.2) * financial_score +
            weights.get("strategic", 0.1) * strategic_score
        )
        
        # Generate analysis with scores and explanation
        analysis = {
            "opportunity_id": opp_id,
            "title": opp_title,
            "agency": opp_agency,
            "value": opp_value,
            "scores": {
                "technical_fit": technical_score,
                "past_performance": past_performance_score,
                "competition": competition_score,
                "financial": financial_score,
                "strategic": strategic_score
            },
            "overall_score": overall_score,
            "recommendation": "Bid" if overall_score >= 70 else "No Bid",
            "explanations": self._generate_explanations(
                technical_score, 
                past_performance_score,
                competition_score,
                financial_score,
                strategic_score,
                tech_stack,
                regulatory_requirements,
                clearance_requirements,
                company_profile
            ),
            "requirements": {
                "technologies": tech_stack,
                "regulations": regulatory_requirements,
                "clearances": clearance_requirements
            }
        }
        
        return analysis
    
    def _calculate_technical_fit(self, tech_stack: List[str], 
                               regulations: List[str],
                               clearances: List[str],
                               company_profile: Dict[str, Any]) -> float:
        """Calculate technical fit score.
        
        Args:
            tech_stack: Required technologies
            regulations: Required regulations
            clearances: Required clearances
            company_profile: Company profile for matching
            
        Returns:
            Technical fit score (0-100)
        """
        score = 0
        total_points = 0
        explanations = []
        
        # Technology match
        company_tech = company_profile.get("technologies", [])
        if tech_stack:
            total_points += 50
            matched_tech = [tech for tech in tech_stack if tech in company_tech]
            tech_score = (len(matched_tech) / len(tech_stack)) * 50
            score += tech_score
        
        # Regulation compliance
        company_regulations = company_profile.get("regulations", [])
        if regulations:
            total_points += 30
            matched_regs = [reg for reg in regulations if reg in company_regulations]
            reg_score = (len(matched_regs) / len(regulations)) * 30
            score += reg_score
        
        # Clearance level
        company_clearances = company_profile.get("clearances", [])
        if clearances:
            total_points += 20
            matched_clearances = [cl for cl in clearances if cl in company_clearances]
            clearance_score = (len(matched_clearances) / len(clearances)) * 20
            score += clearance_score
        
        # Normalize score if we had requirements
        final_score = score if total_points == 0 else (score / total_points) * 100
        
        return round(final_score, 1)
    
    def _calculate_past_performance(self, agency: str,
                                  tech_stack: List[str],
                                  company_profile: Dict[str, Any]) -> float:
        """Calculate past performance score.
        
        Args:
            agency: Target agency
            tech_stack: Required technologies
            company_profile: Company profile for matching
            
        Returns:
            Past performance score (0-100)
        """
        score = 50  # Default middle score
        
        # Check past performance with this agency
        past_contracts = company_profile.get("past_contracts", [])
        agency_contracts = [c for c in past_contracts if c.get("agency") == agency]
        
        if agency_contracts:
            # Calculate average performance rating for this agency
            ratings = [c.get("performance_rating", 0) for c in agency_contracts]
            avg_rating = sum(ratings) / len(ratings) if ratings else 0
            
            # Scale to 0-100
            agency_score = (avg_rating / 5) * 100 if avg_rating else 50
            
            # Weight agency-specific performance heavily
            score = agency_score
        else:
            # Check for contracts with similar technologies
            tech_contracts = []
            for contract in past_contracts:
                contract_tech = contract.get("technologies", [])
                if any(tech in tech_stack for tech in contract_tech):
                    tech_contracts.append(contract)
            
            if tech_contracts:
                # Calculate average performance rating for similar contracts
                ratings = [c.get("performance_rating", 0) for c in tech_contracts]
                avg_rating = sum(ratings) / len(ratings) if ratings else 0
                
                # Scale to 0-100
                tech_score = (avg_rating / 5) * 100 if avg_rating else 50
                
                # Use technology-related performance
                score = tech_score
        
        return round(score, 1)
    
    def _calculate_competition_score(self, agency: str,
                                   tech_stack: List[str],
                                   value: float,
                                   company_profile: Dict[str, Any]) -> float:
        """Calculate competition score.
        
        Args:
            agency: Target agency
            tech_stack: Required technologies
            value: Contract value
            company_profile: Company profile for matching
            
        Returns:
            Competition score (0-100)
        """
        # This is a simplified version - in reality would use data about
        # competitors and win rates in specific markets
        
        # Higher score means less competition or better competitive position
        
        # Check if company has agency-specific advantages
        agency_advantages = False
        for relationship in company_profile.get("agency_relationships", []):
            if relationship.get("agency") == agency and relationship.get("strength", 0) > 3:
                agency_advantages = True
                break
        
        # Check if company has technology-specific advantages
        tech_advantages = False
        for tech in tech_stack:
            if tech in company_profile.get("technology_strengths", []):
                tech_advantages = True
                break
        
        # Base score - higher for both advantages
        if agency_advantages and tech_advantages:
            score = 85
        elif agency_advantages:
            score = 75
        elif tech_advantages:
            score = 65
        else:
            score = 50
        
        # Adjust for contract size - larger contracts often have more competition
        if value > 10000000:  # $10M+
            score -= 15
        elif value > 5000000:  # $5M-$10M
            score -= 10
        elif value > 1000000:  # $1M-$5M
            score -= 5
        
        # Ensure score is in 0-100 range
        score = max(0, min(100, score))
        
        return round(score, 1)
    
    def _calculate_financial_score(self, value: float,
                                 company_profile: Dict[str, Any]) -> float:
        """Calculate financial score.
        
        Args:
            value: Contract value
            company_profile: Company profile for matching
            
        Returns:
            Financial score (0-100)
        """
        # Base score starts at 50
        score = 50
        
        # Get company financial thresholds
        min_contract = company_profile.get("min_contract_value", 100000)
        target_contract = company_profile.get("target_contract_value", 1000000)
        max_contract = company_profile.get("max_contract_value", 10000000)
        
        # Too small
        if value < min_contract:
            score = 30
        # Sweet spot
        elif min_contract <= value <= max_contract:
            # Calculate how close to target value (100 if exactly target)
            if value <= target_contract:
                ratio = (value - min_contract) / (target_contract - min_contract)
                score = 70 + (ratio * 30)
            else:
                ratio = 1 - ((value - target_contract) / (max_contract - target_contract))
                score = 70 + (ratio * 30)
        # Too large
        else:
            score = 40
        
        # Ensure score is in 0-100 range
        score = max(0, min(100, score))
        
        return round(score, 1)
    
    def _calculate_strategic_score(self, agency: str,
                                 tech_stack: List[str],
                                 company_profile: Dict[str, Any]) -> float:
        """Calculate strategic alignment score.
        
        Args:
            agency: Target agency
            tech_stack: Required technologies
            company_profile: Company profile for matching
            
        Returns:
            Strategic score (0-100)
        """
        score = 50  # Default middle score
        
        # Check if agency is a strategic target
        strategic_agencies = company_profile.get("strategic_agencies", [])
        if agency in strategic_agencies:
            score += 25
        
        # Check if technologies align with strategic direction
        strategic_technologies = company_profile.get("strategic_technologies", [])
        strategic_tech_count = sum(1 for tech in tech_stack if tech in strategic_technologies)
        
        if tech_stack:
            strategic_ratio = strategic_tech_count / len(tech_stack)
            score += strategic_ratio * 25
        
        # Ensure score is in 0-100 range
        score = max(0, min(100, score))
        
        return round(score, 1)
    
    def _generate_explanations(self, technical_score: float,
                             past_performance_score: float,
                             competition_score: float,
                             financial_score: float,
                             strategic_score: float,
                             tech_stack: List[str],
                             regulations: List[str],
                             clearances: List[str],
                             company_profile: Dict[str, Any]) -> Dict[str, str]:
        """Generate explanations for scores.
        
        Args:
            technical_score: Technical fit score
            past_performance_score: Past performance score
            competition_score: Competition score
            financial_score: Financial score
            strategic_score: Strategic score
            tech_stack: Required technologies
            regulations: Required regulations
            clearances: Required clearances
            company_profile: Company profile for matching
            
        Returns:
            Dictionary of explanations for each score
        """
        explanations = {}
        
        # Technical fit explanation
        if technical_score >= 80:
            explanations["technical_fit"] = "Strong technical alignment with our capabilities."
        elif technical_score >= 60:
            explanations["technical_fit"] = "Good technical alignment, but some capability gaps exist."
        elif technical_score >= 40:
            explanations["technical_fit"] = "Moderate technical alignment, significant capability gaps."
        else:
            explanations["technical_fit"] = "Poor technical alignment, major capability gaps."
        
        # Past performance explanation
        if past_performance_score >= 80:
            explanations["past_performance"] = "Strong past performance record with this agency/technology."
        elif past_performance_score >= 60:
            explanations["past_performance"] = "Good past performance, some relationship building needed."
        elif past_performance_score >= 40:
            explanations["past_performance"] = "Limited past performance with this agency/technology."
        else:
            explanations["past_performance"] = "No relevant past performance to leverage."
        
        # Competition explanation
        if competition_score >= 80:
            explanations["competition"] = "Favorable competitive landscape, limited competition expected."
        elif competition_score >= 60:
            explanations["competition"] = "Manageable competitive landscape, some strong competitors."
        elif competition_score >= 40:
            explanations["competition"] = "Challenging competitive landscape, several strong competitors."
        else:
            explanations["competition"] = "Highly competitive landscape, significant competition expected."
        
        # Financial explanation
        if financial_score >= 80:
            explanations["financial"] = "Excellent financial fit, aligns with target contract value."
        elif financial_score >= 60:
            explanations["financial"] = "Good financial fit, within preferred contract range."
        elif financial_score >= 40:
            explanations["financial"] = "Acceptable financial fit, but not ideal contract size."
        else:
            explanations["financial"] = "Poor financial fit, outside of preferred contract range."
        
        # Strategic explanation
        if strategic_score >= 80:
            explanations["strategic"] = "Strong strategic alignment with company growth objectives."
        elif strategic_score >= 60:
            explanations["strategic"] = "Good strategic alignment, supports some company objectives."
        elif strategic_score >= 40:
            explanations["strategic"] = "Limited strategic alignment, tangential to company objectives."
        else:
            explanations["strategic"] = "Poor strategic alignment, does not support company objectives."
        
        return explanations