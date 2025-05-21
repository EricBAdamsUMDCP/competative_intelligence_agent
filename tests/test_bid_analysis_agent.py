# tests/test_bid_analysis_agent.py
import unittest
import asyncio
import sys
import os
import json
from unittest.mock import patch, MagicMock

# Add project root to path to enable imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.agents.bid_analysis_agent import BidAnalysisAgent

# Sample test data
SAMPLE_COMPANY_PROFILE = {
    "name": "TechGov Solutions",
    "technologies": [
        "cloud", "cybersecurity", "artificial intelligence", 
        "machine learning", "zero trust", "DevSecOps"
    ],
    "regulations": [
        "NIST", "CMMC", "FedRAMP", "FISMA", "HIPAA"
    ],
    "clearances": [
        "Top Secret", "TS/SCI", "Secret"
    ],
    "past_contracts": [
        {
            "agency": "Department of Defense",
            "title": "Cloud Migration Services",
            "value": 5600000,
            "performance_rating": 4.8,
            "technologies": ["cloud", "DevSecOps", "cybersecurity"]
        },
        {
            "agency": "Department of Health and Human Services",
            "title": "AI-Driven Health Data Analytics",
            "value": 3200000,
            "performance_rating": 4.5,
            "technologies": ["artificial intelligence", "machine learning"]
        }
    ],
    "agency_relationships": [
        {
            "agency": "Department of Defense",
            "strength": 4,
            "contacts": 5
        }
    ],
    "technology_strengths": [
        "cybersecurity", "zero trust", "cloud"
    ],
    "min_contract_value": 500000,
    "target_contract_value": 5000000,
    "max_contract_value": 15000000,
    "strategic_agencies": [
        "Department of Defense", "Department of Homeland Security"
    ],
    "strategic_technologies": [
        "zero trust", "artificial intelligence", "DevSecOps"
    ]
}

SAMPLE_OPPORTUNITY = {
    "id": "opp123",
    "title": "Cybersecurity Services for DoD",
    "agency": "Department of Defense",
    "value": 6000000,
    "description": "Provide cybersecurity services including zero trust implementation and cloud security for the Department of Defense.",
    "entity_summary": {
        "tech_stack": ["cybersecurity", "zero trust", "cloud"],
        "regulatory_requirements": ["NIST", "CMMC"],
        "clearance_requirements": ["Top Secret"],
        "agencies_involved": ["Department of Defense"]
    }
}

GOOD_OPPORTUNITY = {
    "id": "opp_good",
    "title": "AI Implementation for DoD",
    "agency": "Department of Defense",
    "value": 5000000,  # Exactly the target value
    "description": "Implement AI and machine learning solutions for the Department of Defense.",
    "entity_summary": {
        "tech_stack": ["artificial intelligence", "machine learning"],
        "regulatory_requirements": ["NIST", "CMMC"],
        "clearance_requirements": ["Secret"],
        "agencies_involved": ["Department of Defense"]
    }
}

BAD_OPPORTUNITY = {
    "id": "opp_bad",
    "title": "Blockchain Implementation for Treasury",
    "agency": "Department of Treasury",
    "value": 20000000,  # Above maximum value
    "description": "Implement blockchain solutions for the Department of Treasury.",
    "entity_summary": {
        "tech_stack": ["blockchain", "distributed ledger"],
        "regulatory_requirements": ["SOX", "GLBA"],
        "clearance_requirements": ["Public Trust"],
        "agencies_involved": ["Department of Treasury"]
    }
}

class TestBidAnalysisAgent(unittest.TestCase):
    """Tests for the BidAnalysisAgent class."""
    
    def setUp(self):
        self.agent = BidAnalysisAgent(
            agent_id="bid_analysis_agent",
            name="Bid Analysis Agent",
            company_profile=SAMPLE_COMPANY_PROFILE
        )
    
    def test_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.agent_id, "bid_analysis_agent")
        self.assertEqual(self.agent.name, "Bid Analysis Agent")
        self.assertEqual(self.agent.config["company_profile"], SAMPLE_COMPANY_PROFILE)
        self.assertIn("weights", self.agent.config)
    
    def test_analyze_opportunity(self):
        """Test analyzing a single opportunity."""
        # Run the agent
        result = asyncio.run(self.agent.execute_task({
            "opportunities": [SAMPLE_OPPORTUNITY]
        }))
        
        # Check the result structure
        self.assertEqual(result["opportunity_count"], 1)
        self.assertEqual(len(result["analyzed_opportunities"]), 1)
        
        # Get the analysis
        analysis = result["analyzed_opportunities"][0]
        
        # Check basic fields
        self.assertEqual(analysis["opportunity_id"], "opp123")
        self.assertEqual(analysis["title"], "Cybersecurity Services for DoD")
        self.assertEqual(analysis["agency"], "Department of Defense")
        self.assertEqual(analysis["value"], 6000000)
        
        # Check scores
        self.assertIn("scores", analysis)
        scores = analysis["scores"]
        self.assertIn("technical_fit", scores)
        self.assertIn("past_performance", scores)
        self.assertIn("competition", scores)
        self.assertIn("financial", scores)
        self.assertIn("strategic", scores)
        
        # Check overall score and recommendation
        self.assertIn("overall_score", analysis)
        self.assertIn("recommendation", analysis)
        
        # Check explanations
        self.assertIn("explanations", analysis)
        explanations = analysis["explanations"]
        self.assertIn("technical_fit", explanations)
        self.assertIn("past_performance", explanations)
        self.assertIn("competition", explanations)
        self.assertIn("financial", explanations)
        self.assertIn("strategic", explanations)
        
        # Check requirements
        self.assertIn("requirements", analysis)
        requirements = analysis["requirements"]
        self.assertEqual(requirements["technologies"], ["cybersecurity", "zero trust", "cloud"])
        self.assertEqual(requirements["regulations"], ["NIST", "CMMC"])
        self.assertEqual(requirements["clearances"], ["Top Secret"])
    
    def test_multiple_opportunities(self):
        """Test analyzing multiple opportunities."""
        # Run the agent
        result = asyncio.run(self.agent.execute_task({
            "opportunities": [GOOD_OPPORTUNITY, BAD_OPPORTUNITY]
        }))
        
        # Check the result structure
        self.assertEqual(result["opportunity_count"], 2)
        self.assertEqual(len(result["analyzed_opportunities"]), 2)
        
        # Results should be sorted by overall score (highest first)
        self.assertTrue(
            result["analyzed_opportunities"][0]["overall_score"] >= 
            result["analyzed_opportunities"][1]["overall_score"]
        )
        
        # Check recommendations
        good_opp = None
        bad_opp = None
        for opp in result["analyzed_opportunities"]:
            if opp["opportunity_id"] == "opp_good":
                good_opp = opp
            elif opp["opportunity_id"] == "opp_bad":
                bad_opp = opp
        
        self.assertIsNotNone(good_opp)
        self.assertIsNotNone(bad_opp)
        
        # The good opportunity should have a higher score
        self.assertTrue(good_opp["overall_score"] > bad_opp["overall_score"])
        
        # Check recommended opportunities
        recommended = result["recommended_opportunities"]
        recommended_ids = [opp["opportunity_id"] for opp in recommended]
        
        # Good opportunity should be recommended if score is high enough
        if good_opp["overall_score"] >= 70:
            self.assertIn("opp_good", recommended_ids)
        
        # Bad opportunity should not be recommended
        self.assertNotIn("opp_bad", recommended_ids)
    
    def test_technical_fit_calculation(self):
        """Test the technical fit score calculation."""
        # Call the internal method directly
        tech_score = self.agent._calculate_technical_fit(
            tech_stack=["cybersecurity", "zero trust", "cloud"],
            regulations=["NIST", "CMMC"],
            clearances=["Top Secret"],
            company_profile=SAMPLE_COMPANY_PROFILE
        )
        
        # All the technologies, regulations, and clearances are in the company profile,
        # so this should have a high score
        self.assertGreaterEqual(tech_score, 90)  # Score should be very high
        
        # Test with some matches and some mismatches
        partial_score = self.agent._calculate_technical_fit(
            tech_stack=["cybersecurity", "blockchain"],  # One match, one mismatch
            regulations=["NIST", "SOX"],  # One match, one mismatch
            clearances=["Secret"],  # Match
            company_profile=SAMPLE_COMPANY_PROFILE
        )
        
        # Should have a moderate score
        self.assertGreater(partial_score, 40)
        self.assertLess(partial_score, 90)
        
        # Test with no matches
        no_match_score = self.agent._calculate_technical_fit(
            tech_stack=["blockchain", "quantum"],
            regulations=["SOX", "GLBA"],
            clearances=["Public Trust"],
            company_profile=SAMPLE_COMPANY_PROFILE
        )
        
        # Should have a low score
        self.assertLess(no_match_score, 40)
    
    def test_past_performance_calculation(self):
        """Test the past performance score calculation."""
        # Call the internal method directly
        
        # Test with an agency where we have past performance
        dod_score = self.agent._calculate_past_performance(
            agency="Department of Defense",
            tech_stack=["cybersecurity", "cloud"],
            company_profile=SAMPLE_COMPANY_PROFILE
        )
        
        # Should have a high score
        self.assertGreaterEqual(dod_score, 70)
        
        # Test with an agency where we don't have past performance
        unknown_score = self.agent._calculate_past_performance(
            agency="Department of Energy",
            tech_stack=["cybersecurity", "cloud"],
            company_profile=SAMPLE_COMPANY_PROFILE
        )
        
        # Should have a moderate score based on similar technologies
        self.assertGreater(unknown_score, 40)
        self.assertLess(unknown_score, 70)
    
    def test_financial_score_calculation(self):
        """Test the financial score calculation."""
        # Call the internal method directly
        
        # Test with the target value
        target_score = self.agent._calculate_financial_score(
            value=5000000,  # Exactly the target value
            company_profile=SAMPLE_COMPANY_PROFILE
        )
        
        # Should have a high score
        self.assertGreaterEqual(target_score, 90)
        
        # Test with a value within range but not at target
        within_range_score = self.agent._calculate_financial_score(
            value=10000000,  # Within range but not at target
            company_profile=SAMPLE_COMPANY_PROFILE
        )
        
        # Should have a good score
        self.assertGreater(within_range_score, 70)
        
        # Test with a value outside the range
        outside_range_score = self.agent._calculate_financial_score(
            value=20000000,  # Above maximum
            company_profile=SAMPLE_COMPANY_PROFILE
        )
        
        # Should have a lower score
        self.assertLess(outside_range_score, 50)
    
    def test_strategic_score_calculation(self):
        """Test the strategic score calculation."""
        # Call the internal method directly
        
        # Test with a strategic agency and strategic technologies
        strategic_score = self.agent._calculate_strategic_score(
            agency="Department of Defense",
            tech_stack=["artificial intelligence", "zero trust"],
            company_profile=SAMPLE_COMPANY_PROFILE
        )
        
        # Should have a high score
        self.assertGreaterEqual(strategic_score, 80)
        
        # Test with a non-strategic agency but strategic technologies
        partial_score = self.agent._calculate_strategic_score(
            agency="Department of Energy",
            tech_stack=["artificial intelligence", "zero trust"],
            company_profile=SAMPLE_COMPANY_PROFILE
        )
        
        # Should have a moderate score
        self.assertGreater(partial_score, 50)
        self.assertLess(partial_score, 80)
        
        # Test with a non-strategic agency and non-strategic technologies
        non_strategic_score = self.agent._calculate_strategic_score(
            agency="Department of Energy",
            tech_stack=["blockchain", "quantum"],
            company_profile=SAMPLE_COMPANY_PROFILE
        )
        
        # Should have a low score
        self.assertLess(non_strategic_score, 50)
    
    def test_custom_weights(self):
        """Test using custom weights for scoring."""
        # Create an agent with custom weights
        custom_agent = BidAnalysisAgent(
            agent_id="custom_weights_agent",
            name="Custom Weights Agent",
            company_profile=SAMPLE_COMPANY_PROFILE,
            config={
                "weights": {
                    "technical_fit": 0.5,  # Higher weight on technical fit
                    "past_performance": 0.2,
                    "competition": 0.1,
                    "financial": 0.1,
                    "strategic": 0.1
                }
            }
        )
        
        # Run both agents on the same opportunity
        standard_result = asyncio.run(self.agent.execute_task({
            "opportunities": [SAMPLE_OPPORTUNITY]
        }))
        
        custom_result = asyncio.run(custom_agent.execute_task({
            "opportunities": [SAMPLE_OPPORTUNITY]
        }))
        
        # Get the analyses
        standard_analysis = standard_result["analyzed_opportunities"][0]
        custom_analysis = custom_result["analyzed_opportunities"][0]
        
        # The scores for individual factors should be the same
        self.assertEqual(
            standard_analysis["scores"]["technical_fit"],
            custom_analysis["scores"]["technical_fit"]
        )
        
        # But the overall scores should be different due to different weights
        self.assertNotEqual(
            standard_analysis["overall_score"],
            custom_analysis["overall_score"]
        )
        
        # The custom agent puts more weight on technical fit, so if technical fit is high,
        # the custom agent should have a higher overall score
        if custom_analysis["scores"]["technical_fit"] > 70:
            self.assertGreater(
                custom_analysis["overall_score"],
                standard_analysis["overall_score"]
            )

if __name__ == "__main__":
    unittest.main()