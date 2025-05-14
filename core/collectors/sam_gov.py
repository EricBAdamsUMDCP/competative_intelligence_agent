# core/collectors/sam_gov.py
import aiohttp
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any
import os
import json

from core.collectors.base_collector import BaseCollector

class SamGovCollector(BaseCollector):
    """Collector for SAM.gov opportunities"""
    
    def __init__(self, source_name: str = "sam.gov", config: Dict[str, Any] = None):
        default_config = {
            'api_key': os.environ.get('SAM_GOV_API_KEY', 'DEMO_KEY'),
            'base_url': 'https://api.sam.gov/opportunities/v1/search',
            'published_since': (datetime.now() - timedelta(days=1)).isoformat(),
            'limit': 100
        }
        
        # Use provided config or default
        config = config or default_config
        
        super().__init__(source_name, config)
    
    async def collect(self) -> List[Dict[str, Any]]:
        """Collect opportunities from SAM.gov"""
        # For now, we'll use a mock implementation since we don't have a real API key
        # In a real implementation, we would make API calls to SAM.gov
        
        # Simulate API call
        self.logger.info("Simulating SAM.gov API call (replace with actual API call in production)")
        
        # Mock data
        mock_data = [
            {
                'opportunityId': 'SAMGOV123456',
                'title': 'Cybersecurity Services for Department of Defense',
                'solicitationNumber': 'DOD-CYBER-2023-01',
                'agency': 'Department of Defense',
                'agencyId': 'DOD',
                'postedDate': (datetime.now() - timedelta(days=1)).isoformat(),
                'type': 'COMBINE',
                'baseType': 'PRESOL',
                'archiveType': 'AUTO15',
                'archiveDate': (datetime.now() + timedelta(days=30)).isoformat(),
                'setAside': 'SBA',
                'responseDeadLine': (datetime.now() + timedelta(days=14)).isoformat(),
                'naicsCode': '541512',
                'classificationCode': 'D',
                'active': 'Yes',
                'award': {
                    'awardee': 'TechDefense Solutions',
                    'awardeeId': 'TDS12345',
                    'amount': 5600000,
                    'date': (datetime.now() - timedelta(days=1)).isoformat()
                },
                'pointOfContact': [
                    {
                        'name': 'John Smith',
                        'email': 'john.smith@defense.gov',
                        'phone': '202-555-1234',
                        'type': 'PRIMARY'
                    }
                ],
                'description': 'The Department of Defense is seeking cybersecurity services to enhance the security posture of critical infrastructure. Services include vulnerability assessment, penetration testing, and security monitoring.',
                'links': [
                    {
                        'url': 'https://sam.gov/opp/123456',
                        'type': 'OPPORTUNITY'
                    }
                ]
            },
            {
                'opportunityId': 'SAMGOV789012',
                'title': 'Cloud Migration Services for Department of Health and Human Services',
                'solicitationNumber': 'HHS-CLOUD-2023-02',
                'agency': 'Department of Health and Human Services',
                'agencyId': 'HHS',
                'postedDate': (datetime.now() - timedelta(days=2)).isoformat(),
                'type': 'PRESOL',
                'baseType': 'PRESOL',
                'archiveType': 'AUTO15',
                'archiveDate': (datetime.now() + timedelta(days=30)).isoformat(),
                'setAside': 'N/A',
                'responseDeadLine': (datetime.now() + timedelta(days=21)).isoformat(),
                'naicsCode': '518210',
                'classificationCode': 'D',
                'active': 'Yes',
                'award': None,
                'pointOfContact': [
                    {
                        'name': 'Sarah Johnson',
                        'email': 'sarah.johnson@hhs.gov',
                        'phone': '202-555-5678',
                        'type': 'PRIMARY'
                    }
                ],
                'description': 'HHS is seeking services to migrate legacy applications to a cloud environment. The contractor will be responsible for assessment, planning, and execution of the migration, as well as ongoing support.',
                'links': [
                    {
                        'url': 'https://sam.gov/opp/789012',
                        'type': 'OPPORTUNITY'
                    }
                ]
            }
        ]
        
        # Artificial delay to simulate network request
        await asyncio.sleep(1)
        
        return mock_data
    
    def process_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process and normalize SAM.gov data"""
        processed = super().process_results(results)
        
        # Normalize fields to our internal format
        for item in processed:
            if 'opportunityId' in item:
                item['id'] = item['opportunityId']
            
            if 'agency' in item and 'agencyId' in item:
                item['agency_data'] = {
                    'name': item['agency'],
                    'id': item['agencyId']
                }
            
            if 'award' in item and item['award']:
                # Format award data
                award = item['award']
                item['award_data'] = {
                    'agency_id': item.get('agencyId'),
                    'agency_name': item.get('agency'),
                    'contractor_id': award.get('awardeeId'),
                    'contractor_name': award.get('awardee'),
                    'opportunity_id': item.get('opportunityId'),
                    'title': item.get('title'),
                    'description': item.get('description'),
                    'value': award.get('amount'),
                    'award_date': award.get('date')
                }
        
        return processed