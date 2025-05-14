# core/collectors/sam_gov.py
import aiohttp
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
import os
import json
import logging

from core.collectors.base_collector import BaseCollector

class SamGovCollector(BaseCollector):
    """Collector for SAM.gov opportunities"""
    
    # Rate limiting properties
    last_request_time = 0
    min_request_interval = 1.0  # Min 1 second between requests
    
    def __init__(self, source_name: str = "sam.gov", config: Dict[str, Any] = None):
        default_config = {
            'api_key': os.environ.get('SAM_GOV_API_KEY', ''),
            'base_url': 'https://api.sam.gov/opportunities/v1/search',
            'published_since': (datetime.now() - timedelta(days=30)).isoformat(),
            'limit': 100,
            'postedFrom': (datetime.now() - timedelta(days=30)).strftime("%m/%d/%Y"),
            'postedTo': datetime.now().strftime("%m/%d/%Y")
        }
        
        # Use provided config or default
        config = config or default_config
        super().__init__(source_name, config)
        
        # Check if API key is available
        if not self.config['api_key']:
            self.logger.warning("SAM.gov API key not provided. API calls will likely fail.")
    
    async def collect(self) -> List[Dict[str, Any]]:
        """Collect opportunities from SAM.gov"""
        self.logger.info(f"Starting collection from {self.source_name}")
        
        headers = {
            "X-Api-Key": self.config['api_key'],
            "Content-Type": "application/json"
        }
        
        # Parameters for the search
        params = {
            "limit": self.config['limit'],
            "api_key": self.config['api_key'],
            "postedFrom": self.config['postedFrom'],
            "postedTo": self.config['postedTo']
        }
        
        results = []
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.config['base_url'],
                    headers=headers,
                    params=params
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        opportunities = data.get('opportunitiesData', [])
                        self.logger.info(f"Retrieved {len(opportunities)} opportunities")
                        
                        # Fetch additional details for each opportunity
                        for opportunity in opportunities:
                            opp_id = opportunity.get('opportunityId')
                            if opp_id:
                                details = await self._fetch_opportunity_details(session, opp_id, headers)
                                if details:
                                    # Merge details with opportunity data
                                    opportunity.update(details)
                            
                            results.append(opportunity)
                    elif response.status == 401:
                        self.logger.error("Authentication failed: Invalid API key or expired credentials")
                        # Return mock data as fallback during development
                        return self._generate_mock_data()
                    elif response.status == 429:
                        self.logger.error("Rate limit exceeded. Please wait before making additional requests")
                        # Return mock data as fallback during development
                        return self._generate_mock_data()
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Error retrieving opportunities: {response.status}, {error_text}")
                        # Return mock data as fallback during development
                        return self._generate_mock_data()
                
                self.logger.info(f"Completed collection from {self.source_name}, found {len(results)} items")
                return results
        except Exception as e:
            self.logger.error(f"Error in SAM.gov API collection: {str(e)}")
            # Return mock data as fallback during development
            return self._generate_mock_data()
    
    async def _fetch_opportunity_details(self, session, opportunity_id, headers):
        """Fetch detailed information for a specific opportunity with rate limiting"""
        details_url = f"https://api.sam.gov/opportunities/v1/opportunity/{opportunity_id}"
        
        # Apply rate limiting
        current_time = time.time()
        time_since_last = current_time - self.__class__.last_request_time
        if time_since_last < self.__class__.min_request_interval:
            await asyncio.sleep(self.__class__.min_request_interval - time_since_last)
        
        # Update the last request time
        self.__class__.last_request_time = time.time()
        
        try:
            async with session.get(
                details_url,
                headers=headers
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    self.logger.warning(f"Failed to get details for opportunity {opportunity_id}: {response.status}")
                    return {}
        except Exception as e:
            self.logger.error(f"Error fetching opportunity details: {str(e)}")
            return {}
    
    def _generate_mock_data(self) -> List[Dict[str, Any]]:
        """Generate mock data when API calls fail"""
        self.logger.info("Generating mock data as fallback")
        
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
                    'name': item.get('agency', ''),
                    'id': item.get('agencyId', '')
                }
            
            if 'award' in item and item['award']:
                # Format award data
                award = item['award']
                item['award_data'] = {
                    'agency_id': item.get('agencyId', ''),
                    'agency_name': item.get('agency', ''),
                    'contractor_id': award.get('awardeeId', award.get('duns', '')),
                    'contractor_name': award.get('awardee', ''),
                    'opportunity_id': item.get('opportunityId', ''),
                    'title': item.get('title', ''),
                    'description': item.get('description', ''),
                    'value': award.get('amount', 0),
                    'award_date': award.get('date', '')
                }
            
            # Format dates consistently
            for date_field in ['postedDate', 'responseDeadLine', 'archiveDate']:
                if date_field in item and item[date_field]:
                    try:
                        # Try to parse and standardize the date format
                        date_obj = datetime.fromisoformat(item[date_field].replace('Z', '+00:00'))
                        item[date_field] = date_obj.isoformat()
                    except (ValueError, TypeError):
                        # Keep original if parsing fails
                        pass
        
        return processed