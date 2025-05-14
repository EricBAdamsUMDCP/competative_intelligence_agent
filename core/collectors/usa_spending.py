# core/collectors/usa_spending.py
import aiohttp
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
import os
import json

from core.collectors.base_collector import BaseCollector

class USASpendingCollector(BaseCollector):
    """Collector for USASpending.gov data"""
    
    # Rate limiting properties
    last_request_time = 0
    min_request_interval = 1.0  # Min 1 second between requests
    
    def __init__(self, source_name: str = "usaspending.gov", config: Dict[str, Any] = None):
        default_config = {
            'base_url': 'https://api.usaspending.gov/api/v2',
            'award_search_url': '/search/spending_by_award/',
            'award_detail_url': '/award/v1/awards/',
            'recipient_url': '/recipient/',
            'agency_url': '/agency/',
            'date_range': {
                'start_date': (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d"),
                'end_date': datetime.now().strftime("%Y-%m-%d")
            },
            'limit': 100,
            'award_types': ['contracts']
        }
        
        # Use provided config or default
        config = config or default_config
        super().__init__(source_name, config)
    
    async def collect(self) -> List[Dict[str, Any]]:
        """Collect contract award data from USASpending.gov"""
        self.logger.info(f"Starting collection from {self.source_name}")
        
        headers = {
            "Content-Type": "application/json"
        }
        
        # Request payload for award search
        payload = {
            "filters": {
                "time_period": [
                    {
                        "start_date": self.config['date_range']['start_date'],
                        "end_date": self.config['date_range']['end_date']
                    }
                ],
                "award_type_codes": self.config['award_types']
            },
            "fields": [
                "Award ID",
                "Recipient Name",
                "Description",
                "Action Date",
                "Amount",
                "Awarding Agency",
                "Awarding Sub Agency",
                "Contract Award Type",
                "recipient_id",
                "awarding_agency_id"
            ],
            "page": 1,
            "limit": self.config['limit'],
            "sort": "Action Date",
            "order": "desc"
        }
        
        results = []
        
        try:
            async with aiohttp.ClientSession() as session:
                # Apply rate limiting
                current_time = time.time()
                time_since_last = current_time - self.__class__.last_request_time
                if time_since_last < self.__class__.min_request_interval:
                    await asyncio.sleep(self.__class__.min_request_interval - time_since_last)
                
                # Update the last request time
                self.__class__.last_request_time = time.time()
                
                async with session.post(
                    self.config['base_url'] + self.config['award_search_url'],
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        awards = data.get('results', [])
                        self.logger.info(f"Retrieved {len(awards)} awards")
                        
                        # Fetch additional details for each award
                        for award in awards:
                            award_id = award.get('Award ID')
                            if award_id:
                                details = await self._fetch_award_details(session, award_id, headers)
                                if details:
                                    # Merge details with award data
                                    award.update(details)
                            
                            results.append(award)
                    elif response.status == 429:
                        self.logger.error("Rate limit exceeded. Please wait before making additional requests")
                        # Return mock data as fallback during development
                        return self._generate_mock_data()
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Error retrieving awards: {response.status}, {error_text}")
                        # Return mock data as fallback during development
                        return self._generate_mock_data()
                
                self.logger.info(f"Completed collection from {self.source_name}, found {len(results)} items")
                return results
        except Exception as e:
            self.logger.error(f"Error in USASpending.gov API collection: {str(e)}")
            # Return mock data as fallback during development
            return self._generate_mock_data()
    
    async def _fetch_award_details(self, session, award_id, headers):
        """Fetch detailed information for a specific award with rate limiting"""
        details_url = f"{self.config['base_url']}{self.config['award_detail_url']}{award_id}/"
        
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
                    self.logger.warning(f"Failed to get details for award {award_id}: {response.status}")
                    return {}
        except Exception as e:
            self.logger.error(f"Error fetching award details: {str(e)}")
            return {}
    
    def _generate_mock_data(self) -> List[Dict[str, Any]]:
        """Generate mock data when API calls fail"""
        self.logger.info("Generating mock data as fallback")
        
        # Mock data
        mock_data = [
            {
                "Award ID": "CONT2023001",
                "Recipient Name": "TechDefense Solutions",
                "Description": "Cybersecurity services for Department of Defense infrastructure",
                "Action Date": datetime.now().strftime("%Y-%m-%d"),
                "Amount": 4200000.00,
                "Awarding Agency": "Department of Defense",
                "Awarding Sub Agency": "Defense Information Systems Agency",
                "Contract Award Type": "Firm Fixed Price",
                "recipient_id": "TDS12345",
                "awarding_agency_id": "DOD"
            },
            {
                "Award ID": "CONT2023002",
                "Recipient Name": "CloudSys Inc",
                "Description": "Cloud migration services for HHS data systems",
                "Action Date": (datetime.now() - timedelta(days=15)).strftime("%Y-%m-%d"),
                "Amount": 2800000.00,
                "Awarding Agency": "Department of Health and Human Services",
                "Awarding Sub Agency": "Office of the Secretary",
                "Contract Award Type": "Time and Materials",
                "recipient_id": "CSI67890",
                "awarding_agency_id": "HHS"
            },
            {
                "Award ID": "CONT2023003",
                "Recipient Name": "Federal IT Partners",
                "Description": "IT infrastructure support for DHS facilities",
                "Action Date": (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
                "Amount": 1500000.00,
                "Awarding Agency": "Department of Homeland Security",
                "Awarding Sub Agency": "Customs and Border Protection",
                "Contract Award Type": "Firm Fixed Price",
                "recipient_id": "FIP54321",
                "awarding_agency_id": "DHS"
            }
        ]
        
        return mock_data
    
    def process_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process and normalize USASpending.gov data"""
        processed = super().process_results(results)
        
        # Normalize fields to our internal format
        for item in processed:
            if 'Award ID' in item:
                item['id'] = item['Award ID']
                
            if 'Awarding Agency' in item:
                item['agency_data'] = {
                    'name': item.get('Awarding Agency', ''),
                    'id': item.get('awarding_agency_id', '')
                }
            
            # Format award data
            item['award_data'] = {
                'agency_id': item.get('awarding_agency_id', ''),
                'agency_name': item.get('Awarding Agency', ''),
                'contractor_id': item.get('recipient_id', ''),
                'contractor_name': item.get('Recipient Name', ''),
                'opportunity_id': item.get('Award ID', ''),
                'title': item.get('Contract Award Type', ''),
                'description': item.get('Description', ''),
                'value': float(item.get('Amount', 0)),
                'award_date': item.get('Action Date', '')
            }
            
            # Format dates consistently
            if 'Action Date' in item and item['Action Date']:
                try:
                    # Convert date format if needed
                    date_obj = datetime.strptime(item['Action Date'], "%Y-%m-%d")
                    item['Action Date'] = date_obj.isoformat()
                except (ValueError, TypeError):
                    # Keep original if parsing fails
                    pass
        
        return processed