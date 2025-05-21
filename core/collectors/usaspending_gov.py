# core/collectors/usaspending_gov.py
import aiohttp
import asyncio
from datetime import datetime, timedelta
import json
import logging
import os
import time
from typing import Dict, List, Any, Optional
import backoff
import calendar

from core.collectors.base_collector import BaseCollector

class USASpendingCollector(BaseCollector):
    """Collector for USASpending.gov contract data.
    
    This collector connects to the USASpending.gov API to retrieve federal contract spending data,
    including detailed transaction information, agency spending patterns, and contractor details.
    
    Attributes:
        source_name: Name of the data source (default: "usaspending.gov")
        config: Configuration dictionary for the collector
    """
    
    def __init__(self, source_name: str = "usaspending.gov", config: Dict[str, Any] = None):
        """Initialize the USASpending.gov collector.
        
        Args:
            source_name: Name of the data source
            config: Configuration dictionary with API settings
        """
        # Calculate default date range (last 90 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        
        default_config = {
            'base_url': 'https://api.usaspending.gov',
            'award_search_endpoint': '/api/v2/search/spending_by_award/',
            'award_endpoint': '/api/v2/awards/',
            'agency_endpoint': '/api/v2/agency/',
            'federal_accounts_endpoint': '/api/v2/federal_accounts/',
            'recipient_endpoint': '/api/v2/recipient/',
            'limit': 100,
            'page': 1,
            'time_period': {
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d')
            },
            'api_timeout': 60,  # Timeout in seconds
            'rate_limit': {
                'max_requests': 5,   # Requests per time_period
                'time_period': 1     # Time period in seconds
            },
            'retry_config': {
                'max_tries': 5,
                'max_time': 300,      # Max retry time in seconds
                'initial_wait': 2,    # Initial wait in seconds
                'max_wait': 60        # Max wait between retries
            },
            'award_filters': {
                'award_type_codes': ['A', 'B', 'C', 'D']  # Procurement awards/contracts
            }
        }
        
        # Use provided config or default
        self.config = default_config
        if config:
            # Deep merge time_period and award_filters
            if 'time_period' in config:
                self.config['time_period'].update(config.pop('time_period', {}))
            if 'award_filters' in config:
                self.config['award_filters'].update(config.pop('award_filters', {}))
            # Update rest of config
            self.config.update(config)
            
        super().__init__(source_name, self.config)
        
        # Set up rate limiting
        self.request_timestamps = []
    
    async def _enforce_rate_limit(self):
        """Enforce rate limiting based on configuration.
        
        This method tracks request timestamps and delays if necessary to stay within
        the configured rate limits.
        """
        current_time = time.time()
        time_window_start = current_time - self.config['rate_limit']['time_period']
        
        # Remove timestamps outside the current time window
        self.request_timestamps = [ts for ts in self.request_timestamps if ts > time_window_start]
        
        # Check if we're at the rate limit
        if len(self.request_timestamps) >= self.config['rate_limit']['max_requests']:
            # Calculate delay needed (time until oldest request exits the window + small buffer)
            oldest_timestamp = min(self.request_timestamps)
            delay_seconds = oldest_timestamp - time_window_start + 0.1
            
            self.logger.info(f"Rate limit reached, waiting {delay_seconds:.2f} seconds")
            await asyncio.sleep(delay_seconds)
            
            # Recursively check again after waiting
            await self._enforce_rate_limit()
        
        # Add current request to timestamps
        self.request_timestamps.append(current_time)
    
    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=lambda self: self.config['retry_config']['max_tries'],
        max_time=lambda self: self.config['retry_config']['max_time'],
        factor=lambda self: self.config['retry_config']['initial_wait'],
        max_value=lambda self: self.config['retry_config']['max_wait']
    )
    async def _make_api_request(self, endpoint: str, method: str = 'GET', data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make a request to the USASpending.gov API.
        
        Args:
            endpoint: API endpoint to call
            method: HTTP method (GET or POST)
            data: Request payload for POST requests
            
        Returns:
            JSON response from the API
            
        Raises:
            Exception: If the API request fails
        """
        # Enforce rate limiting
        await self._enforce_rate_limit()
        
        # Prepare request
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        full_url = endpoint if endpoint.startswith('http') else f"{self.config['base_url']}{endpoint}"
        
        try:
            self.logger.info(f"Making {method} API request to {full_url}")
            
            async with aiohttp.ClientSession() as session:
                if method.upper() == 'GET':
                    request_func = session.get
                    kwargs = {'params': data} if data else {}
                else:  # POST
                    request_func = session.post
                    kwargs = {'json': data} if data else {}
                
                async with request_func(
                    full_url,
                    headers=headers,
                    timeout=self.config['api_timeout'],
                    **kwargs
                ) as response:
                    response_text = await response.text()
                    
                    # Handle API errors
                    if response.status != 200:
                        self.logger.error(f"API request failed: {response.status} - {response_text}")
                        raise Exception(f"USASpending.gov API request failed: {response.status}")
                    
                    try:
                        data = json.loads(response_text)
                        return data
                    except json.JSONDecodeError:
                        self.logger.error(f"Failed to parse API response as JSON: {response_text[:1000]}...")
                        raise Exception("Invalid JSON response from USASpending.gov API")
                        
        except aiohttp.ClientError as e:
            self.logger.error(f"API request error: {str(e)}")
            raise
    
    async def collect(self) -> List[Dict[str, Any]]:
        """Collect contract spending data from USASpending.gov API.
        
        Returns:
            List of contract data dictionaries
            
        Raises:
            Exception: If data collection fails
        """
        self.logger.info("Starting USASpending.gov data collection")
        
        # Create search payload
        search_payload = {
            "filters": {
                "time_period": [
                    {
                        "start_date": self.config['time_period']['start_date'],
                        "end_date": self.config['time_period']['end_date']
                    }
                ],
                "award_type_codes": self.config['award_filters']['award_type_codes']
            },
            "fields": [
                "Award ID",
                "Recipient Name",
                "Recipient DUNS Number",
                "Recipient UEI",
                "Award Amount",
                "Description",
                "Award Type",
                "Awarding Agency",
                "Awarding Sub Agency",
                "Award Date",
                "Period of Performance Start Date",
                "Period of Performance Current End Date",
                "Place of Performance City Code",
                "Place of Performance State Code",
                "Place of Performance Zip Code",
                "Place of Performance Country Code",
                "NAICS Code",
                "PSC Code"
            ],
            "page": self.config['page'],
            "limit": self.config['limit'],
            "sort": "Award Amount",
            "order": "desc",
            "subawards": False
        }
        
        # If additional filters are specified, add them
        if 'naics_codes' in self.config['award_filters']:
            search_payload['filters']['naics_codes'] = self.config['award_filters']['naics_codes']
        
        if 'psc_codes' in self.config['award_filters']:
            search_payload['filters']['psc_codes'] = self.config['award_filters']['psc_codes']
        
        if 'awarding_agency' in self.config['award_filters']:
            search_payload['filters']['awarding_agency'] = self.config['award_filters']['awarding_agency']
        
        all_results = []
        page = 1
        has_next = True
        
        while has_next:
            search_payload['page'] = page
            
            try:
                # Make search request
                data = await self._make_api_request(
                    self.config['award_search_endpoint'],
                    method='POST',
                    data=search_payload
                )
                
                # Extract results
                results = data.get('results', [])
                all_results.extend(results)
                
                self.logger.info(f"Retrieved {len(results)} contracts (page {page})")
                
                # Check if there are more pages
                has_next = data.get('page_metadata', {}).get('has_next_page', False)
                page += 1
                
                # Add brief delay between requests
                await asyncio.sleep(0.5)
                
            except Exception as e:
                self.logger.error(f"Error collecting data from USASpending.gov: {str(e)}")
                if page > 1:
                    # If we have some results, continue with partial data
                    self.logger.info(f"Continuing with {len(all_results)} results collected so far")
                    break
                else:
                    # If first page fails, we have no data
                    raise
        
        # For each award, get detailed data
        enriched_results = []
        for result in all_results[:min(len(all_results), 50)]:  # Limit detailed lookups to 50 awards
            try:
                # Extract award ID from generated_internal_id or Award ID field
                award_id = result.get('generated_internal_id', result.get('Award ID', ''))
                
                if not award_id:
                    self.logger.warning(f"Missing award ID for result: {result}")
                    enriched_results.append(result)
                    continue
                
                # Get detailed award data
                detail_data = await self._make_api_request(
                    f"{self.config['award_endpoint']}{award_id}/"
                )
                
                # Merge detail data with search result
                merged_data = {**result, 'award_detail': detail_data}
                enriched_results.append(merged_data)
                
                self.logger.info(f"Retrieved detailed data for award {award_id}")
                
                # Add brief delay between requests
                await asyncio.sleep(0.5)
                
            except Exception as e:
                self.logger.error(f"Error fetching award details: {str(e)}")
                # Continue with next award
                enriched_results.append(result)
        
        self.logger.info(f"Completed USASpending.gov data collection with {len(enriched_results)} items")
        return enriched_results
    
    def process_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process and normalize USASpending.gov data.
        
        Args:
            results: Raw data from USASpending.gov API
            
        Returns:
            Normalized list of contract dictionaries
        """
        processed = super().process_results(results)
        
        # Normalize fields to our internal format
        for item in processed:
            # Create a standardized contract data structure
            contract_data = {
                'id': self._extract_field(item, ['generated_internal_id', 'Award ID']),
                'title': self._extract_field(item, ['award_detail.description', 'Description']),
                'award_date': self._format_date(self._extract_field(item, ['award_detail.period_of_performance.action_date', 'Award Date'])),
                'value': self._extract_numeric(item, ['award_detail.total_obligation', 'Award Amount']),
                'recipient': {
                    'name': self._extract_field(item, ['award_detail.recipient.recipient_name', 'Recipient Name']),
                    'duns': self._extract_field(item, ['award_detail.recipient.duns', 'Recipient DUNS Number']),
                    'uei': self._extract_field(item, ['award_detail.recipient.uei', 'Recipient UEI']),
                    'address': self._extract_address(item),
                    'business_types': self._extract_field(item, ['award_detail.recipient.business_types_description'], [])
                },
                'awarding_agency': {
                    'id': self._extract_field(item, ['award_detail.awarding_agency.id']),
                    'name': self._extract_field(item, ['award_detail.awarding_agency.name', 'Awarding Agency']),
                    'abbreviation': self._extract_field(item, ['award_detail.awarding_agency.abbreviation']),
                    'subtier_name': self._extract_field(item, ['award_detail.awarding_agency.subtier_agency.name', 'Awarding Sub Agency'])
                },
                'period_of_performance': {
                    'start_date': self._format_date(self._extract_field(item, ['award_detail.period_of_performance.start_date', 'Period of Performance Start Date'])),
                    'end_date': self._format_date(self._extract_field(item, ['award_detail.period_of_performance.end_date', 'Period of Performance Current End Date'])),
                    'potential_end_date': self._format_date(self._extract_field(item, ['award_detail.period_of_performance.potential_end_date']))
                },
                'place_of_performance': self._extract_place_of_performance(item),
                'naics': {
                    'code': self._extract_field(item, ['award_detail.latest_transaction.contract_data.naics', 'NAICS Code']),
                    'description': self._extract_field(item, ['award_detail.latest_transaction.contract_data.naics_description'])
                },
                'psc': {
                    'code': self._extract_field(item, ['award_detail.latest_transaction.contract_data.product_or_service_code', 'PSC Code']),
                    'description': self._extract_field(item, ['award_detail.latest_transaction.contract_data.product_or_service_description'])
                },
                'type': self._extract_field(item, ['award_detail.type', 'Award Type']),
                'type_description': self._extract_field(item, ['award_detail.type_description']),
                'subaward_count': self._extract_numeric(item, ['award_detail.subaward_count']),
                'total_subaward_amount': self._extract_numeric(item, ['award_detail.total_subaward_amount']),
                'executive_details': self._extract_executive_details(item)
            }
            
            # Format for our knowledge graph - this is what other collectors also produce
            item['award_data'] = {
                'agency_id': contract_data['awarding_agency']['id'],
                'agency_name': contract_data['awarding_agency']['name'],
                'contractor_id': contract_data['recipient']['uei'] or contract_data['recipient']['duns'],
                'contractor_name': contract_data['recipient']['name'],
                'opportunity_id': contract_data['id'],
                'title': contract_data['title'],
                'description': contract_data['title'],  # Often the same in USASpending data
                'value': contract_data['value'],
                'award_date': contract_data['award_date'],
                'naics_code': contract_data['naics']['code'],
                'contract_number': contract_data['id'],
                'period_of_performance': (
                    f"{contract_data['period_of_performance']['start_date']} to "
                    f"{contract_data['period_of_performance']['end_date']}"
                ),
                'place_of_performance': self._format_place_of_performance(contract_data['place_of_performance']),
                'contract_type': contract_data['type_description']
            }
            
            # Add the full normalized contract data
            item['contract_data'] = contract_data
        
        return processed
    
    def _extract_field(self, data: Dict[str, Any], paths: List[str], default=None) -> Any:
        """Extract a field from a nested dictionary using a list of possible paths.
        
        Args:
            data: The dictionary to extract from
            paths: List of dot-separated paths to try
            default: Default value if not found
            
        Returns:
            The extracted value or default if not found
        """
        for path in paths:
            parts = path.split('.')
            value = data
            
            try:
                for part in parts:
                    if isinstance(value, dict) and part in value:
                        value = value[part]
                    else:
                        value = None
                        break
                
                if value is not None:
                    return value
            except (KeyError, TypeError):
                continue
        
        return default
    
    def _extract_numeric(self, data: Dict[str, Any], paths: List[str], default=0) -> float:
        """Extract a numeric field and convert to float.
        
        Args:
            data: The dictionary to extract from
            paths: List of dot-separated paths to try
            default: Default value if not found
            
        Returns:
            Extracted value as float or default
        """
        value = self._extract_field(data, paths, default)
        
        if value is not None:
            try:
                # Handle string amounts with currency symbols or commas
                if isinstance(value, str):
                    # Remove currency symbols, commas, and spaces
                    clean_value = value.replace('$', '').replace(',', '').replace(' ', '')
                    return float(clean_value)
                return float(value)
            except (ValueError, TypeError):
                self.logger.warning(f"Failed to convert to numeric: {value}")
        
        return default
    
    def _format_date(self, date_str: Optional[str]) -> str:
        """Format a date string to ISO format.
        
        Args:
            date_str: Date string to format
            
        Returns:
            ISO formatted date or empty string if invalid
        """
        if not date_str:
            return ''
            
        # Try different date formats
        date_formats = [
            '%Y-%m-%d',            # 2023-01-15
            '%m/%d/%Y',            # 01/15/2023
            '%Y-%m-%dT%H:%M:%S',   # 2023-01-15T00:00:00
            '%Y-%m-%dT%H:%M:%S.%f', # 2023-01-15T00:00:00.000
            '%Y/%m/%d',            # 2023/01/15
            '%b %d, %Y'            # Jan 15, 2023
        ]
        
        for fmt in date_formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.isoformat()
            except ValueError:
                continue
        
        # Return as-is if we couldn't parse it
        return date_str
    
    def _extract_address(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Extract address information from award data.
        
        Args:
            item: Award data dictionary
            
        Returns:
            Address dictionary
        """
        # Try to get address from detailed award data first
        recipient_location = self._extract_field(item, ['award_detail.recipient.location'], {})
        
        if recipient_location:
            return {
                'address_line1': recipient_location.get('address_line1', ''),
                'address_line2': recipient_location.get('address_line2', ''),
                'city': recipient_location.get('city_name', ''),
                'state': recipient_location.get('state_code', ''),
                'zip': recipient_location.get('zip5', ''),
                'country': recipient_location.get('country_name', '')
            }
        
        # Fallback to extracting from fields if available
        return {
            'address_line1': '',
            'address_line2': '',
            'city': '',
            'state': '',
            'zip': '',
            'country': ''
        }
    
    def _extract_place_of_performance(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Extract place of performance information from award data.
        
        Args:
            item: Award data dictionary
            
        Returns:
            Place of performance dictionary
        """
        # Try to get from detailed award data first
        pop = self._extract_field(item, ['award_detail.place_of_performance'], {})
        
        if pop:
            return {
                'city': pop.get('city_name', ''),
                'county': pop.get('county_name', ''),
                'state': pop.get('state_code', ''),
                'zip': pop.get('zip5', ''),
                'country': pop.get('country_name', '')
            }
        
        # Fallback to extracting from fields if available
        return {
            'city': self._extract_field(item, ['Place of Performance City Code'], ''),
            'state': self._extract_field(item, ['Place of Performance State Code'], ''),
            'zip': self._extract_field(item, ['Place of Performance Zip Code'], ''),
            'country': self._extract_field(item, ['Place of Performance Country Code'], '')
        }
    
    def _format_place_of_performance(self, pop: Dict[str, Any]) -> str:
        """Format place of performance as a string.
        
        Args:
            pop: Place of performance dictionary
            
        Returns:
            Formatted place of performance string
        """
        parts = []
        
        if pop.get('city'):
            parts.append(pop['city'])
        
        if pop.get('state'):
            parts.append(pop['state'])
        
        if pop.get('zip'):
            parts.append(pop['zip'])
        
        if pop.get('country') and pop.get('country') != 'UNITED STATES':
            parts.append(pop['country'])
        
        return ', '.join(parts)
    
    def _extract_executive_details(self, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract executive compensation details if available.
        
        Args:
            item: Award data dictionary
            
        Returns:
            List of executive compensation dictionaries
        """
        executives = self._extract_field(
            item, 
            ['award_detail.executive_details.officers'], 
            []
        )
        
        if not executives:
            return []
            
        return [
            {
                'name': exec.get('name', ''),
                'amount': exec.get('amount', 0)
            }
            for exec in executives if exec.get('name')
        ]