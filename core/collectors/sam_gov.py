# core/collectors/sam_gov.py
import aiohttp
import asyncio
from datetime import datetime, timedelta
import json
import logging
import os
import time
from typing import Dict, List, Any, Optional
import backoff

from core.collectors.base_collector import BaseCollector

class SamGovCollector(BaseCollector):
    """Collector for SAM.gov opportunities and contract awards.
    
    This collector connects to the SAM.gov API to retrieve contract opportunities,
    awards, and related data. It handles authentication, pagination, and rate limiting
    according to SAM.gov API requirements.
    
    Attributes:
        source_name: Name of the data source (default: "sam.gov")
        config: Configuration dictionary for the collector
        auth_token: Cached authentication token
        token_expiry: Expiration timestamp for the auth token
    """
    
    def __init__(self, source_name: str = "sam.gov", config: Dict[str, Any] = None):
        """Initialize the SAM.gov collector.
        
        Args:
            source_name: Name of the data source
            config: Configuration dictionary with API settings
        """
        default_config = {
            'api_key': os.environ.get('SAM_GOV_API_KEY'),
            'base_url': 'https://api.sam.gov/opportunities/v1/search',
            'auth_url': 'https://api.sam.gov/oauth2/token',
            'published_since': (datetime.now() - timedelta(days=30)).isoformat(),
            'posted_date_start': (datetime.now() - timedelta(days=30)).strftime("%m/%d/%Y"),
            'posted_date_end': datetime.now().strftime("%m/%d/%Y"),
            'limit': 100,
            'page': 0,
            'sort': '-modifiedDate',
            'include_awards': True,
            'api_timeout': 60,  # Timeout in seconds
            'rate_limit': {
                'max_requests': 100,  # Requests per time_period
                'time_period': 60     # Time period in seconds
            },
            'retry_config': {
                'max_tries': 5,
                'max_time': 300,      # Max retry time in seconds
                'initial_wait': 2,    # Initial wait in seconds
                'max_wait': 60        # Max wait between retries
            }
        }
        
        # Use provided config or default, with environment variables as fallback
        if config:
            self.config = {**default_config, **config}
        else:
            self.config = default_config
            
        # If no API key in config, try environment
        if not self.config['api_key']:
            self.config['api_key'] = os.environ.get('SAM_GOV_API_KEY')
            if not self.config['api_key']:
                raise ValueError("No SAM.gov API key provided. Set SAM_GOV_API_KEY environment variable or pass in config.")
                
        super().__init__(source_name, self.config)
        
        # Authentication state
        self.auth_token = None
        self.token_expiry = 0
        
        # Set up rate limiting
        self.request_timestamps = []
        
    async def _get_auth_token(self) -> str:
        """Get a valid authentication token from SAM.gov.
        
        Returns:
            String containing the valid authentication token
        
        Raises:
            Exception: If authentication fails
        """
        # Check if we have a valid cached token
        current_time = time.time()
        if self.auth_token and current_time < self.token_expiry:
            return self.auth_token
            
        self.logger.info("Obtaining new SAM.gov authentication token")
        
        # Prepare request
        auth_data = {
            'grant_type': 'client_credentials',
            'client_id': self.config['api_key'],
            'client_secret': self.config['api_key']  # Note: SAM.gov API often uses API key as both ID and secret
        }
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'application/json'
        }
        
        try:
            # Make authentication request
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config['auth_url'], 
                    data=auth_data,
                    headers=headers,
                    timeout=self.config['api_timeout']
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        self.logger.error(f"Authentication failed: {response.status} - {error_text}")
                        raise Exception(f"Failed to authenticate with SAM.gov: {response.status}")
                    
                    auth_response = await response.json()
                    
                    # Extract token and expiry
                    if 'access_token' not in auth_response:
                        raise Exception(f"Invalid authentication response from SAM.gov: {auth_response}")
                    
                    self.auth_token = auth_response['access_token']
                    # Set expiry slightly before actual expiry to be safe
                    expires_in = auth_response.get('expires_in', 3600)  # Default 1 hour
                    self.token_expiry = current_time + expires_in - 60  # 60 second buffer
                    
                    self.logger.info(f"Successfully obtained authentication token, expires in {expires_in} seconds")
                    return self.auth_token
                    
        except aiohttp.ClientError as e:
            self.logger.error(f"Authentication request error: {str(e)}")
            raise
    
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
    async def _make_api_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make an authenticated request to the SAM.gov API.
        
        Args:
            endpoint: API endpoint to call
            params: Query parameters for the request
            
        Returns:
            JSON response from the API
            
        Raises:
            Exception: If the API request fails
        """
        # Get authentication token
        token = await self._get_auth_token()
        
        # Enforce rate limiting
        await self._enforce_rate_limit()
        
        # Prepare request
        headers = {
            'Authorization': f'Bearer {token}',
            'Accept': 'application/json'
        }
        
        full_url = endpoint if endpoint.startswith('http') else f"{self.config['base_url']}/{endpoint}"
        
        try:
            self.logger.info(f"Making API request to {full_url}")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    full_url,
                    params=params,
                    headers=headers,
                    timeout=self.config['api_timeout']
                ) as response:
                    response_text = await response.text()
                    
                    # Handle API errors
                    if response.status != 200:
                        self.logger.error(f"API request failed: {response.status} - {response_text}")
                        
                        # Special handling for authentication errors
                        if response.status == 401:
                            self.logger.info("Authentication token expired, clearing cached token")
                            self.auth_token = None
                            self.token_expiry = 0
                        
                        raise Exception(f"SAM.gov API request failed: {response.status}")
                    
                    try:
                        data = json.loads(response_text)
                        return data
                    except json.JSONDecodeError:
                        self.logger.error(f"Failed to parse API response as JSON: {response_text[:1000]}...")
                        raise Exception("Invalid JSON response from SAM.gov API")
                        
        except aiohttp.ClientError as e:
            self.logger.error(f"API request error: {str(e)}")
            raise
    
    async def collect(self) -> List[Dict[str, Any]]:
        """Collect opportunities and awards from SAM.gov API.
        
        Returns:
            List of opportunity and award data dictionaries
            
        Raises:
            Exception: If data collection fails
        """
        self.logger.info("Starting SAM.gov data collection")
        
        all_results = []
        total_pages = 1
        current_page = 0
        
        while current_page < total_pages:
            # Prepare query parameters
            params = {
                'api_key': self.config['api_key'],
                'limit': self.config['limit'],
                'page': current_page,
                'postedFrom': self.config['posted_date_start'],
                'postedTo': self.config['posted_date_end'],
                'sortBy': self.config['sort']
            }
            
            try:
                # Make API request
                data = await self._make_api_request('', params)
                
                # Extract results
                opportunities = data.get('opportunitiesData', [])
                all_results.extend(opportunities)
                
                self.logger.info(f"Retrieved {len(opportunities)} opportunities (page {current_page + 1})")
                
                # Update pagination info
                if current_page == 0:
                    total_results = data.get('totalRecords', 0)
                    total_pages = (total_results + self.config['limit'] - 1) // self.config['limit']
                    self.logger.info(f"Found {total_results} total opportunities across {total_pages} pages")
                
                current_page += 1
                
                # Check if we have all results or reached the end
                if not opportunities or len(all_results) >= total_results:
                    break
                    
            except Exception as e:
                self.logger.error(f"Error collecting data from SAM.gov: {str(e)}")
                if current_page > 0:
                    # If we have some results, continue with partial data
                    self.logger.info(f"Continuing with {len(all_results)} results collected so far")
                    break
                else:
                    # If first page fails, we have no data
                    raise
        
        # If we're configured to include awards and have opportunity IDs, fetch award data
        if self.config['include_awards'] and all_results:
            await self._fetch_award_data(all_results)
        
        self.logger.info(f"Completed SAM.gov data collection with {len(all_results)} items")
        return all_results
    
    async def _fetch_award_data(self, opportunities: List[Dict[str, Any]]):
        """Fetch award data for opportunities.
        
        Args:
            opportunities: List of opportunity dictionaries to fetch awards for
        """
        self.logger.info(f"Fetching award data for {len(opportunities)} opportunities")
        
        # Get opportunity IDs that need award data
        opportunity_ids = [
            opp.get('opportunityId') for opp in opportunities 
            if opp.get('opportunityId') and opp.get('award') is None
        ]
        
        if not opportunity_ids:
            self.logger.info("No opportunities need award data")
            return
            
        # Process in batches to avoid large requests
        batch_size = 10
        for i in range(0, len(opportunity_ids), batch_size):
            batch_ids = opportunity_ids[i:i+batch_size]
            
            try:
                # Make award API request
                params = {
                    'api_key': self.config['api_key'],
                    'opportunityIds': ','.join(batch_ids)
                }
                
                award_data = await self._make_api_request('awards', params)
                awards = award_data.get('awardsData', [])
                
                # Map awards back to opportunities
                award_map = {award.get('opportunityId'): award for award in awards}
                
                for opp in opportunities:
                    opp_id = opp.get('opportunityId')
                    if opp_id in award_map:
                        opp['award'] = award_map[opp_id]
                
                self.logger.info(f"Retrieved {len(awards)} awards for batch of {len(batch_ids)} opportunities")
                
            except Exception as e:
                self.logger.error(f"Error fetching award data: {str(e)}")
                # Continue with next batch
    
    def process_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process and normalize SAM.gov data.
        
        Args:
            results: Raw data from SAM.gov API
            
        Returns:
            Normalized list of opportunity and award dictionaries
        """
        processed = super().process_results(results)
        
        # Normalize fields to our internal format
        for item in processed:
            # Basic opportunity fields
            if 'opportunityId' in item:
                item['id'] = item['opportunityId']
            
            # Handle agency information
            if 'agencyCode' in item or 'agency' in item:
                agency_name = item.get('agency', '')
                agency_id = item.get('agencyCode', item.get('agencyId', ''))
                
                if not agency_id and agency_name:
                    # Try to extract agency ID from name
                    if '(' in agency_name and ')' in agency_name:
                        start = agency_name.rfind('(') + 1
                        end = agency_name.rfind(')')
                        if end > start:
                            agency_id = agency_name[start:end].strip()
                
                item['agency_data'] = {
                    'name': agency_name,
                    'id': agency_id
                }
            
            # Handle dates
            for date_field in ['postedDate', 'closeDate', 'updatedDate', 'publishDate']:
                if date_field in item and item[date_field]:
                    try:
                        # Convert to ISO format if not already
                        if 'T' not in item[date_field]:
                            dt = datetime.strptime(item[date_field], '%m/%d/%Y')
                            item[date_field] = dt.isoformat()
                    except (ValueError, TypeError):
                        self.logger.warning(f"Failed to parse date: {item[date_field]}")
            
            # Normalize award data if present
            if 'award' in item and item['award']:
                award = item['award']
                
                # Format award data for our knowledge graph
                item['award_data'] = {
                    'agency_id': item.get('agencyCode', item.get('agencyId', '')),
                    'agency_name': item.get('agency', ''),
                    'contractor_id': award.get('awardeeId', award.get('awardeeCode', award.get('awardeeUEI', ''))),
                    'contractor_name': award.get('awardee', ''),
                    'opportunity_id': item.get('opportunityId', ''),
                    'title': item.get('title', ''),
                    'description': item.get('description', ''),
                    'value': self._parse_award_amount(award),
                    'award_date': award.get('date', award.get('awardDate', '')),
                    'naics_code': item.get('naicsCode', ''),
                    'contract_number': award.get('contractNumber', ''),
                    'period_of_performance': award.get('periodOfPerformance', '')
                }
                
                # Ensure all award_data fields have at least empty strings
                for key in item['award_data']:
                    if item['award_data'][key] is None:
                        item['award_data'][key] = ''
        
        return processed
    
    def _parse_award_amount(self, award: Dict[str, Any]) -> float:
        """Parse award amount from various possible fields.
        
        Args:
            award: Award data dictionary
            
        Returns:
            Floating point award amount or 0.0 if not found
        """
        amount = 0.0
        
        # Check various possible field names
        for field in ['amount', 'awardAmount', 'obligatedAmount', 'baseAndAllOptionsValue']:
            if field in award and award[field]:
                try:
                    # Handle string amounts with currency symbols or commas
                    if isinstance(award[field], str):
                        # Remove currency symbols, commas, and spaces
                        clean_amount = award[field].replace('$', '').replace(',', '').replace(' ', '')
                        amount = float(clean_amount)
                    else:
                        amount = float(award[field])
                    break
                except (ValueError, TypeError):
                    self.logger.warning(f"Failed to parse award amount: {award[field]}")
        
        return amount