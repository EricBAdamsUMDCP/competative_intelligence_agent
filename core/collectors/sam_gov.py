# core/collectors/sam_gov.py
import aiohttp
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
import os
import json
import logging
from urllib.parse import urlencode, urljoin
import base64
import re
import tempfile
import zipfile
import io

from core.collectors.base_collector import BaseCollector

class SamGovCollector(BaseCollector):
    """
    Collector for SAM.gov opportunities using the official API
    
    SAM.gov API documentation: https://open.gsa.gov/api/get-opportunities-public-api/
    
    This collector supports:
    - Searching for opportunities using various filters
    - Downloading opportunity attachments
    - Pagination of results
    - Tracking changes since last collection
    """
    
    def __init__(self, source_name: str = "sam.gov", config: Optional[Dict[str, Any]] = None):
        default_config = {
            'api_key': os.environ.get('SAM_GOV_API_KEY'),
            'base_url': 'https://api.sam.gov/opportunities/v2',
            'search_endpoint': '/search',
            'attachment_endpoint': '/attachment',
            # Time period parameters
            'published_since': (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
            'published_until': None,
            'limit': 100,
            'page': 1,
            # Search filters
            'notice_type': None,  # e.g., 'COMBINE', 'PRESOL', 'AWARD', etc.
            'keywords': None,
            'naics_code': None,
            'psc_code': None,
            'set_aside_code': None,  # e.g., 'SBA', 'SDVOSBC', etc.
            'agency_code': None,
            'state_code': None,
            'zip_code': None,
            'opportunity_status': 'active',  # 'active', 'inactive', or 'all'
            # Mode options
            'use_mock': False,  # Set to True for testing with mock data
            'api_key_expiry': None,  # Track API key expiry to remind about 90-day rotation
            'download_attachments': False,  # Whether to download attachments
            'attachment_dir': None,  # Directory to save attachments (None = temp directory)
            'max_attachment_size_mb': 20,  # Maximum size of attachments to download
            'use_pagination': True,  # Whether to paginate through all results
            'max_pages': 10,  # Maximum number of pages to retrieve
            'save_raw_response': False,  # Whether to save raw API responses
            'raw_response_dir': None,  # Directory to save raw responses
        }
        
        # Use provided config or default
        merged_config = default_config.copy()
        if config is not None:
            merged_config.update(config)
        
        super().__init__(source_name, merged_config)
        
        # Check API key
        if not self.config['use_mock'] and not self.config['api_key']:
            self.logger.warning("SAM.gov API key not set. Either set the SAM_GOV_API_KEY environment variable or provide it in the config.")
        
        # Initialize tracking variables
        self.last_modified_date = None
        self.total_opportunities = 0
        self.total_downloaded = 0
        self.total_attachments = 0
    
    async def collect(self) -> List[Dict[str, Any]]:
        """Collect opportunities from SAM.gov API"""
        if self.config['use_mock']:
            return await self._mock_collection()
        
        # Check API key expiry
        self._check_api_key_expiry()
        
        # Initialize results
        all_opportunities = []
        
        # Get the first page of results
        opportunities, metadata = await self._fetch_opportunities_page(self.config['page'])
        
        # Add to results
        if opportunities:
            all_opportunities.extend(opportunities)
        
        # Handle pagination if enabled
        if self.config['use_pagination'] and metadata and 'totalRecords' in metadata:
            total_records = metadata['totalRecords']
            records_per_page = self.config['limit']
            total_pages = (total_records + records_per_page - 1) // records_per_page
            
            # Cap at max_pages
            total_pages = min(total_pages, self.config['max_pages'])
            
            self.logger.info(f"Found {total_records} total records across {total_pages} pages")
            
            # Fetch remaining pages
            if total_pages > 1:
                tasks = []
                for page in range(2, total_pages + 1):
                    tasks.append(self._fetch_opportunities_page(page))
                
                # Gather results, handling any errors
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result_item in results:
                    if isinstance(result_item, Exception):
                        self.logger.error(f"Error fetching page: {str(result_item)}")
                    else:
                        # Unpack the tuple returned from _fetch_opportunities_page
                        # We know it returns (List[Dict], Optional[Dict]) or empty values for errors
                        page_opportunities: List[Dict[str, Any]] = []
                        page_metadata: Optional[Dict[str, Any]] = None
                        
                        # Safely unpack the result tuple
                        if isinstance(result_item, tuple) and len(result_item) >= 1:
                            page_opportunities = result_item[0] or []
                            if len(result_item) >= 2:
                                page_metadata = result_item[1]
                        
                        if page_opportunities:
                            all_opportunities.extend(page_opportunities)
        
        self.total_opportunities = len(all_opportunities)
        self.logger.info(f"Retrieved {self.total_opportunities} opportunities from SAM.gov")
        
        # Download attachments if configured
        if self.config['download_attachments'] and all_opportunities:
            await self._download_attachments(all_opportunities)
        
        return all_opportunities
    
    async def _fetch_opportunities_page(self, page: int) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Fetch a single page of opportunities from SAM.gov API"""
        # Build request parameters
        params = {
            'api_key': self.config['api_key'],
            'limit': self.config['limit'],
            'page': page,
            'postedFrom': self.config['published_since'],
            'includeSections': 'O,A,F,P,H',  # O=opportunity, A=award, F=description, P=points of contact, H=history
            'format': 'json'
        }
        
        # Add optional filters if provided
        if self.config['published_until']:
            params['postedTo'] = self.config['published_until']
        
        if self.config['notice_type']:
            params['noticeType'] = self.config['notice_type']
        
        if self.config['keywords']:
            if isinstance(self.config['keywords'], list):
                params['keyword'] = ' '.join(self.config['keywords'])
            else:
                params['keyword'] = self.config['keywords']
        
        if self.config['naics_code']:
            params['naicsCode'] = self.config['naics_code']
        
        if self.config['psc_code']:
            params['pscCode'] = self.config['psc_code']
        
        if self.config['set_aside_code']:
            params['setAsideCode'] = self.config['set_aside_code']
        
        if self.config['agency_code']:
            params['agencyCode'] = self.config['agency_code']
        
        if self.config['state_code']:
            params['placeOfPerformanceStateCode'] = self.config['state_code']
        
        if self.config['zip_code']:
            params['placeOfPerformanceZipCode'] = self.config['zip_code']
        
        if self.config['opportunity_status']:
            params['status'] = self.config['opportunity_status']
        
        # Build URL
        url = f"{self.config['base_url']}{self.config['search_endpoint']}?{urlencode(params)}"
        
        # Make the request
        async with aiohttp.ClientSession() as session:
            try:
                self.logger.info(f"Fetching page {page} from SAM.gov API")
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Save raw response if configured
                        if self.config['save_raw_response']:
                            await self._save_raw_response(f"page_{page}", data)
                        
                        # Extract opportunities and metadata
                        opportunities = data.get('opportunitiesData', [])
                        metadata = data.get('metadata', {})
                        
                        self.logger.info(f"Retrieved {len(opportunities)} opportunities on page {page}")
                        return opportunities, metadata
                    else:
                        content = await response.text()
                        self.logger.error(f"SAM.gov API error: {response.status} - {content}")
                        return [], None
            except Exception as e:
                self.logger.error(f"Error fetching opportunities page {page}: {str(e)}")
                raise
    
    async def _download_attachments(self, opportunities: List[Dict[str, Any]]) -> None:
        """Download attachments for opportunities"""
        self.logger.info("Downloading attachments for opportunities")
        
        # Create attachment directory if needed
        attachment_dir = self.config['attachment_dir']
        if not attachment_dir:
            attachment_dir = tempfile.mkdtemp(prefix="samgov_attachments_")
        else:
            os.makedirs(attachment_dir, exist_ok=True)
        
        # Get opportunities with resources
        opportunities_with_attachments = [
            opp for opp in opportunities 
            if 'resources' in opp and opp['resources']
        ]
        
        if not opportunities_with_attachments:
            self.logger.info("No attachments found in opportunities")
            return
        
        self.logger.info(f"Found {len(opportunities_with_attachments)} opportunities with attachments")
        
        # Download attachments
        async with aiohttp.ClientSession() as session:
            for opp in opportunities_with_attachments:
                opp_id = opp.get('opportunityId', 'unknown')
                resources = opp.get('resources', [])
                
                for resource in resources:
                    if 'attachments' not in resource:
                        continue
                    
                    attachments = resource.get('attachments', [])
                    if not attachments:
                        continue
                    
                    # Create opportunity directory
                    opp_dir = os.path.join(attachment_dir, opp_id)
                    os.makedirs(opp_dir, exist_ok=True)
                    
                    for attachment in attachments:
                        attachment_id = attachment.get('attachmentId')
                        attachment_name = attachment.get('name', f"attachment_{attachment_id}")
                        
                        # Check size if available
                        size_mb = attachment.get('size', 0) / (1024 * 1024)
                        if size_mb > self.config['max_attachment_size_mb']:
                            self.logger.warning(f"Skipping large attachment {attachment_name} ({size_mb:.2f} MB)")
                            continue
                        
                        try:
                            # Build attachment URL
                            attachment_url = f"{self.config['base_url']}{self.config['attachment_endpoint']}/{attachment_id}"
                            params = {'api_key': self.config['api_key']}
                            url = f"{attachment_url}?{urlencode(params)}"
                            
                            # Download attachment
                            async with session.get(url) as response:
                                if response.status == 200:
                                    # Save attachment
                                    file_path = os.path.join(opp_dir, self._sanitize_filename(attachment_name))
                                    content = await response.read()
                                    
                                    with open(file_path, 'wb') as f:
                                        f.write(content)
                                    
                                    # Update attachment path in opportunity
                                    attachment['local_path'] = file_path
                                    self.total_attachments += 1
                                    
                                    self.logger.debug(f"Downloaded attachment: {file_path}")
                                else:
                                    content = await response.text()
                                    self.logger.error(f"Error downloading attachment {attachment_id}: {response.status} - {content}")
                        except Exception as e:
                            self.logger.error(f"Error downloading attachment {attachment_id}: {str(e)}")
        
        self.logger.info(f"Downloaded {self.total_attachments} attachments to {attachment_dir}")
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize a filename to be safe for the file system"""
        # Replace invalid characters with underscores
        sanitized = re.sub(r'[\\/*?:"<>|]', '_', filename)
        # Limit length
        if len(sanitized) > 255:
            sanitized = sanitized[:250] + sanitized[-5:]
        return sanitized
    
    async def _save_raw_response(self, prefix: str, data: Dict[str, Any]) -> None:
        """Save raw API response to a file"""
        try:
            # Create directory if needed
            save_dir = self.config['raw_response_dir']
            if not save_dir:
                save_dir = tempfile.mkdtemp(prefix="samgov_responses_")
            else:
                os.makedirs(save_dir, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{prefix}_{timestamp}.json"
            filepath = os.path.join(save_dir, filename)
            
            # Write to file
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.debug(f"Saved raw response to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving raw response: {str(e)}")
    
    def _check_api_key_expiry(self) -> None:
        """Check if the API key is near expiry and log warnings"""
        if not self.config['api_key_expiry']:
            return
        
        try:
            expiry_date = datetime.fromisoformat(self.config['api_key_expiry'])
            days_until_expiry = (expiry_date - datetime.now()).days
            
            if days_until_expiry <= 0:
                self.logger.error("⚠️ SAM.gov API key has EXPIRED. Please update with a new key.")
            elif days_until_expiry <= 7:
                self.logger.warning(f"⚠️ SAM.gov API key expires in {days_until_expiry} days. Please rotate the key ASAP.")
            elif days_until_expiry <= 15:
                self.logger.info(f"SAM.gov API key expires in {days_until_expiry} days. Consider rotating the key soon.")
        except Exception as e:
            self.logger.error(f"Error checking API key expiry: {str(e)}")
    
    async def _mock_collection(self) -> List[Dict[str, Any]]:
        """Generate mock data for testing purposes"""
        self.logger.info("Using mock SAM.gov data (for testing only)")
        
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
                ],
                'resources': [
                    {
                        'attachments': [
                            {
                                'attachmentId': 'att123456',
                                'name': 'RFP_Document.pdf',
                                'size': 1024 * 1024,  # 1 MB
                                'description': 'Request for Proposal Document'
                            }
                        ]
                    }
                ],
                'history': [
                    {
                        'date': (datetime.now() - timedelta(days=10)).isoformat(),
                        'action': 'DRAFT',
                        'user': 'system'
                    },
                    {
                        'date': (datetime.now() - timedelta(days=5)).isoformat(),
                        'action': 'PUBLISHED',
                        'user': 'john.smith@defense.gov'
                    },
                    {
                        'date': (datetime.now() - timedelta(days=1)).isoformat(),
                        'action': 'UPDATED',
                        'user': 'john.smith@defense.gov'
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
                ],
                'resources': [
                    {
                        'attachments': [
                            {
                                'attachmentId': 'att789012',
                                'name': 'Statement_of_Work.docx',
                                'size': 512 * 1024,  # 512 KB
                                'description': 'Statement of Work Document'
                            },
                            {
                                'attachmentId': 'att789013',
                                'name': 'Q_and_A.pdf',
                                'size': 256 * 1024,  # 256 KB
                                'description': 'Questions and Answers'
                            }
                        ]
                    }
                ],
                'history': [
                    {
                        'date': (datetime.now() - timedelta(days=7)).isoformat(),
                        'action': 'DRAFT',
                        'user': 'system'
                    },
                    {
                        'date': (datetime.now() - timedelta(days=2)).isoformat(),
                        'action': 'PUBLISHED',
                        'user': 'sarah.johnson@hhs.gov'
                    }
                ]
            },
            {
                'opportunityId': 'SAMGOV345678',
                'title': 'Artificial Intelligence Research and Development',
                'solicitationNumber': 'NASA-AI-2023-03',
                'agency': 'National Aeronautics and Space Administration',
                'agencyId': 'NASA',
                'postedDate': (datetime.now() - timedelta(days=5)).isoformat(),
                'type': 'RFP',
                'baseType': 'RFP',
                'archiveType': 'AUTO30',
                'archiveDate': (datetime.now() + timedelta(days=60)).isoformat(),
                'setAside': 'N/A',
                'responseDeadLine': (datetime.now() + timedelta(days=30)).isoformat(),
                'naicsCode': '541715',
                'classificationCode': 'A',
                'active': 'Yes',
                'award': None,
                'pointOfContact': [
                    {
                        'name': 'Michael Chen',
                        'email': 'michael.chen@nasa.gov',
                        'phone': '202-555-9012',
                        'type': 'PRIMARY'
                    }
                ],
                'description': 'NASA is seeking proposals for artificial intelligence research and development to support space exploration missions. The focus areas include autonomous systems, machine learning for data analysis, and predictive maintenance of spacecraft components.',
                'links': [
                    {
                        'url': 'https://sam.gov/opp/345678',
                        'type': 'OPPORTUNITY'
                    }
                ],
                'resources': [
                    {
                        'attachments': [
                            {
                                'attachmentId': 'att345678',
                                'name': 'Technical_Requirements.pdf',
                                'size': 1.5 * 1024 * 1024,  # 1.5 MB
                                'description': 'Technical Requirements Document'
                            },
                            {
                                'attachmentId': 'att345679',
                                'name': 'Evaluation_Criteria.docx',
                                'size': 750 * 1024,  # 750 KB
                                'description': 'Evaluation Criteria Document'
                            }
                        ]
                    }
                ],
                'history': [
                    {
                        'date': (datetime.now() - timedelta(days=15)).isoformat(),
                        'action': 'DRAFT',
                        'user': 'system'
                    },
                    {
                        'date': (datetime.now() - timedelta(days=10)).isoformat(),
                        'action': 'REVIEWED',
                        'user': 'approver@nasa.gov'
                    },
                    {
                        'date': (datetime.now() - timedelta(days=5)).isoformat(),
                        'action': 'PUBLISHED',
                        'user': 'michael.chen@nasa.gov'
                    }
                ]
            }
        ]
        
        # Artificial delay to simulate network request
        await asyncio.sleep(1)
        
        self.total_opportunities = len(mock_data)
        
        # If attachments are enabled, create mock attachment files
        if self.config['download_attachments']:
            await self._create_mock_attachments(mock_data)
        
        return mock_data
    
    async def _create_mock_attachments(self, opportunities: List[Dict[str, Any]]) -> None:
        """Create mock attachment files for testing"""
        self.logger.info("Creating mock attachment files")
        
        # Create attachment directory
        attachment_dir = self.config['attachment_dir']
        if not attachment_dir:
            attachment_dir = tempfile.mkdtemp(prefix="samgov_mock_attachments_")
        else:
            os.makedirs(attachment_dir, exist_ok=True)
        
        # Create mock attachments for each opportunity
        for opp in opportunities:
            opp_id = opp.get('opportunityId', 'unknown')
            resources = opp.get('resources', [])
            
            if not resources:
                continue
            
            # Create opportunity directory
            opp_dir = os.path.join(attachment_dir, opp_id)
            os.makedirs(opp_dir, exist_ok=True)
            
            for resource in resources:
                if 'attachments' not in resource:
                    continue
                
                attachments = resource.get('attachments', [])
                if not attachments:
                    continue
                
                for attachment in attachments:
                    attachment_name = attachment.get('name', f"attachment_{attachment.get('attachmentId', 'unknown')}")
                    attachment_desc = attachment.get('description', 'Mock attachment')
                    
                    # Create a mock file
                    file_path = os.path.join(opp_dir, self._sanitize_filename(attachment_name))
                    file_content = f"Mock attachment for {opp_id}\n\nDescription: {attachment_desc}\n\nGenerated for testing purposes."
                    
                    with open(file_path, 'w') as f:
                        f.write(file_content)
                    
                    # Update attachment
                    attachment['local_path'] = file_path
                    self.total_attachments += 1
                    
                    self.logger.debug(f"Created mock attachment: {file_path}")
        
        self.logger.info(f"Created {self.total_attachments} mock attachments in {attachment_dir}")
    
    def process_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process and normalize SAM.gov data"""
        processed = super().process_results(results)
        
        # Normalize fields to our internal format
        for item in processed:
            # Extract common fields
            if 'opportunityId' in item:
                item['id'] = item['opportunityId']
            
            if 'agency' in item and 'agencyId' in item:
                item['agency_data'] = {
                    'name': item['agency'],
                    'id': item['agencyId']
                }
            
            # Additional information for entity extraction
            if 'description' in item and item['description']:
                item['additional_info'] = self._build_additional_info(item)
            
            if 'award' in item and item['award']:
                # Format award data for knowledge graph
                award = item['award']
                item['award_data'] = {
                    'agency_id': item.get('agencyId', ''),
                    'agency_name': item.get('agency', ''),
                    'contractor_id': award.get('awardeeId', ''),
                    'contractor_name': award.get('awardee', ''),
                    'opportunity_id': item.get('opportunityId', ''),
                    'title': item.get('title', ''),
                    'description': item.get('description', ''),
                    'value': award.get('amount', 0),
                    'award_date': award.get('date', '')
                }
                
                # Add additional award data if available
                if 'modNumber' in award:
                    item['award_data']['mod_number'] = award['modNumber']
                
                if 'contractAwardNumber' in award:
                    item['award_data']['contract_award_number'] = award['contractAwardNumber']
            
            # Extract NAICS code
            if 'naicsCode' in item:
                if 'extracted_entities' not in item:
                    item['extracted_entities'] = {}
                
                if 'NAICS_CODE' not in item['extracted_entities']:
                    item['extracted_entities']['NAICS_CODE'] = []
                
                item['extracted_entities']['NAICS_CODE'].append({
                    'text': item['naicsCode'],
                    'start_char': 0,
                    'end_char': 0,
                    'confidence': 1.0
                })
            
            # Extract PSC code if present
            if 'classificationCode' in item:
                if 'extracted_entities' not in item:
                    item['extracted_entities'] = {}
                
                if 'PSC_CODE' not in item['extracted_entities']:
                    item['extracted_entities']['PSC_CODE'] = []
                
                item['extracted_entities']['PSC_CODE'].append({
                    'text': item['classificationCode'],
                    'start_char': 0,
                    'end_char': 0,
                    'confidence': 1.0
                })
            
            # Extract set-aside code if present
            if 'setAside' in item and item['setAside'] and item['setAside'] != 'N/A':
                if 'extracted_entities' not in item:
                    item['extracted_entities'] = {}
                
                if 'SET_ASIDE' not in item['extracted_entities']:
                    item['extracted_entities']['SET_ASIDE'] = []
                
                item['extracted_entities']['SET_ASIDE'].append({
                    'text': item['setAside'],
                    'start_char': 0,
                    'end_char': 0,
                    'confidence': 1.0
                })
            
            # Extract point of contact information
            if 'pointOfContact' in item and item['pointOfContact']:
                if 'extracted_entities' not in item:
                    item['extracted_entities'] = {}
                
                if 'PERSON' not in item['extracted_entities']:
                    item['extracted_entities']['PERSON'] = []
                
                for contact in item['pointOfContact']:
                    if 'name' in contact:
                        contact_info = {
                            'text': contact['name'],
                            'start_char': 0,
                            'end_char': 0,
                            'confidence': 1.0
                        }
                        
                        # Add additional contact details
                        if 'email' in contact:
                            contact_info['email'] = contact['email']
                        
                        if 'phone' in contact:
                            contact_info['phone'] = contact['phone']
                        
                        if 'type' in contact:
                            contact_info['type'] = contact['type']
                        
                        item['extracted_entities']['PERSON'].append(contact_info)
            
            # Extract URLs from links
            if 'links' in item and item['links']:
                if 'extracted_entities' not in item:
                    item['extracted_entities'] = {}
                
                if 'URL' not in item['extracted_entities']:
                    item['extracted_entities']['URL'] = []
                
                for link in item['links']:
                    if 'url' in link:
                        item['extracted_entities']['URL'].append({
                            'text': link['url'],
                            'type': link.get('type', 'UNKNOWN'),
                            'start_char': 0,
                            'end_char': 0,
                            'confidence': 1.0
                        })
            
            # Extract dates
            dates = {}
            if 'postedDate' in item:
                dates['posted_date'] = item['postedDate']
            
            if 'responseDeadLine' in item:
                dates['response_deadline'] = item['responseDeadLine']
            
            if 'archiveDate' in item:
                dates['archive_date'] = item['archiveDate']
            
            item['dates'] = dates
            
            # Add attachment metadata if available
            if 'resources' in item and item['resources']:
                attachments_info = []
                
                for resource in item['resources']:
                    if 'attachments' in resource:
                        for attachment in resource['attachments']:
                            attachment_info = {
                                'id': attachment.get('attachmentId', ''),
                                'name': attachment.get('name', ''),
                                'description': attachment.get('description', ''),
                                'size': attachment.get('size', 0)
                            }
                            
                            if 'local_path' in attachment:
                                attachment_info['local_path'] = attachment['local_path']
                            
                            attachments_info.append(attachment_info)
                
                if attachments_info:
                    item['attachments'] = attachments_info
        
        return processed
    
    def _build_additional_info(self, item: Dict[str, Any]) -> str:
        """Build a string of additional information from the item"""
        info_parts = []
        
        if 'naicsCode' in item:
            info_parts.append(f"NAICS: {item['naicsCode']}")
        
        if 'classificationCode' in item:
            info_parts.append(f"PSC: {item['classificationCode']}")
        
        if 'setAside' in item and item['setAside'] and item['setAside'] != 'N/A':
            info_parts.append(f"Set-Aside: {item['setAside']}")
        
        if 'type' in item:
            info_parts.append(f"Type: {item['type']}")
        
        if 'responseDeadLine' in item:
            info_parts.append(f"Response Due: {item['responseDeadLine']}")
        
        if 'active' in item:
            info_parts.append(f"Status: {item['active']}")
        
        return ', '.join(info_parts)
    
    async def search_opportunities(self, 
                                  keywords: Optional[Union[List[str], str]] = None, 
                                  agencies: Optional[Union[List[str], str]] = None, 
                                  naics_codes: Optional[Union[List[str], str]] = None,
                                  published_since: Optional[str] = None,
                                  published_until: Optional[str] = None,
                                  notice_type: Optional[str] = None,
                                  psc_code: Optional[str] = None,
                                  set_aside_code: Optional[str] = None,
                                  state_code: Optional[str] = None,
                                  zip_code: Optional[str] = None,
                                  opportunity_status: Optional[str] = None,
                                  limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Search for specific opportunities using the SAM.gov API"""
        # Create a custom config for this search
        search_config = self.config.copy()
        
        # Handle each parameter carefully, ensuring we don't pass None values
        if keywords is not None:
            search_config['keywords'] = keywords
        
        if agencies is not None:
            search_config['agency_code'] = agencies[0] if isinstance(agencies, list) and agencies else agencies
        
        if naics_codes is not None:
            search_config['naics_code'] = naics_codes[0] if isinstance(naics_codes, list) and naics_codes else naics_codes
        
        if published_since is not None:
            search_config['published_since'] = published_since
        
        if published_until is not None:
            search_config['published_until'] = published_until
        
        if notice_type is not None:
            search_config['notice_type'] = notice_type
        
        if psc_code is not None:
            search_config['psc_code'] = psc_code
        
        if set_aside_code is not None:
            search_config['set_aside_code'] = set_aside_code
        
        if state_code is not None:
            search_config['state_code'] = state_code
        
        if zip_code is not None:
            search_config['zip_code'] = zip_code
        
        if opportunity_status is not None:
            search_config['opportunity_status'] = opportunity_status
        
        if limit is not None:
            search_config['limit'] = limit
        
        # Create a temporary collector with the search config
        temp_collector = SamGovCollector(config=search_config)
        
        # Run the collection
        results = await temp_collector.collect()
        
        # Process the results
        return temp_collector.process_results(results)
    
    async def validate_api_key(self) -> bool:
        """Validate that the API key is working"""
        if not self.config['api_key']:
            return False
        
        # Make a minimal request to validate the key
        params = {
            'limit': 1,
            'api_key': self.config['api_key'],
            'format': 'json'
        }
        
        url = f"{self.config['base_url']}{self.config['search_endpoint']}?{urlencode(params)}"
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        self.logger.info("SAM.gov API key is valid")
                        return True
                    elif response.status == 401 or response.status == 403:
                        self.logger.error("SAM.gov API key is invalid or expired")
                        return False
                    else:
                        self.logger.warning(f"Unexpected status when validating SAM.gov API key: {response.status}")
                        return False
            except Exception as e:
                self.logger.error(f"Error validating SAM.gov API key: {str(e)}")
                return False
    
    async def set_api_key_expiry(self, days_from_now: int = 90):
        """Set the API key expiry date (default 90 days from now)"""
        self.config['api_key_expiry'] = (datetime.now() + timedelta(days=days_from_now)).isoformat()
        self.logger.info(f"API key expiry set to {self.config['api_key_expiry']}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the collector's last run"""
        return {
            'total_opportunities': self.total_opportunities,
            'total_downloaded': self.total_downloaded,
            'total_attachments': self.total_attachments,
            'last_run': self.last_run,
            'last_modified_date': self.last_modified_date
        }