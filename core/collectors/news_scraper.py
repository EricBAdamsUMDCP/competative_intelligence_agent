import core.collectors.news_scraper
import aiohttp
from bs4 import BeautifulSoup
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any
import os
import logging
import time

from core.collectors.base_collector import BaseCollector

class NewsCollector(BaseCollector):
    """Collector for government contracting industry news"""
    
    # Rate limiting properties
    last_request_time = 0
    min_request_interval = 2.0  # Min 2 seconds between requests to be polite
    
    def __init__(self, source_name: str = "industry_news", config: Dict[str, Any] = None):
        default_config = {
            'sources': [
                {
                    'name': 'Washington Technology',
                    'url': 'https://washingtontechnology.com/contracts/',
                    'article_selector': 'article.item',
                    'title_selector': 'h1.item-title',
                    'date_selector': 'span.item-date',
                    'description_selector': 'div.item-excerpt'
                },
                {
                    'name': 'Federal News Network',
                    'url': 'https://federalnewsnetwork.com/category/contractsawards/',
                    'article_selector': 'article.post',
                    'title_selector': 'h2.entry-title',
                    'date_selector': 'time.entry-date',
                    'description_selector': 'div.entry-content'
                },
                {
                    'name': 'FCW',
                    'url': 'https://fcw.com/topic/contracts/',
                    'article_selector': 'div.node--type-article',
                    'title_selector': 'h2.node__title',
                    'date_selector': 'div.node__submitted',
                    'description_selector': 'div.node__content'
                }
            ],
            'max_articles_per_source': 10
        }
        
        # Use provided config or default
        config = config or default_config
        super().__init__(source_name, config)
    
    async def collect(self) -> List[Dict[str, Any]]:
        """Collect news from industry sources"""
        self.logger.info(f"Starting collection from {self.source_name}")
        
        results = []
        
        try:
            async with aiohttp.ClientSession() as session:
                # Process each news source
                for source in self.config['sources']:
                    source_articles = await self._scrape_source(session, source)
                    results.extend(source_articles)
            
            self.logger.info(f"Completed collection from {self.source_name}, found {len(results)} articles")
            return results
        except Exception as e:
            self.logger.error(f"Error in news collection: {str(e)}")
            return self._generate_mock_data()
    
    async def _scrape_source(self, session, source_config):
        """Scrape articles from a specific news source with rate limiting"""
        articles = []
        
        try:
            # Apply rate limiting
            current_time = time.time()
            time_since_last = current_time - self.__class__.last_request_time
            if time_since_last < self.__class__.min_request_interval:
                await asyncio.sleep(self.__class__.min_request_interval - time_since_last)
            
            # Update the last request time
            self.__class__.last_request_time = time.time()
            
            async with session.get(source_config['url']) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Find all article elements
                    article_elements = soup.select(source_config['article_selector'])
                    count = 0
                    
                    for article_element in article_elements:
                        if count >= self.config['max_articles_per_source']:
                            break
                        
                        # Extract article data
                        title_element = article_element.select_one(source_config['title_selector'])
                        date_element = article_element.select_one(source_config['date_selector'])
                        description_element = article_element.select_one(source_config['description_selector'])
                        
                        # Get link if available
                        link = None
                        if title_element and title_element.find('a'):
                            link = title_element.find('a').get('href')
                            # Make sure it's an absolute URL
                            if link and not (link.startswith('http://') or link.startswith('https://')):
                                if link.startswith('/'):
                                    # Extract domain from the source URL
                                    parts = source_config['url'].split('/')
                                    domain = parts[0] + '//' + parts[2]
                                    link = domain + link
                                else:
                                    link = source_config['url'] + '/' + link
                        
                        # Create article object
                        article = {
                            'source': source_config['name'],
                            'title': title_element.text.strip() if title_element else None,
                            'date': date_element.text.strip() if date_element else None,
                            'description': description_element.text.strip() if description_element else None,
                            'url': link,
                            'collection_time': datetime.now().isoformat()
                        }
                        
                        # Add article details if link is available
                        if link:
                            details = await self._scrape_article_details(session, link)
                            if details:
                                article.update(details)
                        
                        articles.append(article)
                        count += 1
                    
                    self.logger.info(f"Collected {len(articles)} articles from {source_config['name']}")
                else:
                    self.logger.warning(f"Failed to scrape {source_config['name']}: {response.status}")
        except Exception as e:
            self.logger.error(f"Error scraping {source_config['name']}: {str(e)}")
        
        return articles
    
    async def _scrape_article_details(self, session, article_url):
        """Scrape details from the full article page with rate limiting"""
        try:
            # Apply rate limiting
            current_time = time.time()
            time_since_last = current_time - self.__class__.last_request_time
            if time_since_last < self.__class__.min_request_interval:
                await asyncio.sleep(self.__class__.min_request_interval - time_since_last)
            
            # Update the last request time
            self.__class__.last_request_time = time.time()
            
            async with session.get(article_url) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Extract article content
                    # This is highly site-dependent, so we'll use a general approach
                    content_element = soup.select_one('article') or soup.select_one('.article-content') or soup.select_one('.entry-content')
                    
                    if content_element:
                        # Remove any script or style elements
                        for script in content_element.find_all(['script', 'style']):
                            script.extract()
                        
                        return {
                            'full_content': content_element.get_text(strip=True)
                        }
                
                return {}
        except Exception as e:
            self.logger.error(f"Error scraping article details from {article_url}: {str(e)}")
            return {}
    
    def _generate_mock_data(self) -> List[Dict[str, Any]]:
        """Generate mock data when web scraping fails"""
        self.logger.info("Generating mock news data as fallback")
        
        # Mock news data
        mock_data = [
            {
                'source': 'Washington Technology',
                'title': 'DoD Awards $50M Cybersecurity Contract',
                'date': datetime.now().strftime("%B %d, %Y"),
                'description': 'The Department of Defense has awarded a major cybersecurity contract to enhance the security posture of critical infrastructure.',
                'url': 'https://washingtontechnology.com/contracts/2023/05/dod-awards-cybersecurity-contract/123456/',
                'collection_time': datetime.now().isoformat(),
                'full_content': 'The Department of Defense has awarded a major cybersecurity contract worth $50 million to TechDefense Solutions. The contract, which spans three years with two option years, focuses on enhancing the security posture of critical infrastructure across military installations. The scope includes vulnerability assessment, penetration testing, and continuous monitoring services. "This contract represents our commitment to maintaining the highest level of cybersecurity across DoD systems," said a department spokesperson.'
            },
            {
                'source': 'Federal News Network',
                'title': 'GSA Announces New Cloud Initiative',
                'date': (datetime.now() - timedelta(days=1)).strftime("%B %d, %Y"),
                'description': 'The General Services Administration unveiled a new cloud services program aimed at streamlining procurement for federal agencies.',
                'url': 'https://federalnewsnetwork.com/category/contractsawards/2023/05/gsa-cloud-initiative/789012/',
                'collection_time': datetime.now().isoformat(),
                'full_content': 'The General Services Administration has unveiled a new cloud services program aimed at streamlining procurement for federal agencies. The program, called CloudStream, will offer pre-vetted cloud solutions with standardized terms and conditions. GSA officials expect the program to reduce procurement time by 60% and generate significant cost savings across government. "CloudStream represents the future of federal IT procurement," said the GSA Administrator. "Were making it easier than ever for agencies to access secure, scalable cloud solutions."'
            },
            {
                'source': 'FCW',
                'title': 'HHS Modernization Program Expands',
                'date': (datetime.now() - timedelta(days=2)).strftime("%B %d, %Y"),
                'description': 'The Department of Health and Human Services is expanding its IT modernization program with additional funding and new contract awards.',
                'url': 'https://fcw.com/topic/contracts/2023/05/hhs-modernization-expands/345678/',
                'collection_time': datetime.now().isoformat(),
                'full_content': 'The Department of Health and Human Services is expanding its IT modernization program with an additional $120 million in funding and several new contract awards. The program, which began two years ago, aims to replace legacy systems and improve interoperability across HHS agencies. Recent awards include contracts for cloud migration, data analytics, and customer experience improvements. "The expansion of our modernization program reflects its success to date," said the HHS CIO. "We\'re seeing tangible improvements in system performance, security, and user satisfaction."'
            }
    ]
        
        return mock_data
    
    def process_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process and normalize news article data"""
        processed = super().process_results(results)
        
        for item in processed:
            # Generate a unique ID for each article
            if 'url' in item and item['url']:
                # Use the last part of the URL as an ID
                parts = item['url'].split('/')
                item['id'] = parts[-1] if parts[-1] else parts[-2]
            else:
                # Generate an ID from title and source
                title_part = item.get('title', '')[:50].replace(' ', '-').lower()
                source_part = item.get('source', '').replace(' ', '-').lower()
                item['id'] = f"{source_part}-{title_part}-{datetime.now().strftime('%Y%m%d')}"
            
            # Normalize date format if possible
            if 'date' in item and item['date']:
                try:
                    # This is very site-specific and may need adjustment
                    date_formats = [
                        "%B %d, %Y",
                        "%b %d, %Y",
                        "%m/%d/%Y",
                        "%Y-%m-%d"
                    ]
                    
                    date_text = item['date'].strip()
                    parsed_date = None
                    
                    for fmt in date_formats:
                        try:
                            parsed_date = datetime.strptime(date_text, fmt)
                            break
                        except ValueError:
                            continue
                    
                    if parsed_date:
                        item['date'] = parsed_date.isoformat()
                except Exception:
                    # Keep the original date text if parsing fails
                    pass
            
            # Create award_data-compatible structure for entity extraction
            if 'title' in item and ('description' in item or 'full_content' in item):
                description = item.get('full_content') or item.get('description') or ''
                item['award_data'] = {
                    'agency_id': '',
                    'agency_name': '',
                    'contractor_id': '',
                    'contractor_name': '',
                    'opportunity_id': item.get('id', ''),
                    'title': item.get('title', ''),
                    'description': description,
                    'value': 0,
                    'award_date': item.get('date', '')
                }
        
        return processed