"""
Google Dorking Module
Advanced Google search queries for OSINT reconnaissance
"""

import asyncio
import aiohttp
import re
import urllib.parse
from typing import Dict, List, Any, Optional
import random
import logging
from bs4 import BeautifulSoup

from core.rate_limiter import RateLimiter
from config import config, USER_AGENTS, RATE_LIMITS

logger = logging.getLogger(__name__)

class GoogleDorker:
    """Google dorking for OSINT reconnaissance."""
    
    def __init__(self, rate_limiter: RateLimiter):
        self.rate_limiter = rate_limiter
        self.session = None
        self.results_cache = {}
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': random.choice(USER_AGENTS)}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def execute_dorks(self, target: str, target_type: str = 'domain') -> Dict[str, Any]:
        """Execute Google dorks for the specified target."""
        logger.info(f"Starting Google dorking for {target_type}: {target}")
        
        results = {
            'target': target,
            'target_type': target_type,
            'dorks_executed': [],
            'results': [],
            'total_results': 0,
            'errors': []
        }
        
        # Get appropriate dork profiles
        dork_templates = config.dork_profiles.get(target_type, config.dork_profiles['domain'])
        
        # Execute each dork query
        for dork_template in dork_templates:
            try:
                await self.rate_limiter.wait('google')
                
                # Format the dork with the target
                dork_query = dork_template.format(target=target)
                results['dorks_executed'].append(dork_query)
                
                logger.debug(f"Executing dork: {dork_query}")
                
                # Execute the search
                search_results = await self._execute_google_search(dork_query)
                
                if search_results:
                    for result in search_results:
                        result['dork_query'] = dork_query
                        results['results'].append(result)
                    
                    results['total_results'] += len(search_results)
                    logger.debug(f"Found {len(search_results)} results for dork: {dork_query}")
                
            except Exception as e:
                error_msg = f"Dork execution failed for query '{dork_template}': {str(e)}"
                logger.error(error_msg)
                results['errors'].append(error_msg)
        
        # Post-process results
        results['results'] = self._deduplicate_results(results['results'])
        results['results'] = self._rank_results(results['results'], target)
        results['total_results'] = len(results['results'])
        
        logger.info(f"Google dorking completed for {target}. Found {results['total_results']} unique results")
        return results
    
    async def _execute_google_search(self, query: str, num_results: int = 10) -> List[Dict[str, Any]]:
        """Execute a Google search query."""
        if not self.session:
            raise RuntimeError("Session not initialized. Use async context manager.")
        
        # Prepare search URL
        search_url = "https://www.google.com/search"
        params = {
            'q': query,
            'num': num_results,
            'hl': 'en',
            'gl': 'us'
        }
        
        try:
            # Add random delay to avoid detection
            await asyncio.sleep(random.uniform(1, 3))
            
            async with self.session.get(search_url, params=params) as response:
                if response.status == 200:
                    html = await response.text()
                    return self._parse_google_results(html, query)
                elif response.status == 429:
                    logger.warning("Google rate limiting detected. Backing off...")
                    await asyncio.sleep(random.uniform(10, 20))
                    return []
                else:
                    logger.warning(f"Google search returned status {response.status}")
                    return []
                    
        except asyncio.TimeoutError:
            logger.error(f"Timeout during Google search for query: {query}")
            return []
        except Exception as e:
            logger.error(f"Error during Google search: {e}")
            return []
    
    def _parse_google_results(self, html: str, query: str) -> List[Dict[str, Any]]:
        """Parse Google search results from HTML."""
        results = []
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find search result containers
            result_containers = soup.find_all('div', class_='g')
            
            for container in result_containers:
                try:
                    # Extract title
                    title_elem = container.find('h3')
                    title = title_elem.get_text() if title_elem else 'No title'
                    
                    # Extract URL
                    link_elem = container.find('a')
                    url = link_elem.get('href') if link_elem else ''
                    
                    # Clean URL (remove Google redirect)
                    if url.startswith('/url?'):
                        url_params = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
                        url = url_params.get('q', [''])[0]
                    
                    # Extract snippet
                    snippet_elem = container.find('span', class_='st') or container.find('div', class_='s')
                    snippet = snippet_elem.get_text() if snippet_elem else ''
                    
                    if url and url.startswith('http'):
                        result = {
                            'title': title.strip(),
                            'url': url.strip(),
                            'snippet': snippet.strip(),
                            'domain': urllib.parse.urlparse(url).netloc,
                            'relevance_score': self._calculate_relevance(title, snippet, query)
                        }
                        results.append(result)
                        
                except Exception as e:
                    logger.debug(f"Error parsing individual result: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Error parsing Google results: {e}")
        
        return results
    
    def _calculate_relevance(self, title: str, snippet: str, query: str) -> float:
        """Calculate relevance score for a search result."""
        score = 0.0
        
        # Extract search terms from query (remove site: operators, etc.)
        search_terms = re.findall(r'"([^"]*)"|\b(\w+)\b', query.lower())
        search_terms = [term[0] if term[0] else term[1] for term in search_terms 
                       if term[0] or (term[1] and not term[1] in ['site', 'filetype', 'inurl', 'intext'])]
        
        text_to_search = (title + ' ' + snippet).lower()
        
        for term in search_terms:
            if term in text_to_search:
                # Higher score for exact matches in title
                if term in title.lower():
                    score += 2.0
                else:
                    score += 1.0
        
        # Bonus for high-value sites
        high_value_domains = ['github.com', 'pastebin.com', 'stackoverflow.com', 'reddit.com']
        for domain in high_value_domains:
            if domain in text_to_search:
                score += 1.5
        
        # Bonus for security-related keywords
        security_keywords = ['password', 'api', 'key', 'token', 'leak', 'dump', 'breach', 'exposed']
        for keyword in security_keywords:
            if keyword in text_to_search:
                score += 1.0
        
        return score
    
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate results based on URL."""
        seen_urls = set()
        unique_results = []
        
        for result in results:
            url = result.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(result)
        
        return unique_results
    
    def _rank_results(self, results: List[Dict[str, Any]], target: str) -> List[Dict[str, Any]]:
        """Rank results by relevance score."""
        return sorted(results, key=lambda x: x.get('relevance_score', 0), reverse=True)
    
    async def search_specific_sites(self, target: str, sites: List[str]) -> Dict[str, Any]:
        """Search for target across specific sites."""
        results = {
            'target': target,
            'sites_searched': sites,
            'results': [],
            'errors': []
        }
        
        for site in sites:
            try:
                await self.rate_limiter.wait('google')
                
                query = f'site:{site} "{target}"'
                search_results = await self._execute_google_search(query)
                
                for result in search_results:
                    result['searched_site'] = site
                    result['dork_query'] = query
                    results['results'].append(result)
                    
            except Exception as e:
                error_msg = f"Site-specific search failed for {site}: {str(e)}"
                logger.error(error_msg)
                results['errors'].append(error_msg)
        
        results['results'] = self._deduplicate_results(results['results'])
        results['total_results'] = len(results['results'])
        
        return results
