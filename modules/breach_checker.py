"""
Breach Checker Module
Check for data breaches and exposed credentials using public APIs
"""

import asyncio
import aiohttp
import hashlib
import json
from typing import Dict, List, Any, Optional
import logging

from core.rate_limiter import RateLimiter
from config import config, USER_AGENTS

logger = logging.getLogger(__name__)

class BreachChecker:
    """Check for data breaches and exposed credentials."""
    
    def __init__(self, rate_limiter: RateLimiter):
        self.rate_limiter = rate_limiter
        self.session = None
        # API endpoints
        self.hibp_api_base = "https://haveibeenpwned.com/api/v3"
        self.pwnedpasswords_api = "https://api.pwnedpasswords.com"
        self.leaklookup_api_base = "https://leak-lookup.com/api"
        self.leaklookup_api_key = config.leaklookup_api_key
    async def check_leaklookup_email(self, email: str) -> Dict[str, Any]:
        """Check Leak-Lookup API for email breaches (if API key is set)."""
        results = {
            'email': email,
            'leaklookup_results': [],
            'errors': []
        }
        if not self.leaklookup_api_key:
            results['errors'].append('Leak-Lookup API key not set.')
            return results
        await self.rate_limiter.wait('breach')
        url = f"{self.leaklookup_api_base}/search"
        headers = {
            'Authorization': self.leaklookup_api_key,
            'Content-Type': 'application/json',
            'User-Agent': 'OSINT-Reconnaissance-Tool'
        }
        payload = {"type": "email", "query": email}
        try:
            async with self.session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    results['leaklookup_results'] = data
                else:
                    results['errors'].append(f"Leak-Lookup API returned status {response.status}")
        except Exception as e:
            results['errors'].append(f"Leak-Lookup API error: {str(e)}")
        return results
        
    async def __aenter__(self):
        """Async context manager entry."""
        headers = {
            'User-Agent': 'OSINT-Reconnaissance-Tool'
        }
        
        # Add HIBP API key if available
        if config.hibp_api_key:
            headers['hibp-api-key'] = config.hibp_api_key
        
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers=headers
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def check_email_breaches(self, email: str) -> Dict[str, Any]:
        """Check if email appears in known data breaches."""
        logger.info(f"Checking breaches for email: {email}")
        
        results = {
            'email': email,
            'breaches': [],
            'paste_count': 0,
            'total_breaches': 0,
            'breach_details': [],
            'errors': []
        }
        
        try:
            # Check Have I Been Pwned
            breach_data = await self._check_hibp_breaches(email)
            if breach_data:
                results['breaches'] = breach_data
                results['total_breaches'] = len(breach_data)
                
                # Get detailed information for each breach
                for breach in breach_data:
                    if isinstance(breach, dict) and 'Name' in breach:
                        details = await self._get_breach_details(breach['Name'])
                        if details:
                            results['breach_details'].append(details)
            
            # Check pastes
            paste_count = await self._check_hibp_pastes(email)
            results['paste_count'] = paste_count
            
        except Exception as e:
            error_msg = f"Breach check failed for {email}: {str(e)}"
            logger.error(error_msg)
            results['errors'].append(error_msg)
        
        return results
    
    async def check_domain_breaches(self, domain: str) -> Dict[str, Any]:
        """Check for breaches affecting a specific domain."""
        logger.info(f"Checking domain breaches for: {domain}")
        
        results = {
            'domain': domain,
            'breaches': [],
            'total_breaches': 0,
            'affected_accounts': 0,
            'errors': []
        }
        
        try:
            # Get all breaches and filter by domain
            all_breaches = await self._get_all_breaches()
            
            domain_breaches = []
            for breach in all_breaches:
                if self._is_domain_affected(breach, domain):
                    domain_breaches.append(breach)
            
            results['breaches'] = domain_breaches
            results['total_breaches'] = len(domain_breaches)
            
            # Calculate total affected accounts
            total_accounts = sum(breach.get('PwnCount', 0) for breach in domain_breaches)
            results['affected_accounts'] = total_accounts
            
        except Exception as e:
            error_msg = f"Domain breach check failed for {domain}: {str(e)}"
            logger.error(error_msg)
            results['errors'].append(error_msg)
        
        return results
    
    async def check_password_exposure(self, password: str) -> Dict[str, Any]:
        """Check if a password has been exposed in breaches (using k-anonymity)."""
        logger.info("Checking password exposure using Pwned Passwords API")
        
        results = {
            'exposed': False,
            'exposure_count': 0,
            'hash_prefix': '',
            'errors': []
        }
        
        try:
            # Hash the password using SHA-1
            sha1_hash = hashlib.sha1(password.encode('utf-8')).hexdigest().upper()
            hash_prefix = sha1_hash[:5]
            hash_suffix = sha1_hash[5:]
            
            results['hash_prefix'] = hash_prefix
            
            # Query the Pwned Passwords API using k-anonymity
            await self.rate_limiter.wait('breach')
            
            url = f"{self.pwnedpasswords_api}/range/{hash_prefix}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    response_text = await response.text()
                    
                    # Parse response to find matching hash
                    for line in response_text.split('\n'):
                        if ':' in line:
                            suffix, count = line.strip().split(':')
                            if suffix == hash_suffix:
                                results['exposed'] = True
                                results['exposure_count'] = int(count)
                                break
                else:
                    error_msg = f"Pwned Passwords API returned status {response.status}"
                    logger.error(error_msg)
                    results['errors'].append(error_msg)
                    
        except Exception as e:
            error_msg = f"Password exposure check failed: {str(e)}"
            logger.error(error_msg)
            results['errors'].append(error_msg)
        
        return results
    
    async def _check_hibp_breaches(self, email: str) -> Optional[List[Dict[str, Any]]]:
        """Check Have I Been Pwned API for email breaches."""
        if not self.session:
            raise RuntimeError("Session not initialized")
        
        await self.rate_limiter.wait('breach')
        
        url = f"{self.hibp_api_base}/breachedaccount/{email}"
        params = {'truncateResponse': 'false'}
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 404:
                    # No breaches found
                    return []
                elif response.status == 429:
                    logger.warning("HIBP rate limit exceeded")
                    await asyncio.sleep(10)
                    return []
                else:
                    logger.error(f"HIBP API returned status {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"HIBP breach check failed: {e}")
            return []
    
    async def _check_hibp_pastes(self, email: str) -> int:
        """Check Have I Been Pwned API for paste appearances."""
        if not self.session:
            raise RuntimeError("Session not initialized")
        
        await self.rate_limiter.wait('breach')
        
        url = f"{self.hibp_api_base}/pasteaccount/{email}"
        
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    pastes = await response.json()
                    return len(pastes) if pastes else 0
                elif response.status == 404:
                    return 0
                else:
                    logger.error(f"HIBP paste API returned status {response.status}")
                    return 0
                    
        except Exception as e:
            logger.error(f"HIBP paste check failed: {e}")
            return 0
    
    async def _get_breach_details(self, breach_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific breach."""
        if not self.session:
            raise RuntimeError("Session not initialized")
        
        await self.rate_limiter.wait('breach')
        
        url = f"{self.hibp_api_base}/breach/{breach_name}"
        
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"HIBP breach details API returned status {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"HIBP breach details failed: {e}")
            return None
    
    async def _get_all_breaches(self) -> List[Dict[str, Any]]:
        """Get all breaches from HIBP."""
        if not self.session:
            raise RuntimeError("Session not initialized")
        
        await self.rate_limiter.wait('breach')
        
        url = f"{self.hibp_api_base}/breaches"
        
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"HIBP all breaches API returned status {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"HIBP all breaches failed: {e}")
            return []
    
    def _is_domain_affected(self, breach: Dict[str, Any], domain: str) -> bool:
        """Check if a breach affects the specified domain."""
        breach_domain = breach.get('Domain', '').lower()
        target_domain = domain.lower()
        
        # Direct domain match
        if breach_domain == target_domain:
            return True
        
        # Check if breach affects subdomains
        if breach_domain.endswith(f'.{target_domain}'):
            return True
        
        # Check breach title and description for domain mentions
        title = breach.get('Title', '').lower()
        description = breach.get('Description', '').lower()
        
        if target_domain in title or target_domain in description:
            return True
        
        return False
    
    async def search_leak_databases(self, query: str) -> Dict[str, Any]:
        """Search public leak databases (implement additional sources as needed)."""
        results = {
            'query': query,
            'sources_searched': [],
            'results': [],
            'total_results': 0,
            'errors': []
        }
        
        # This would be expanded to include additional leak databases
        # For now, we focus on HIBP which is the most reliable
        
        # Add placeholder for future expansion
        results['sources_searched'] = ['hibp']
        results['note'] = 'Additional leak database sources can be integrated here'
        
        return results
