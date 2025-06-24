"""
Reconnaissance Orchestrator
Coordinates and manages all reconnaissance modules
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from modules.dns_enum import DNSEnumerator
from modules.google_dorker import GoogleDorker
from modules.breach_checker import BreachChecker
from modules.social_scraper import SocialScraper
from core.rate_limiter import RateLimiter
from utils.validators import validate_target, get_target_type
from config import Config

logger = logging.getLogger(__name__)

class ReconOrchestrator:
    """Orchestrates reconnaissance across multiple modules."""
    
    def __init__(self, config: Config):
        self.config = config
        self.rate_limiter = RateLimiter(config)
        self.start_time = None
        self.results = {}
        
    async def run_reconnaissance(self, target: str, modules: List[str]) -> Dict[str, Any]:
        """Execute reconnaissance across specified modules."""
        self.start_time = datetime.now()
        
        logger.info(f"Starting reconnaissance for target: {target}")
        logger.info(f"Modules to execute: {', '.join(modules)}")
        
        # Validate target
        if not validate_target(target):
            raise ValueError(f"Invalid target format: {target}")
        
        # Determine target type
        target_type = get_target_type(target)
        logger.info(f"Target type detected: {target_type}")
        
        # Initialize results structure
        self.results = {
            'target': target,
            'target_type': target_type,
            'start_time': self.start_time.isoformat(),
            'end_time': None,
            'duration_seconds': 0,
            'modules_executed': modules,
            'results': {},
            'summary': {
                'total_findings': 0,
                'modules_successful': 0,
                'modules_failed': 0,
                'errors': []
            }
        }
        
        # Create module execution tasks
        tasks = []
        
        if 'dns' in modules:
            task = asyncio.create_task(self._execute_dns_module(target))
            tasks.append(('dns', task))
        
        if 'dorks' in modules:
            task = asyncio.create_task(self._execute_dorks_module(target, target_type))
            tasks.append(('dorks', task))
        
        if 'breach' in modules:
            task = asyncio.create_task(self._execute_breach_module(target, target_type))
            tasks.append(('breach', task))
        
        if 'social' in modules:
            task = asyncio.create_task(self._execute_social_module(target, target_type))
            tasks.append(('social', task))
        
        if 'emails' in modules:
            task = asyncio.create_task(self._execute_email_module(target, target_type))
            tasks.append(('emails', task))
        
        # Execute tasks with concurrency control
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        async def execute_with_semaphore(module_name: str, task: asyncio.Task):
            async with semaphore:
                try:
                    result = await task
                    self.results['results'][module_name] = result
                    self.results['summary']['modules_successful'] += 1
                    logger.info(f"Module '{module_name}' completed successfully")
                    return result
                except Exception as e:
                    error_msg = f"Module '{module_name}' failed: {str(e)}"
                    logger.error(error_msg)
                    self.results['summary']['errors'].append(error_msg)
                    self.results['summary']['modules_failed'] += 1
                    self.results['results'][module_name] = {
                        'error': error_msg,
                        'module': module_name
                    }
                    return None
        
        # Execute all tasks
        execution_tasks = [
            execute_with_semaphore(module_name, task)
            for module_name, task in tasks
        ]
        
        await asyncio.gather(*execution_tasks, return_exceptions=True)
        
        # Finalize results
        self._finalize_results()
        
        logger.info(f"Reconnaissance completed for {target}")
        logger.info(f"Duration: {self.results['duration_seconds']:.2f} seconds")
        logger.info(f"Successful modules: {self.results['summary']['modules_successful']}")
        logger.info(f"Failed modules: {self.results['summary']['modules_failed']}")
        
        return self.results
    
    async def _execute_dns_module(self, target: str) -> Dict[str, Any]:
        """Execute DNS enumeration module."""
        logger.debug("Executing DNS enumeration module")
        
        dns_enum = DNSEnumerator(self.rate_limiter)
        return await dns_enum.enumerate_domain(target)
    
    async def _execute_dorks_module(self, target: str, target_type: str) -> Dict[str, Any]:
        """Execute Google dorking module."""
        logger.debug("Executing Google dorking module")
        
        async with GoogleDorker(self.rate_limiter) as dorker:
            return await dorker.execute_dorks(target, target_type)
    
    async def _execute_breach_module(self, target: str, target_type: str) -> Dict[str, Any]:
        """Execute breach checking module."""
        logger.debug("Executing breach checking module")
        
        async with BreachChecker(self.rate_limiter) as breach_checker:
            if target_type == 'email':
                return await breach_checker.check_email_breaches(target)
            elif target_type == 'domain':
                return await breach_checker.check_domain_breaches(target)
            else:
                # For usernames, try email format
                if '@' not in target:
                    # Try common email formats
                    common_domains = ['gmail.com', 'yahoo.com', 'outlook.com']
                    results = {
                        'target': target,
                        'attempted_emails': [],
                        'breaches_found': [],
                        'errors': []
                    }
                    
                    for domain in common_domains:
                        email = f"{target}@{domain}"
                        try:
                            breach_result = await breach_checker.check_email_breaches(email)
                            results['attempted_emails'].append(email)
                            if breach_result.get('total_breaches', 0) > 0:
                                results['breaches_found'].append(breach_result)
                        except Exception as e:
                            results['errors'].append(f"Failed to check {email}: {str(e)}")
                    
                    return results
                else:
                    return await breach_checker.check_email_breaches(target)
    
    async def _execute_social_module(self, target: str, target_type: str) -> Dict[str, Any]:
        """Execute social media scraping module."""
        logger.debug("Executing social media scraping module")
        
        async with SocialScraper(self.rate_limiter) as social_scraper:
            return await social_scraper.search_social_media(target, target_type)
    
    async def _execute_email_module(self, target: str, target_type: str) -> Dict[str, Any]:
        """Execute email enumeration (integrated with other modules)."""
        logger.debug("Executing email enumeration module")
        
        results = {
            'target': target,
            'target_type': target_type,
            'emails_found': [],
            'sources': [],
            'total_emails': 0
        }
        
        # Email enumeration is integrated into other modules
        # This module aggregates email findings from:
        # 1. Google dorking results
        # 2. DNS TXT record analysis
        # 3. Social media profile analysis
        
        # For now, return a structure that can be populated by other modules
        # In a full implementation, this would use additional email enumeration techniques
        
        if target_type == 'domain':
            # Common email patterns for domain
            common_patterns = [
                f"admin@{target}",
                f"info@{target}",
                f"contact@{target}",
                f"support@{target}",
                f"sales@{target}",
                f"hello@{target}",
                f"webmaster@{target}"
            ]
            
            results['common_patterns'] = common_patterns
            results['note'] = "Email enumeration integrated with other modules"
        
        return results
    
    def _finalize_results(self):
        """Finalize reconnaissance results with summary information."""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        self.results['end_time'] = end_time.isoformat()
        self.results['duration_seconds'] = duration.total_seconds()
        
        # Calculate total findings
        total_findings = 0
        for module_name, module_results in self.results['results'].items():
            if isinstance(module_results, dict) and 'error' not in module_results:
                # Count findings based on module type
                if module_name == 'dns':
                    total_findings += len(module_results.get('records', {}))
                    total_findings += len(module_results.get('subdomains', []))
                elif module_name == 'dorks':
                    total_findings += module_results.get('total_results', 0)
                elif module_name == 'breach':
                    if 'total_breaches' in module_results:
                        total_findings += module_results.get('total_breaches', 0)
                    elif 'breaches_found' in module_results:
                        total_findings += len(module_results.get('breaches_found', []))
                elif module_name == 'social':
                    total_findings += len(module_results.get('profiles_found', []))
                    total_findings += len(module_results.get('mentions_found', []))
                elif module_name == 'emails':
                    total_findings += module_results.get('total_emails', 0)
        
        self.results['summary']['total_findings'] = total_findings
        
        # Generate execution summary
        self.results['summary']['execution_summary'] = (
            f"Executed {len(self.results['modules_executed'])} modules "
            f"in {duration.total_seconds():.2f} seconds. "
            f"Found {total_findings} total findings across all modules."
        )
