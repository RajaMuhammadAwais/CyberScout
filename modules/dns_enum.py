"""
DNS Enumeration Module
High-performance DNS reconnaissance with concurrent lookups
"""

import asyncio
import dns.resolver
import dns.reversename
import dns.exception
from typing import Dict, List, Any, Optional
import ipaddress
import socket
from concurrent.futures import ThreadPoolExecutor
import logging

from core.rate_limiter import RateLimiter
from config import config, TIMEOUTS

logger = logging.getLogger(__name__)

class DNSEnumerator:
    """DNS enumeration with concurrent processing."""
    
    def __init__(self, rate_limiter: RateLimiter):
        self.rate_limiter = rate_limiter
        self.resolver = dns.resolver.Resolver()
        self.resolver.timeout = TIMEOUTS['dns']
        self.resolver.lifetime = TIMEOUTS['dns'] * 2
        
        # Configure DNS servers
        if config.dns_servers:
            self.resolver.nameservers = config.dns_servers
    
    async def enumerate_domain(self, domain: str) -> Dict[str, Any]:
        """Perform comprehensive DNS enumeration on a domain."""
        logger.info(f"Starting DNS enumeration for domain: {domain}")
        
        results = {
            'domain': domain,
            'records': {},
            'subdomains': [],
            'reverse_dns': {},
            'zone_transfer': {},
            'errors': []
        }
        
        # Record types to query
        record_types = ['A', 'AAAA', 'MX', 'NS', 'TXT', 'SOA', 'CNAME', 'PTR']
        
        # Create tasks for concurrent execution
        tasks = []
        
        # Basic record lookups
        for record_type in record_types:
            task = asyncio.create_task(
                self._query_record(domain, record_type)
            )
            tasks.append((record_type, task))
        
        # Subdomain enumeration
        subdomain_task = asyncio.create_task(self._enumerate_subdomains(domain))
        
        # Zone transfer attempt
        zone_transfer_task = asyncio.create_task(self._attempt_zone_transfer(domain))
        
        # Execute all tasks concurrently
        for record_type, task in tasks:
            try:
                await self.rate_limiter.wait('dns')
                record_data = await task
                if record_data:
                    results['records'][record_type] = record_data
            except Exception as e:
                error_msg = f"Failed to query {record_type} for {domain}: {str(e)}"
                logger.error(error_msg)
                results['errors'].append(error_msg)
        
        # Handle subdomain enumeration
        try:
            subdomains = await subdomain_task
            results['subdomains'] = subdomains
            
            # Perform reverse DNS lookups for A records
            if 'A' in results['records']:
                reverse_dns_tasks = []
                for ip in results['records']['A']:
                    task = asyncio.create_task(self._reverse_dns_lookup(ip))
                    reverse_dns_tasks.append((ip, task))
                
                for ip, task in reverse_dns_tasks:
                    try:
                        await self.rate_limiter.wait('dns')
                        reverse_result = await task
                        if reverse_result:
                            results['reverse_dns'][ip] = reverse_result
                    except Exception as e:
                        logger.error(f"Reverse DNS lookup failed for {ip}: {e}")
            
        except Exception as e:
            error_msg = f"Subdomain enumeration failed: {str(e)}"
            logger.error(error_msg)
            results['errors'].append(error_msg)
        
        # Handle zone transfer
        try:
            zone_data = await zone_transfer_task
            results['zone_transfer'] = zone_data
        except Exception as e:
            logger.error(f"Zone transfer attempt failed: {e}")
        
        logger.info(f"DNS enumeration completed for {domain}")
        return results
    
    async def _query_record(self, domain: str, record_type: str) -> Optional[List[str]]:
        """Query a specific DNS record type."""
        def _sync_query():
            try:
                answers = self.resolver.resolve(domain, record_type)
                return [str(answer) for answer in answers]
            except dns.resolver.NXDOMAIN:
                return None
            except dns.resolver.NoAnswer:
                return None
            except dns.exception.Timeout:
                logger.warning(f"DNS timeout for {domain} {record_type}")
                return None
            except Exception as e:
                logger.error(f"DNS query error for {domain} {record_type}: {e}")
                return None
        
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            return await loop.run_in_executor(executor, _sync_query)
    
    async def _enumerate_subdomains(self, domain: str) -> List[Dict[str, Any]]:
        """Enumerate subdomains using common prefixes."""
        common_subdomains = [
            'www', 'mail', 'ftp', 'localhost', 'webmail', 'smtp', 'pop', 'ns1', 'webdisk',
            'ns2', 'cpanel', 'whm', 'autodiscover', 'autoconfig', 'test', 'dev', 'staging',
            'admin', 'api', 'blog', 'shop', 'forum', 'help', 'support', 'docs', 'portal',
            'vpn', 'secure', 'ssl', 'ftp2', 'server', 'ns', 'email', 'imap', 'pop3',
            'mx', 'exchange', 'app', 'mobile', 'cdn', 'static', 'media', 'img', 'video'
        ]
        
        subdomains = []
        tasks = []
        
        for subdomain in common_subdomains:
            full_domain = f"{subdomain}.{domain}"
            task = asyncio.create_task(self._check_subdomain(full_domain))
            tasks.append((subdomain, full_domain, task))
        
        for subdomain, full_domain, task in tasks:
            try:
                await self.rate_limiter.wait('dns')
                result = await task
                if result:
                    subdomains.append({
                        'subdomain': subdomain,
                        'full_domain': full_domain,
                        'records': result
                    })
            except Exception as e:
                logger.error(f"Subdomain check failed for {full_domain}: {e}")
        
        return subdomains
    
    async def _check_subdomain(self, domain: str) -> Optional[Dict[str, List[str]]]:
        """Check if a subdomain exists and get its records."""
        def _sync_check():
            try:
                # Try A record first
                a_records = self.resolver.resolve(domain, 'A')
                result = {'A': [str(record) for record in a_records]}
                
                # Also try CNAME
                try:
                    cname_records = self.resolver.resolve(domain, 'CNAME')
                    result['CNAME'] = [str(record) for record in cname_records]
                except:
                    pass
                
                return result
            except:
                return None
        
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            return await loop.run_in_executor(executor, _sync_check)
    
    async def _reverse_dns_lookup(self, ip: str) -> Optional[str]:
        """Perform reverse DNS lookup for an IP address."""
        def _sync_reverse():
            try:
                # Validate IP address
                ipaddress.ip_address(ip)
                reverse_name = dns.reversename.from_address(ip)
                answers = self.resolver.resolve(reverse_name, 'PTR')
                return str(answers[0])
            except Exception:
                return None
        
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            return await loop.run_in_executor(executor, _sync_reverse)
    
    async def _attempt_zone_transfer(self, domain: str) -> Dict[str, Any]:
        """Attempt DNS zone transfer (usually fails but worth trying)."""
        def _sync_zone_transfer():
            result = {
                'attempted': True,
                'successful': False,
                'nameservers_tested': [],
                'error': None
            }
            
            try:
                # Get nameservers for the domain
                ns_answers = self.resolver.resolve(domain, 'NS')
                nameservers = [str(ns) for ns in ns_answers]
                result['nameservers_tested'] = nameservers
                
                # Try zone transfer on each nameserver
                for ns in nameservers:
                    try:
                        zone = dns.zone.from_xfr(dns.query.xfr(ns, domain))
                        result['successful'] = True
                        result['records'] = []
                        
                        for name, node in zone.nodes.items():
                            for rdataset in node.rdatasets:
                                for rdata in rdataset:
                                    result['records'].append({
                                        'name': str(name),
                                        'type': dns.rdatatype.to_text(rdataset.rdtype),
                                        'data': str(rdata)
                                    })
                        break
                    except Exception as e:
                        continue
                
                if not result['successful']:
                    result['error'] = "Zone transfer refused by all nameservers (expected)"
                
            except Exception as e:
                result['error'] = f"Zone transfer failed: {str(e)}"
            
            return result
        
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            return await loop.run_in_executor(executor, _sync_zone_transfer)
