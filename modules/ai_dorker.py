"""
AI-Powered Google Dorking Module
Intelligent automated Google search query generation and execution using machine learning
"""

import asyncio
import aiohttp
import re
import json
import random
from typing import Dict, List, Any, Optional, Tuple
import logging
from urllib.parse import quote_plus
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from core.rate_limiter import RateLimiter
from config import USER_AGENTS

logger = logging.getLogger(__name__)

class AIDorker:
    """AI-powered Google dorking with intelligent query generation."""
    
    def __init__(self, rate_limiter: RateLimiter):
        self.rate_limiter = rate_limiter
        self.session = None
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Dynamic dork patterns based on target analysis
        self.base_patterns = {
            'sensitive_files': [
                'filetype:{ext} site:{domain}',
                'inurl:{domain} filetype:{ext}',
                'site:{domain} ext:{ext}',
                '{domain} filetype:{ext} password',
                '{domain} filetype:{ext} config'
            ],
            'subdomain_discovery': [
                'site:*.{domain}',
                'inurl:{domain} -site:{domain}',
                'related:{domain}',
                'link:{domain}',
                '{domain} subdomain'
            ],
            'technology_stack': [
                '{domain} "powered by"',
                '{domain} "built with"',
                '{domain} technology stack',
                'site:{domain} framework',
                '{domain} CMS version'
            ],
            'security_issues': [
                '{domain} vulnerability',
                '{domain} security issue',
                '{domain} exploit',
                'site:{domain} error',
                '{domain} debug information'
            ],
            'data_leaks': [
                '{domain} "database" leaked',
                '{domain} exposed credentials',
                '{domain} api key',
                '{domain} secret token',
                'site:pastebin.com {domain}'
            ],
            'social_engineering': [
                '{domain} employee directory',
                '{domain} staff list',
                '{domain} contact information',
                '{domain} organizational chart',
                '{domain} email format'
            ]
        }
        
        # File extensions of interest
        self.sensitive_extensions = [
            'pdf', 'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx',
            'txt', 'log', 'conf', 'config', 'ini', 'xml', 'json',
            'sql', 'db', 'backup', 'bak', 'old', 'tmp'
        ]
        
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
    
    async def execute_ai_dorking(self, target: str, target_type: str = 'domain') -> Dict[str, Any]:
        """Execute AI-powered Google dorking for the target."""
        logger.info(f"Starting AI-powered dorking for {target_type}: {target}")
        
        results = {
            'target': target,
            'target_type': target_type,
            'ai_generated_queries': [],
            'query_categories': {},
            'results': [],
            'intelligence_score': 0.0,
            'total_results': 0,
            'errors': []
        }
        
        try:
            # Analyze target to generate intelligent queries
            intelligent_queries = await self._generate_intelligent_queries(target, target_type)
            results['ai_generated_queries'] = intelligent_queries
            
            # Execute queries with AI-driven prioritization
            for category, queries in intelligent_queries.items():
                category_results = []
                
                for query_info in queries:
                    try:
                        await self.rate_limiter.wait('google')
                        
                        query = query_info['query']
                        search_results = await self._execute_google_search(query)
                        
                        if search_results:
                            # Analyze and score results using AI
                            scored_results = await self._analyze_results_with_ai(
                                search_results, target, query_info['intent']
                            )
                            category_results.extend(scored_results)
                            
                        query_info['results_count'] = len(search_results)
                        
                    except Exception as e:
                        error_msg = f"AI dork execution failed for query '{query}': {str(e)}"
                        logger.error(error_msg)
                        results['errors'].append(error_msg)
                
                results['query_categories'][category] = {
                    'queries': queries,
                    'results': category_results,
                    'total_results': len(category_results)
                }
                results['results'].extend(category_results)
            
            # Calculate overall intelligence score
            results['intelligence_score'] = self._calculate_intelligence_score(results)
            results['total_results'] = len(results['results'])
            
            # Rank results by relevance and risk
            results['results'] = self._rank_results_by_ai(results['results'], target)
            
        except Exception as e:
            error_msg = f"AI dorking failed: {str(e)}"
            logger.error(error_msg)
            results['errors'].append(error_msg)
        
        logger.info(f"AI dorking completed for {target}. Intelligence score: {results['intelligence_score']:.2f}")
        return results
    
    async def _generate_intelligent_queries(self, target: str, target_type: str) -> Dict[str, List[Dict[str, Any]]]:
        """Generate intelligent Google dork queries using AI analysis."""
        queries = {}
        
        # Analyze target characteristics
        domain_info = self._analyze_domain_characteristics(target) if target_type == 'domain' else {}
        
        # Generate category-specific queries
        for category, patterns in self.base_patterns.items():
            category_queries = []
            
            for pattern in patterns:
                # Generate base query
                base_query = self._format_query_pattern(pattern, target, target_type)
                
                if base_query:
                    query_info = {
                        'query': base_query,
                        'pattern': pattern,
                        'intent': category,
                        'priority': self._calculate_query_priority(category, domain_info),
                        'expected_risk': self._assess_query_risk(category, pattern)
                    }
                    category_queries.append(query_info)
                
                # Generate variations using AI
                if category == 'sensitive_files':
                    for ext in self.sensitive_extensions:
                        variant_query = self._format_query_pattern(pattern, target, target_type, ext=ext)
                        if variant_query and variant_query != base_query:
                            query_info = {
                                'query': variant_query,
                                'pattern': pattern,
                                'intent': f"{category}_{ext}",
                                'priority': self._calculate_query_priority(category, domain_info, ext),
                                'expected_risk': self._assess_query_risk(category, pattern, ext)
                            }
                            category_queries.append(query_info)
            
            # Sort by priority and limit queries
            category_queries.sort(key=lambda x: x['priority'], reverse=True)
            queries[category] = category_queries[:5]  # Top 5 per category
        
        # Generate context-aware queries using NLP
        context_queries = await self._generate_context_aware_queries(target, target_type)
        if context_queries:
            queries['ai_context'] = context_queries
        
        return queries
    
    def _analyze_domain_characteristics(self, domain: str) -> Dict[str, Any]:
        """Analyze domain characteristics to inform query generation."""
        characteristics = {
            'tld': domain.split('.')[-1] if '.' in domain else '',
            'subdomain_count': len(domain.split('.')) - 2,
            'length': len(domain),
            'has_numbers': bool(re.search(r'\d', domain)),
            'has_hyphens': '-' in domain,
            'common_words': []
        }
        
        # Extract potential keywords from domain
        domain_parts = re.split(r'[.-]', domain.lower())
        common_tech_words = [
            'app', 'api', 'web', 'dev', 'test', 'admin', 'user',
            'client', 'server', 'db', 'data', 'secure', 'cloud'
        ]
        
        characteristics['common_words'] = [
            word for word in domain_parts if word in common_tech_words
        ]
        
        return characteristics
    
    def _format_query_pattern(self, pattern: str, target: str, target_type: str, **kwargs) -> Optional[str]:
        """Format query pattern with target and additional parameters."""
        try:
            if target_type == 'domain':
                formatted = pattern.format(domain=target, **kwargs)
            elif target_type == 'email':
                domain = target.split('@')[1] if '@' in target else target
                formatted = pattern.format(domain=domain, email=target, **kwargs)
            else:
                formatted = pattern.format(target=target, **kwargs)
            
            return formatted
        except (KeyError, IndexError):
            return None
    
    def _calculate_query_priority(self, category: str, domain_info: Dict[str, Any], ext: str = None) -> float:
        """Calculate query priority using AI-based scoring."""
        base_priorities = {
            'sensitive_files': 0.9,
            'security_issues': 0.8,
            'data_leaks': 0.85,
            'subdomain_discovery': 0.7,
            'technology_stack': 0.6,
            'social_engineering': 0.5
        }
        
        priority = base_priorities.get(category, 0.5)
        
        # Adjust based on domain characteristics
        if domain_info.get('has_numbers'):
            priority += 0.1  # Numbered domains might be test environments
        
        if domain_info.get('common_words'):
            priority += 0.05 * len(domain_info['common_words'])
        
        # Adjust based on file extension risk
        if ext:
            high_risk_extensions = ['sql', 'db', 'backup', 'config', 'log']
            if ext in high_risk_extensions:
                priority += 0.2
        
        return min(priority, 1.0)
    
    def _assess_query_risk(self, category: str, pattern: str, ext: str = None) -> str:
        """Assess the risk level of a query."""
        high_risk_categories = ['security_issues', 'data_leaks']
        high_risk_extensions = ['sql', 'db', 'backup', 'config']
        high_risk_patterns = ['password', 'secret', 'token', 'vulnerability']
        
        if category in high_risk_categories:
            return 'high'
        
        if ext and ext in high_risk_extensions:
            return 'high'
        
        if any(keyword in pattern.lower() for keyword in high_risk_patterns):
            return 'high'
        
        return 'medium' if category in ['sensitive_files', 'subdomain_discovery'] else 'low'
    
    async def _generate_context_aware_queries(self, target: str, target_type: str) -> List[Dict[str, Any]]:
        """Generate context-aware queries using NLP analysis."""
        context_queries = []
        
        try:
            # Analyze target using NLP
            blob = TextBlob(target)
            
            # Generate semantic variations
            if target_type == 'domain':
                domain_parts = target.replace('.', ' ').replace('-', ' ')
                blob = TextBlob(domain_parts)
                
                # Extract noun phrases for query enhancement
                noun_phrases = list(blob.noun_phrases)
                
                for phrase in noun_phrases[:3]:  # Limit to top 3
                    enhanced_query = f'"{phrase}" site:{target}'
                    context_queries.append({
                        'query': enhanced_query,
                        'pattern': 'ai_semantic',
                        'intent': 'semantic_discovery',
                        'priority': 0.6,
                        'expected_risk': 'low'
                    })
            
            # Generate industry-specific queries
            industry_keywords = self._detect_industry_context(target)
            for keyword in industry_keywords[:2]:  # Limit to top 2
                industry_query = f'{target} "{keyword}" sensitive'
                context_queries.append({
                    'query': industry_query,
                    'pattern': 'ai_industry',
                    'intent': 'industry_specific',
                    'priority': 0.7,
                    'expected_risk': 'medium'
                })
        
        except Exception as e:
            logger.debug(f"Context-aware query generation failed: {e}")
        
        return context_queries
    
    def _detect_industry_context(self, target: str) -> List[str]:
        """Detect potential industry context from target."""
        industry_mappings = {
            'bank': ['financial', 'banking', 'transaction'],
            'health': ['medical', 'patient', 'healthcare'],
            'edu': ['student', 'academic', 'research'],
            'gov': ['government', 'public', 'citizen'],
            'tech': ['software', 'development', 'api'],
            'shop': ['customer', 'order', 'payment'],
            'news': ['article', 'content', 'media']
        }
        
        detected_keywords = []
        target_lower = target.lower()
        
        for indicator, keywords in industry_mappings.items():
            if indicator in target_lower:
                detected_keywords.extend(keywords)
        
        return detected_keywords
    
    async def _execute_google_search(self, query: str, num_results: int = 10) -> List[Dict[str, Any]]:
        """Execute Google search with enhanced parsing."""
        if not self.session:
            raise RuntimeError("Session not initialized")
        
        search_url = "https://www.google.com/search"
        params = {
            'q': query,
            'num': num_results,
            'hl': 'en',
            'gl': 'us'
        }
        
        try:
            await asyncio.sleep(random.uniform(2, 5))  # Enhanced delay
            
            async with self.session.get(search_url, params=params) as response:
                if response.status == 200:
                    html = await response.text()
                    return self._parse_google_results_enhanced(html, query)
                elif response.status == 429:
                    logger.warning("Google rate limiting detected")
                    await asyncio.sleep(random.uniform(15, 30))
                    return []
                else:
                    logger.warning(f"Google search returned status {response.status}")
                    return []
        
        except Exception as e:
            logger.error(f"Google search failed: {e}")
            return []
    
    def _parse_google_results_enhanced(self, html: str, query: str) -> List[Dict[str, Any]]:
        """Enhanced Google results parsing with AI analysis."""
        from bs4 import BeautifulSoup
        
        results = []
        soup = BeautifulSoup(html, 'html.parser')
        
        # Multiple selectors for robust parsing
        selectors = [
            'div.g',
            'div[data-hveid]',
            'div.rc',
            'div.yuRUbf'
        ]
        
        result_containers = []
        for selector in selectors:
            containers = soup.select(selector)
            if containers:
                result_containers = containers
                break
        
        for container in result_containers:
            try:
                # Extract title
                title_selectors = ['h3', 'a h3', '.LC20lb', '.DKV0Md']
                title = ''
                for sel in title_selectors:
                    title_elem = container.select_one(sel)
                    if title_elem:
                        title = title_elem.get_text().strip()
                        break
                
                # Extract URL
                link_elem = container.select_one('a[href]')
                url = link_elem.get('href') if link_elem else ''
                
                # Clean Google redirect URLs
                if url.startswith('/url?'):
                    from urllib.parse import parse_qs, urlparse
                    parsed = urlparse(url)
                    if 'q' in parse_qs(parsed.query):
                        url = parse_qs(parsed.query)['q'][0]
                
                # Extract snippet
                snippet_selectors = ['.VwiC3b', '.s3v9rd', '.st', '.IsZvec']
                snippet = ''
                for sel in snippet_selectors:
                    snippet_elem = container.select_one(sel)
                    if snippet_elem:
                        snippet = snippet_elem.get_text().strip()
                        break
                
                if url and url.startswith('http'):
                    result = {
                        'title': title,
                        'url': url,
                        'snippet': snippet,
                        'domain': self._extract_domain(url),
                        'query_used': query
                    }
                    results.append(result)
            
            except Exception as e:
                logger.debug(f"Error parsing result: {e}")
                continue
        
        return results
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc
        except:
            return ''
    
    async def _analyze_results_with_ai(self, results: List[Dict[str, Any]], target: str, intent: str) -> List[Dict[str, Any]]:
        """Analyze search results using AI to extract intelligence."""
        analyzed_results = []
        
        for result in results:
            try:
                # Calculate relevance score using AI
                relevance_score = self._calculate_ai_relevance(result, target, intent)
                
                # Detect sensitive content indicators
                sensitivity_score = self._detect_sensitivity(result, intent)
                
                # Extract intelligence indicators
                intelligence_indicators = self._extract_intelligence_indicators(result, intent)
                
                analyzed_result = {
                    **result,
                    'relevance_score': relevance_score,
                    'sensitivity_score': sensitivity_score,
                    'intelligence_indicators': intelligence_indicators,
                    'risk_level': self._assess_result_risk(sensitivity_score, intelligence_indicators),
                    'intent_category': intent
                }
                
                analyzed_results.append(analyzed_result)
                
            except Exception as e:
                logger.debug(f"AI analysis failed for result: {e}")
                analyzed_results.append(result)
        
        return analyzed_results
    
    def _calculate_ai_relevance(self, result: Dict[str, Any], target: str, intent: str) -> float:
        """Calculate relevance score using AI analysis."""
        text_content = f"{result.get('title', '')} {result.get('snippet', '')}"
        
        # Keyword matching
        target_keywords = target.replace('.', ' ').replace('-', ' ').split()
        keyword_matches = sum(1 for keyword in target_keywords if keyword.lower() in text_content.lower())
        keyword_score = keyword_matches / len(target_keywords) if target_keywords else 0
        
        # Intent-specific scoring
        intent_keywords = {
            'sensitive_files': ['download', 'file', 'document', 'pdf', 'doc'],
            'security_issues': ['vulnerability', 'exploit', 'security', 'bug', 'flaw'],
            'data_leaks': ['leak', 'exposed', 'dump', 'breach', 'database'],
            'subdomain_discovery': ['subdomain', 'dns', 'domain'],
            'technology_stack': ['powered by', 'built with', 'framework', 'cms'],
            'social_engineering': ['employee', 'staff', 'contact', 'directory']
        }
        
        intent_score = 0
        if intent in intent_keywords:
            intent_matches = sum(1 for keyword in intent_keywords[intent] 
                               if keyword.lower() in text_content.lower())
            intent_score = intent_matches / len(intent_keywords[intent])
        
        # Domain relevance
        domain_score = 1.0 if target in result.get('domain', '') else 0.5
        
        # Combined score
        relevance = (keyword_score * 0.4 + intent_score * 0.4 + domain_score * 0.2)
        return min(relevance, 1.0)
    
    def _detect_sensitivity(self, result: Dict[str, Any], intent: str) -> float:
        """Detect sensitivity indicators in search results."""
        text_content = f"{result.get('title', '')} {result.get('snippet', '')}".lower()
        url_content = result.get('url', '').lower()
        
        sensitive_indicators = [
            'password', 'passwd', 'secret', 'key', 'token', 'api_key',
            'config', 'configuration', 'database', 'db', 'backup',
            'admin', 'administrator', 'root', 'login', 'auth',
            'private', 'internal', 'confidential', 'restricted',
            'vulnerability', 'exploit', 'security', 'breach'
        ]
        
        sensitivity_score = 0
        for indicator in sensitive_indicators:
            if indicator in text_content or indicator in url_content:
                sensitivity_score += 0.1
        
        # Boost score for high-risk file extensions in URL
        high_risk_extensions = ['.sql', '.db', '.bak', '.config', '.log']
        for ext in high_risk_extensions:
            if ext in url_content:
                sensitivity_score += 0.2
        
        return min(sensitivity_score, 1.0)
    
    def _extract_intelligence_indicators(self, result: Dict[str, Any], intent: str) -> List[str]:
        """Extract intelligence indicators from search results."""
        indicators = []
        text_content = f"{result.get('title', '')} {result.get('snippet', '')}".lower()
        url = result.get('url', '').lower()
        
        # File type indicators
        file_extensions = re.findall(r'\.([a-zA-Z0-9]{2,4})', url)
        if file_extensions:
            indicators.extend([f"file_type_{ext}" for ext in file_extensions])
        
        # Technology indicators
        tech_patterns = {
            'cms_wordpress': ['wordpress', 'wp-content', 'wp-admin'],
            'cms_drupal': ['drupal', 'sites/default'],
            'framework_laravel': ['laravel', 'artisan'],
            'database_mysql': ['mysql', 'phpmyadmin'],
            'server_apache': ['apache', 'httpd'],
            'server_nginx': ['nginx'],
            'cloud_aws': ['amazonaws', 'aws', 's3'],
            'version_control': ['git', 'github', 'gitlab', '.git']
        }
        
        for tech, patterns in tech_patterns.items():
            if any(pattern in text_content or pattern in url for pattern in patterns):
                indicators.append(tech)
        
        # Security indicators
        security_patterns = [
            'default_credentials', 'weak_password', 'no_auth',
            'directory_listing', 'error_disclosure', 'debug_mode'
        ]
        
        if 'index of' in text_content:
            indicators.append('directory_listing')
        if any(error in text_content for error in ['error', 'warning', 'exception']):
            indicators.append('error_disclosure')
        
        return indicators
    
    def _assess_result_risk(self, sensitivity_score: float, indicators: List[str]) -> str:
        """Assess overall risk level of a search result."""
        high_risk_indicators = [
            'file_type_sql', 'file_type_db', 'file_type_bak',
            'directory_listing', 'error_disclosure', 'default_credentials'
        ]
        
        if sensitivity_score > 0.6:
            return 'high'
        elif sensitivity_score > 0.3 or any(indicator in high_risk_indicators for indicator in indicators):
            return 'medium'
        else:
            return 'low'
    
    def _calculate_intelligence_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall intelligence score for the dorking session."""
        if not results['results']:
            return 0.0
        
        # Factor in number and quality of results
        result_count_score = min(len(results['results']) / 50, 1.0)  # Max at 50 results
        
        # Factor in average relevance and sensitivity
        avg_relevance = np.mean([r.get('relevance_score', 0) for r in results['results']])
        avg_sensitivity = np.mean([r.get('sensitivity_score', 0) for r in results['results']])
        
        # Factor in diversity of intelligence indicators
        all_indicators = []
        for result in results['results']:
            all_indicators.extend(result.get('intelligence_indicators', []))
        
        unique_indicators = len(set(all_indicators))
        indicator_diversity_score = min(unique_indicators / 20, 1.0)  # Max at 20 unique indicators
        
        # Calculate composite score
        intelligence_score = (
            result_count_score * 0.3 +
            avg_relevance * 0.3 +
            avg_sensitivity * 0.2 +
            indicator_diversity_score * 0.2
        )
        
        return intelligence_score
    
    def _rank_results_by_ai(self, results: List[Dict[str, Any]], target: str) -> List[Dict[str, Any]]:
        """Rank results using AI-based scoring."""
        def calculate_composite_score(result):
            relevance = result.get('relevance_score', 0)
            sensitivity = result.get('sensitivity_score', 0)
            indicator_count = len(result.get('intelligence_indicators', []))
            
            # Risk-weighted scoring
            risk_multiplier = {'high': 1.5, 'medium': 1.2, 'low': 1.0}
            risk_factor = risk_multiplier.get(result.get('risk_level', 'low'), 1.0)
            
            composite_score = (relevance * 0.4 + sensitivity * 0.4 + 
                             min(indicator_count / 10, 0.2) * 0.2) * risk_factor
            
            return composite_score
        
        # Sort by composite score
        ranked_results = sorted(results, key=calculate_composite_score, reverse=True)
        
        # Add ranking information
        for i, result in enumerate(ranked_results):
            result['ai_rank'] = i + 1
            result['composite_score'] = calculate_composite_score(result)
        
        return ranked_results