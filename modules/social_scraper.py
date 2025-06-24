"""
Social Media Scraper Module
Ethical social media reconnaissance for OSINT purposes
"""

import asyncio
import aiohttp
import re
import json
from typing import Dict, List, Any, Optional
import urllib.parse
import logging
from bs4 import BeautifulSoup
import random

from core.rate_limiter import RateLimiter
from config import config, USER_AGENTS, RATE_LIMITS

logger = logging.getLogger(__name__)

class SocialScraper:
    """Ethical social media scraping for OSINT."""
    
    def __init__(self, rate_limiter: RateLimiter):
        self.rate_limiter = rate_limiter
        self.session = None
        
        # Social media platforms to search
        self.platforms = {
            'twitter': {
                'search_url': 'https://twitter.com/search',
                'user_url': 'https://twitter.com/{username}',
                'rate_limit': 3.0
            },
            'linkedin': {
                'search_url': 'https://www.linkedin.com/search/results/people/',
                'user_url': 'https://www.linkedin.com/in/{username}',
                'rate_limit': 5.0
            },
            'github': {
                'search_url': 'https://github.com/search',
                'user_url': 'https://github.com/{username}',
                'api_url': 'https://api.github.com',
                'rate_limit': 2.0
            },
            'reddit': {
                'search_url': 'https://www.reddit.com/search',
                'user_url': 'https://www.reddit.com/user/{username}',
                'rate_limit': 2.0
            }
        }
    
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
    
    async def search_social_media(self, target: str, target_type: str = 'username') -> Dict[str, Any]:
        """Search across multiple social media platforms."""
        logger.info(f"Starting social media search for {target_type}: {target}")
        
        results = {
            'target': target,
            'target_type': target_type,
            'platforms_searched': [],
            'profiles_found': [],
            'mentions_found': [],
            'total_results': 0,
            'errors': []
        }
        
        # Search each platform
        search_tasks = []
        for platform_name, platform_config in self.platforms.items():
            task = asyncio.create_task(
                self._search_platform(platform_name, target, target_type)
            )
            search_tasks.append((platform_name, task))
        
        # Execute searches concurrently
        for platform_name, task in search_tasks:
            try:
                await self.rate_limiter.wait('social')
                platform_results = await task
                
                if platform_results:
                    results['platforms_searched'].append(platform_name)
                    
                    # Categorize results
                    for result in platform_results:
                        if result.get('type') == 'profile':
                            results['profiles_found'].append(result)
                        elif result.get('type') == 'mention':
                            results['mentions_found'].append(result)
                    
                    results['total_results'] += len(platform_results)
                    
            except Exception as e:
                error_msg = f"Social media search failed for {platform_name}: {str(e)}"
                logger.error(error_msg)
                results['errors'].append(error_msg)
        
        # Post-process results
        results['profiles_found'] = self._deduplicate_profiles(results['profiles_found'])
        results['mentions_found'] = self._rank_mentions(results['mentions_found'], target)
        
        logger.info(f"Social media search completed for {target}. Found {results['total_results']} results")
        return results
    
    async def _search_platform(self, platform: str, target: str, target_type: str) -> List[Dict[str, Any]]:
        """Search a specific social media platform."""
        if platform == 'github':
            return await self._search_github(target, target_type)
        elif platform == 'twitter':
            return await self._search_twitter(target, target_type)
        elif platform == 'linkedin':
            return await self._search_linkedin(target, target_type)
        elif platform == 'reddit':
            return await self._search_reddit(target, target_type)
        else:
            return []
    
    async def _search_github(self, target: str, target_type: str) -> List[Dict[str, Any]]:
        """Search GitHub for users and repositories."""
        results = []
        
        if not self.session:
            return results
        
        try:
            # Search for users
            if target_type in ['username', 'email']:
                user_results = await self._github_user_search(target)
                results.extend(user_results)
            
            # Search for repositories and code
            if target_type in ['domain', 'email', 'username']:
                repo_results = await self._github_repo_search(target)
                results.extend(repo_results)
                
                code_results = await self._github_code_search(target)
                results.extend(code_results)
        
        except Exception as e:
            logger.error(f"GitHub search failed: {e}")
        
        return results
    
    async def _github_user_search(self, target: str) -> List[Dict[str, Any]]:
        """Search GitHub users."""
        results = []
        
        # Use GitHub API if available, otherwise fall back to web scraping
        api_url = f"https://api.github.com/search/users"
        params = {'q': target}
        
        try:
            async with self.session.get(api_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    for user in data.get('items', [])[:10]:  # Limit to top 10
                        result = {
                            'platform': 'github',
                            'type': 'profile',
                            'username': user.get('login'),
                            'url': user.get('html_url'),
                            'avatar_url': user.get('avatar_url'),
                            'public_repos': user.get('public_repos'),
                            'followers': user.get('followers'),
                            'created_at': user.get('created_at'),
                            'relevance_score': self._calculate_github_relevance(user, target)
                        }
                        results.append(result)
                        
        except Exception as e:
            logger.error(f"GitHub user search failed: {e}")
        
        return results
    
    async def _github_repo_search(self, target: str) -> List[Dict[str, Any]]:
        """Search GitHub repositories."""
        results = []
        
        api_url = f"https://api.github.com/search/repositories"
        params = {'q': target}
        
        try:
            async with self.session.get(api_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    for repo in data.get('items', [])[:10]:  # Limit to top 10
                        result = {
                            'platform': 'github',
                            'type': 'repository',
                            'name': repo.get('full_name'),
                            'url': repo.get('html_url'),
                            'description': repo.get('description'),
                            'language': repo.get('language'),
                            'stars': repo.get('stargazers_count'),
                            'forks': repo.get('forks_count'),
                            'updated_at': repo.get('updated_at'),
                            'owner': repo.get('owner', {}).get('login'),
                            'relevance_score': self._calculate_repo_relevance(repo, target)
                        }
                        results.append(result)
                        
        except Exception as e:
            logger.error(f"GitHub repo search failed: {e}")
        
        return results
    
    async def _github_code_search(self, target: str) -> List[Dict[str, Any]]:
        """Search GitHub code for sensitive information."""
        results = []
        
        # Define sensitive search patterns
        sensitive_patterns = [
            f'"{target}" password',
            f'"{target}" api_key',
            f'"{target}" secret',
            f'"{target}" token',
            f'"{target}" config'
        ]
        
        api_url = f"https://api.github.com/search/code"
        
        for pattern in sensitive_patterns:
            try:
                await asyncio.sleep(1)  # GitHub API rate limiting
                
                params = {'q': pattern}
                
                async with self.session.get(api_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for item in data.get('items', [])[:5]:  # Limit results
                            result = {
                                'platform': 'github',
                                'type': 'code_mention',
                                'file_path': item.get('path'),
                                'repository': item.get('repository', {}).get('full_name'),
                                'url': item.get('html_url'),
                                'score': item.get('score'),
                                'search_pattern': pattern,
                                'relevance_score': 5.0  # High relevance for code mentions
                            }
                            results.append(result)
                    elif response.status == 422:
                        # Search query might be too broad
                        logger.debug(f"GitHub code search query too broad: {pattern}")
                        continue
                        
            except Exception as e:
                logger.error(f"GitHub code search failed for pattern '{pattern}': {e}")
                continue
        
        return results
    
    async def _search_twitter(self, target: str, target_type: str) -> List[Dict[str, Any]]:
        """Search Twitter (X) for mentions and profiles."""
        results = []
        
        # Note: Twitter's API requires authentication and has strict rate limits
        # This implementation focuses on web scraping approach
        # In production, consider using Twitter API v2 with proper authentication
        
        try:
            # Search for user profile
            if target_type == 'username':
                profile_url = f"https://twitter.com/{target}"
                profile_data = await self._scrape_twitter_profile(profile_url, target)
                if profile_data:
                    results.append(profile_data)
            
            # Note: Twitter search requires authentication for API access
            # Web scraping Twitter is challenging due to anti-bot measures
            # Consider using official Twitter API in production
            
        except Exception as e:
            logger.error(f"Twitter search failed: {e}")
        
        return results
    
    async def _scrape_twitter_profile(self, url: str, username: str) -> Optional[Dict[str, Any]]:
        """Scrape basic Twitter profile information."""
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Extract basic profile information
                    # Note: Twitter's structure changes frequently
                    profile_data = {
                        'platform': 'twitter',
                        'type': 'profile',
                        'username': username,
                        'url': url,
                        'exists': True,
                        'relevance_score': 3.0
                    }
                    
                    return profile_data
                    
        except Exception as e:
            logger.error(f"Twitter profile scraping failed: {e}")
        
        return None
    
    async def _search_linkedin(self, target: str, target_type: str) -> List[Dict[str, Any]]:
        """Search LinkedIn for profiles."""
        results = []
        
        # LinkedIn requires authentication for most operations
        # This is a basic implementation that checks profile existence
        
        try:
            if target_type == 'username':
                profile_url = f"https://www.linkedin.com/in/{target}"
                profile_data = await self._check_linkedin_profile(profile_url, target)
                if profile_data:
                    results.append(profile_data)
                    
        except Exception as e:
            logger.error(f"LinkedIn search failed: {e}")
        
        return results
    
    async def _check_linkedin_profile(self, url: str, username: str) -> Optional[Dict[str, Any]]:
        """Check if LinkedIn profile exists."""
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    return {
                        'platform': 'linkedin',
                        'type': 'profile',
                        'username': username,
                        'url': url,
                        'exists': True,
                        'relevance_score': 3.0
                    }
                    
        except Exception as e:
            logger.error(f"LinkedIn profile check failed: {e}")
        
        return None
    
    async def _search_reddit(self, target: str, target_type: str) -> List[Dict[str, Any]]:
        """Search Reddit for users and mentions."""
        results = []
        
        try:
            if target_type == 'username':
                profile_url = f"https://www.reddit.com/user/{target}"
                profile_data = await self._check_reddit_profile(profile_url, target)
                if profile_data:
                    results.append(profile_data)
                    
        except Exception as e:
            logger.error(f"Reddit search failed: {e}")
        
        return results
    
    async def _check_reddit_profile(self, url: str, username: str) -> Optional[Dict[str, Any]]:
        """Check if Reddit profile exists."""
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    return {
                        'platform': 'reddit',
                        'type': 'profile',
                        'username': username,
                        'url': url,
                        'exists': True,
                        'relevance_score': 2.0
                    }
                    
        except Exception as e:
            logger.error(f"Reddit profile check failed: {e}")
        
        return None
    
    def _calculate_github_relevance(self, user: Dict[str, Any], target: str) -> float:
        """Calculate relevance score for GitHub user."""
        score = 1.0
        
        username = user.get('login', '').lower()
        target_lower = target.lower()
        
        # Exact match
        if username == target_lower:
            score += 5.0
        elif target_lower in username:
            score += 3.0
        
        # Activity indicators
        if user.get('public_repos', 0) > 0:
            score += 1.0
        if user.get('followers', 0) > 10:
            score += 1.0
        
        return score
    
    def _calculate_repo_relevance(self, repo: Dict[str, Any], target: str) -> float:
        """Calculate relevance score for GitHub repository."""
        score = 1.0
        
        name = repo.get('full_name', '').lower()
        description = repo.get('description', '').lower() if repo.get('description') else ''
        target_lower = target.lower()
        
        # Name matches
        if target_lower in name:
            score += 3.0
        
        # Description matches
        if target_lower in description:
            score += 2.0
        
        # Activity indicators
        if repo.get('stargazers_count', 0) > 0:
            score += 1.0
        
        return score
    
    def _deduplicate_profiles(self, profiles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate profiles based on URL."""
        seen_urls = set()
        unique_profiles = []
        
        for profile in profiles:
            url = profile.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_profiles.append(profile)
        
        return unique_profiles
    
    def _rank_mentions(self, mentions: List[Dict[str, Any]], target: str) -> List[Dict[str, Any]]:
        """Rank mentions by relevance score."""
        return sorted(mentions, key=lambda x: x.get('relevance_score', 0), reverse=True)
