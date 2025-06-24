"""
Rate Limiter
Manages request rates to avoid overwhelming target services
"""

import asyncio
import time
from typing import Dict, Optional
from dataclasses import dataclass, field
import logging

from config import Config, RATE_LIMITS

logger = logging.getLogger(__name__)

@dataclass
class RateLimitState:
    """State tracking for rate limiting."""
    last_request_time: float = 0.0
    request_count: int = 0
    reset_time: float = 0.0
    rate_limit: float = 1.0

class RateLimiter:
    """Rate limiter for managing request timing across modules."""
    
    def __init__(self, config: Config):
        self.config = config
        self.states: Dict[str, RateLimitState] = {}
        self.global_rate_limit = config.rate_limit
        self.lock = asyncio.Lock()
        
        # Initialize rate limit states
        for service, rate_limit in RATE_LIMITS.items():
            self.states[service] = RateLimitState(rate_limit=rate_limit)
    
    async def wait(self, service: str, custom_rate: Optional[float] = None):
        """Wait for rate limit compliance before making a request."""
        async with self.lock:
            current_time = time.time()
            
            # Get or create rate limit state
            if service not in self.states:
                rate_limit = custom_rate or self.global_rate_limit
                self.states[service] = RateLimitState(rate_limit=rate_limit)
            
            state = self.states[service]
            
            # Calculate wait time
            time_since_last = current_time - state.last_request_time
            required_wait = state.rate_limit - time_since_last
            
            if required_wait > 0:
                logger.debug(f"Rate limiting {service}: waiting {required_wait:.2f} seconds")
                await asyncio.sleep(required_wait)
                current_time = time.time()
            
            # Update state
            state.last_request_time = current_time
            state.request_count += 1
            
            logger.debug(f"Rate limiter: {service} request #{state.request_count}")
    
    async def wait_if_needed(self, service: str, min_interval: float = None):
        """Wait only if minimum interval hasn't passed."""
        if min_interval is None:
            await self.wait(service)
            return
        
        async with self.lock:
            current_time = time.time()
            
            if service not in self.states:
                self.states[service] = RateLimitState(rate_limit=min_interval)
            
            state = self.states[service]
            time_since_last = current_time - state.last_request_time
            
            if time_since_last < min_interval:
                wait_time = min_interval - time_since_last
                logger.debug(f"Minimum interval wait for {service}: {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
                current_time = time.time()
            
            state.last_request_time = current_time
            state.request_count += 1
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get rate limiting statistics."""
        stats = {}
        current_time = time.time()
        
        for service, state in self.states.items():
            stats[service] = {
                'total_requests': state.request_count,
                'rate_limit': state.rate_limit,
                'time_since_last': current_time - state.last_request_time,
                'can_request_now': (current_time - state.last_request_time) >= state.rate_limit
            }
        
        return stats
    
    def reset_service(self, service: str):
        """Reset rate limiting state for a service."""
        if service in self.states:
            self.states[service] = RateLimitState(
                rate_limit=self.states[service].rate_limit
            )
            logger.info(f"Reset rate limiting state for service: {service}")
    
    def reset_all(self):
        """Reset all rate limiting states."""
        for service in self.states:
            self.reset_service(service)
        logger.info("Reset all rate limiting states")
    
    async def batch_wait(self, service: str, batch_size: int):
        """Wait for batch processing with adaptive rate limiting."""
        base_rate = self.states.get(service, RateLimitState()).rate_limit
        
        # Increase wait time for larger batches to be more respectful
        adaptive_rate = base_rate * (1 + (batch_size - 1) * 0.1)
        
        await self.wait(service, adaptive_rate)
    
    def set_rate_limit(self, service: str, rate_limit: float):
        """Dynamically adjust rate limit for a service."""
        if service not in self.states:
            self.states[service] = RateLimitState()
        
        self.states[service].rate_limit = rate_limit
        logger.info(f"Updated rate limit for {service}: {rate_limit} seconds")
    
    async def respect_server_limits(self, service: str, response_headers: Dict[str, str]):
        """Respect server-provided rate limiting headers."""
        # Common rate limiting headers
        rate_limit_headers = {
            'x-ratelimit-remaining': 'remaining',
            'x-ratelimit-reset': 'reset_time',
            'x-ratelimit-limit': 'limit',
            'retry-after': 'retry_after'
        }
        
        for header_name, header_type in rate_limit_headers.items():
            header_value = response_headers.get(header_name)
            if header_value:
                try:
                    value = float(header_value)
                    
                    if header_type == 'remaining' and value <= 1:
                        # If we're close to rate limit, wait longer
                        logger.warning(f"{service}: Rate limit nearly exceeded, backing off")
                        await asyncio.sleep(5.0)
                    
                    elif header_type == 'retry_after':
                        # Server explicitly told us to wait
                        logger.warning(f"{service}: Server requested retry after {value} seconds")
                        await asyncio.sleep(value)
                    
                    elif header_type == 'reset_time':
                        # Update our reset time tracking
                        if service in self.states:
                            self.states[service].reset_time = time.time() + value
                
                except ValueError:
                    logger.debug(f"Invalid rate limit header value: {header_name}={header_value}")
