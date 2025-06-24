"""
Configuration management for OSINT Reconnaissance Tool
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Any

@dataclass
class Config:
    """Configuration class for the OSINT tool."""
    
    # Performance settings
    rate_limit: float = 1.0
    timeout: int = 30
    max_concurrent: int = 10
    
    # Logging settings
    verbose: bool = False
    
    # API configurations
    google_api_key: str = None
    hibp_api_key: str = None
    
    # Social media configurations
    twitter_bearer_token: str = None
    linkedin_session: str = None
    
    # DNS settings
    dns_servers: List[str] = None
    
    # Google dorking settings
    dork_profiles: Dict[str, List[str]] = None
    
    def __post_init__(self):
        """Initialize configuration with environment variables and defaults."""
        # Load API keys from environment
        self.google_api_key = os.getenv('GOOGLE_API_KEY', '')
        self.hibp_api_key = os.getenv('HIBP_API_KEY', '')
        self.twitter_bearer_token = os.getenv('TWITTER_BEARER_TOKEN', '')
        self.linkedin_session = os.getenv('LINKEDIN_SESSION', '')
        
        # Load other configuration from environment
        self.rate_limit = float(os.getenv('RATE_LIMIT', self.rate_limit))
        self.timeout = int(os.getenv('TIMEOUT', self.timeout))
        self.max_concurrent = int(os.getenv('MAX_CONCURRENT', self.max_concurrent))
        self.verbose = os.getenv('VERBOSE', '').lower() in ('true', '1', 'yes')
        self.output_dir = os.getenv('OUTPUT_DIR', self.output_dir)
        self.ethical_mode = os.getenv('ETHICAL_MODE', 'true').lower() in ('true', '1', 'yes')
        self.web_host = os.getenv('WEB_HOST', self.web_host)
        self.web_port = int(os.getenv('WEB_PORT', self.web_port))
        self.debug = os.getenv('DEBUG', '').lower() in ('true', '1', 'yes')
        
        # Load DNS servers from environment
        dns_env = os.getenv('DNS_SERVERS', '8.8.8.8,1.1.1.1,208.67.222.222,9.9.9.9')
        if dns_env:
            self.dns_servers = [server.strip() for server in dns_env.split(',')]
        
        # Set default DNS servers
        if self.dns_servers is None:
            self.dns_servers = [
                '8.8.8.8',      # Google
                '1.1.1.1',      # Cloudflare
                '208.67.222.222', # OpenDNS
                '9.9.9.9'       # Quad9
            ]
        
        # Set default Google dork profiles
        if self.dork_profiles is None:
            self.dork_profiles = {
                'domain': [
                    'site:pastebin.com "{target}"',
                    'site:github.com "{target}"',
                    'site:stackoverflow.com "{target}"',
                    'site:reddit.com "{target}"',
                    'inurl:"{target}" filetype:pdf',
                    'inurl:"{target}" filetype:doc',
                    'inurl:"{target}" filetype:xls',
                    '"@{target}" filetype:txt',
                    '"{target}" "password" filetype:log',
                    '"{target}" "api key" OR "api_key"'
                ],
                'email': [
                    '"{target}" site:pastebin.com',
                    '"{target}" site:github.com',
                    '"{target}" "password" OR "leaked"',
                    '"{target}" filetype:sql',
                    '"{target}" intext:"email" OR intext:"mail"'
                ],
                'username': [
                    '"{target}" site:twitter.com',
                    '"{target}" site:linkedin.com',
                    '"{target}" site:facebook.com',
                    '"{target}" site:instagram.com',
                    '"{target}" site:github.com',
                    '"{target}" "profile" OR "account"'
                ]
            }

# Global configuration instance
config = Config()

# Rate limiting settings
RATE_LIMITS = {
    'google': 1.0,      # 1 second between Google searches
    'social': 2.0,      # 2 seconds between social media requests
    'dns': 0.1,         # 0.1 second between DNS queries
    'breach': 5.0       # 5 seconds between breach API calls
}

# User agents for web scraping
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0'
]

# Timeout settings
TIMEOUTS = {
    'dns': 5,
    'http': 10,
    'social': 15,
    'breach': 20
}

# Output formats
OUTPUT_FORMATS = {
    'json': {
        'indent': 2,
        'sort_keys': True
    },
    'csv': {
        'delimiter': ',',
        'quotechar': '"'
    }
}
