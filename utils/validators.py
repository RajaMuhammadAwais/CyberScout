"""
Input Validation Utilities
Validates targets, formats, and other user inputs
"""

import re
import ipaddress
from typing import Optional
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)

def validate_target(target: str) -> bool:
    """Validate if target is in a supported format."""
    if not target or not isinstance(target, str):
        return False
    
    target = target.strip()
    
    # Check if it's a valid domain
    if is_valid_domain(target):
        return True
    
    # Check if it's a valid email
    if is_valid_email(target):
        return True
    
    # Check if it's a valid IP address
    if is_valid_ip(target):
        return True
    
    # Check if it's a valid username (alphanumeric with some special chars)
    if is_valid_username(target):
        return True
    
    # Check if it's a valid URL
    if is_valid_url(target):
        return True
    
    return False

def get_target_type(target: str) -> str:
    """Determine the type of target."""
    if not target:
        return 'unknown'
    
    target = target.strip()
    
    if is_valid_email(target):
        return 'email'
    elif is_valid_domain(target):
        return 'domain'
    elif is_valid_ip(target):
        return 'ip'
    elif is_valid_url(target):
        return 'url'
    elif is_valid_username(target):
        return 'username'
    else:
        return 'unknown'

def is_valid_domain(domain: str) -> bool:
    """Check if string is a valid domain name."""
    if not domain or len(domain) > 253:
        return False
    
    # Remove trailing dot if present
    if domain.endswith('.'):
        domain = domain[:-1]
    
    # Domain regex pattern
    domain_pattern = re.compile(
        r'^(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)*[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?$'
    )
    
    if not domain_pattern.match(domain):
        return False
    
    # Check each label
    labels = domain.split('.')
    for label in labels:
        if not label or len(label) > 63:
            return False
        if label.startswith('-') or label.endswith('-'):
            return False
    
    # Must have at least one dot for a proper domain
    return '.' in domain

def is_valid_email(email: str) -> bool:
    """Check if string is a valid email address."""
    if not email:
        return False
    
    # Basic email regex pattern
    email_pattern = re.compile(
        r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    )
    
    if not email_pattern.match(email):
        return False
    
    # Split and validate parts
    try:
        local, domain = email.rsplit('@', 1)
        
        # Local part checks
        if not local or len(local) > 64:
            return False
        
        # Domain part checks
        if not is_valid_domain(domain):
            return False
        
        return True
    except ValueError:
        return False

def is_valid_ip(ip: str) -> bool:
    """Check if string is a valid IP address."""
    try:
        ipaddress.ip_address(ip)
        return True
    except ValueError:
        return False

def is_valid_username(username: str) -> bool:
    """Check if string is a valid username."""
    if not username:
        return False
    
    # Username should be 3-30 characters, alphanumeric with some special chars
    username_pattern = re.compile(r'^[a-zA-Z0-9._-]{3,30}$')
    
    return username_pattern.match(username) is not None

def is_valid_url(url: str) -> bool:
    """Check if string is a valid URL."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False

def validate_output_format(format_type: str) -> bool:
    """Validate output format."""
    valid_formats = ['json', 'csv', 'terminal', 'all']
    return format_type in valid_formats

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file system operations."""
    # Remove or replace dangerous characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip(' .')
    
    # Limit length
    if len(sanitized) > 200:
        sanitized = sanitized[:200]
    
    # Ensure it's not empty
    if not sanitized:
        sanitized = 'output'
    
    return sanitized

def validate_port(port: str) -> bool:
    """Validate if port number is valid."""
    try:
        port_num = int(port)
        return 1 <= port_num <= 65535
    except ValueError:
        return False

def validate_cidr(cidr: str) -> bool:
    """Validate CIDR notation."""
    try:
        ipaddress.ip_network(cidr, strict=False)
        return True
    except ValueError:
        return False

def extract_domain_from_url(url: str) -> Optional[str]:
    """Extract domain from URL."""
    try:
        parsed = urlparse(url)
        return parsed.netloc.lower()
    except Exception:
        return None

def extract_domain_from_email(email: str) -> Optional[str]:
    """Extract domain from email address."""
    try:
        if '@' in email:
            return email.split('@')[1].lower()
    except Exception:
        pass
    return None

def normalize_target(target: str) -> str:
    """Normalize target for consistent processing."""
    if not target:
        return target
    
    target = target.strip().lower()
    
    # Remove protocol from URLs
    if target.startswith(('http://', 'https://')):
        target = extract_domain_from_url(target) or target
    
    # Remove trailing dots from domains
    if target.endswith('.') and is_valid_domain(target[:-1]):
        target = target[:-1]
    
    return target

def validate_regex_pattern(pattern: str) -> bool:
    """Validate if string is a valid regex pattern."""
    try:
        re.compile(pattern)
        return True
    except re.error:
        return False

def is_private_ip(ip: str) -> bool:
    """Check if IP address is private."""
    try:
        ip_obj = ipaddress.ip_address(ip)
        return ip_obj.is_private
    except ValueError:
        return False

def is_public_domain(domain: str) -> bool:
    """Check if domain is likely public (not localhost, private, etc.)."""
    if not domain:
        return False
    
    domain = domain.lower()
    
    # Check for localhost variations
    localhost_patterns = [
        'localhost',
        '127.0.0.1',
        '::1',
        '0.0.0.0'
    ]
    
    if domain in localhost_patterns:
        return False
    
    # Check for private domain patterns
    private_patterns = [
        '.local',
        '.internal',
        '.corp',
        '.lan'
    ]
    
    for pattern in private_patterns:
        if domain.endswith(pattern):
            return False
    
    # Check if it resolves to a private IP (would need DNS lookup)
    # For now, just return True for valid domains
    return is_valid_domain(domain)

def validate_dork_query(query: str) -> bool:
    """Validate Google dork query for safety."""
    if not query:
        return False
    
    # Check for potentially harmful operators
    dangerous_patterns = [
        'site:file://',
        'site:ftp://',
        'inurl:admin',
        'inurl:login',
        'inurl:password'
    ]
    
    query_lower = query.lower()
    
    # Allow these patterns as they're legitimate for OSINT
    # but log them for awareness
    for pattern in dangerous_patterns:
        if pattern in query_lower:
            logger.info(f"Potentially sensitive dork query detected: {pattern}")
    
    # Basic validation - ensure it's not too long and has reasonable content
    if len(query) > 500:
        return False
    
    # Must contain some actual search terms
    if len(query.strip()) < 3:
        return False
    
    return True
