# OSINT Reconnaissance Tool

## Overview

This is a high-performance automated OSINT (Open Source Intelligence) reconnaissance tool designed for cybersecurity professionals, threat hunters, red teamers, and bug bounty researchers. The tool automates the collection of publicly available intelligence across multiple vectors including DNS enumeration, breach checking, Google dorking, and social media reconnaissance.

The tool is built with a Python-based web interface and CLI, featuring async/concurrent processing for maximum performance. It rivals existing industry tools like SpiderFoot, Recon-ng, theHarvester, and Amass while addressing their limitations through better concurrency, modularity, and user experience.

**Current Status**: Fully functional with both CLI and web interfaces deployed and tested. The tool successfully performs multi-vector reconnaissance with real-time monitoring capabilities.

## System Architecture

### Frontend Architecture
- **Web Interface**: Flask-based web application providing a user-friendly interface
- **Static Assets**: Bootstrap 5, Font Awesome, and custom CSS/JavaScript
- **Real-time Updates**: AJAX-based task monitoring and progress tracking
- **Responsive Design**: Mobile-first approach with modern UI components

### Backend Architecture
- **Core Orchestrator**: Central coordination system managing all reconnaissance modules
- **Modular Design**: Separate modules for different OSINT techniques (DNS, social media, breaches, Google dorking)
- **Async Processing**: Python asyncio-based concurrent execution
- **Rate Limiting**: Built-in rate limiting to avoid overwhelming target services
- **Output Management**: Flexible output formatting (JSON, CSV, terminal display)

### Module Structure
- **DNS Enumeration**: Concurrent DNS lookups for various record types
- **Breach Checker**: Integration with breach databases and APIs
- **Google Dorker**: Automated Google search queries for intelligence gathering
- **Social Scraper**: Ethical social media reconnaissance across multiple platforms

## Key Components

### Core Components
1. **ReconOrchestrator** (`core/orchestrator.py`): Main coordination system
2. **RateLimiter** (`core/rate_limiter.py`): Request rate management
3. **OutputManager** (`core/output_manager.py`): Result formatting and display
4. **Config** (`config.py`): Centralized configuration management

### Reconnaissance Modules
1. **DNSEnumerator** (`modules/dns_enum.py`): DNS record enumeration
2. **BreachChecker** (`modules/breach_checker.py`): Data breach verification
3. **GoogleDorker** (`modules/google_dorker.py`): Advanced Google search queries
4. **SocialScraper** (`modules/social_scraper.py`): Social media intelligence gathering

### Utilities
1. **Logger** (`utils/logger.py`): Centralized logging configuration
2. **Validators** (`utils/validators.py`): Input validation and target type detection

### Web Interface
1. **Flask Application** (`web_interface.py`): Web server and API endpoints
2. **HTML Templates** (`templates/`): User interface templates
3. **Static Assets** (`static/`): CSS, JavaScript, and other frontend resources

## Data Flow

1. **Input Processing**: Target validation and type detection (domain, email, IP, username)
2. **Module Selection**: User selects which reconnaissance modules to execute
3. **Task Orchestration**: ReconOrchestrator coordinates module execution with rate limiting
4. **Concurrent Execution**: Multiple modules run simultaneously using async/await patterns
5. **Result Aggregation**: OutputManager collects and formats results from all modules
6. **Output Generation**: Results delivered in requested format (web UI, JSON, CSV, terminal)

## External Dependencies

### Python Packages
- **aiohttp**: Async HTTP client for web requests
- **beautifulsoup4**: HTML parsing for web scraping
- **dnspython**: DNS query functionality
- **flask**: Web framework for the user interface
- **werkzeug**: WSGI utilities for Flask
- **colorama**: Terminal color support

### External APIs
- **Google Custom Search API**: For advanced search queries (optional)
- **Have I Been Pwned API**: For breach checking (optional)
- **Twitter API**: For social media reconnaissance (optional)
- **LinkedIn API**: For professional network intelligence (optional)

### DNS Infrastructure
- Multiple DNS servers for redundancy (Google, Cloudflare, OpenDNS, Quad9)

## Deployment Strategy

### Development Environment
- **Replit Integration**: Configured for easy deployment on Replit platform
- **Python 3.11**: Modern Python version with async/await support
- **Package Management**: UV for fast package installation and dependency management

### Production Considerations
- **Environment Variables**: Secure API key management
- **Rate Limiting**: Configurable rate limits to respect service terms
- **Error Handling**: Comprehensive error handling and logging
- **Scalability**: Async architecture supports high concurrency

### Configuration
- **Modular Configuration**: Environment-based configuration system
- **API Integration**: Optional API keys for enhanced functionality
- **Performance Tuning**: Configurable timeouts, rate limits, and concurrency settings

## Recent Changes

- June 24, 2025: Complete OSINT reconnaissance tool implementation with AI capabilities
  - CLI interface with chainable reconnaissance modules (--dns --dorks --ai-dorks --breach --social --emails)
  - AI-powered Google dorking module using machine learning for intelligent query generation
  - TextBlob NLP integration for semantic analysis and context-aware search patterns
  - Dynamic risk assessment and relevance scoring without requiring databases
  - Scikit-learn integration for advanced query prioritization and result ranking
  - Web interface with real-time task monitoring and AI dorking option
  - Environment configuration system with .env file support for API keys and settings
  - DNS enumeration with concurrent record lookups and subdomain discovery
  - Breach checking integration and social media reconnaissance capabilities
  - Rate limiting and ethical scraping controls with configurable performance settings
  - Multiple output formats (JSON, CSV, terminal) with download capabilities
  - Comprehensive documentation, installation automation, and demo scripts

## User Preferences

Preferred communication style: Simple, everyday language.