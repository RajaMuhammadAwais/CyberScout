# OSINT Reconnaissance Tool

A high-performance automated OSINT (Open Source Intelligence) reconnaissance tool designed for cybersecurity professionals, threat hunters, red teamers, and bug bounty researchers. This tool automates the collection of publicly available intelligence across multiple vectors including DNS enumeration, breach checking, Google dorking, and social media reconnaissance.

## Features


### 🚀 Core Capabilities
- **Multi-Vector Reconnaissance**: DNS, Google dorking, breach checking, social media, and email enumeration
- **Concurrent Processing**: Async/await architecture for maximum performance
- **Dual Interface**: Both CLI and web-based interfaces
- **Rate Limiting**: Built-in ethical scraping controls
- **Multiple Output Formats**: JSON, CSV, and formatted terminal output
- **Real-time Monitoring**: Web interface with live task tracking
- **AI-Powered Enrichment**: DeepSeek integration for report summarization, threat scoring, and enrichment (web & CLI)

### 🔍 Reconnaissance Modules
- **DNS Enumeration**: A, MX, NS, TXT records, subdomain discovery, reverse DNS
- **Google Dorking**: Advanced search queries with relevance scoring
- **Breach Checking**: Integration with Have I Been Pwned API and Leak-Lookup API
- **Social Media**: Profile discovery across GitHub, Twitter, LinkedIn, Reddit
- **Email Enumeration**: Email discovery and validation techniques
- **AI Enrichment**: DeepSeek-powered summarization, threat scoring, and advice
### 🔒 Security Best Practices
- `.env` and sensitive files are excluded from git via `.gitignore`.
- API keys and secrets are never hardcoded—use environment variables only.
- Rate limiting and ethical mode are enforced by default.
### 🤖 AI Features (DeepSeek)
- **/api/summary/<task_id>**: Get an AI-generated summary of any completed task (web & CLI)
- **/api/threat_score/<task_id>**: Get an AI-generated risk/threat score and justification (web & CLI)
- **/api/deepseek_enrich**: General AI enrichment/summarization endpoint (web)
- **--summarize-task <task_id>**: CLI option for AI summary
- **--threat-score <task_id>**: CLI option for AI threat scoring
- **--deepseek-enrich "text"**: CLI option for general enrichment

## Quick Start

### Prerequisites
- Python 3.11 or higher
- Internet connection for external API calls

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd osint-reconnaissance-tool
```

2. **Install dependencies**:
```bash
pip install aiohttp beautifulsoup4 dnspython flask werkzeug colorama
```

3. **Run the tool**:

**Web Interface**:
```bash
python main.py --web
```
Then open http://localhost:5000 in your browser.

**CLI Interface**:
```bash
python main.py --target example.com --dns --dorks
```

## Usage

### Command Line Interface

#### Basic Usage
```bash
# DNS enumeration only
python main.py --target example.com --dns

# Multiple modules
python main.py --target example.com --dns --dorks --breach

# All modules with JSON output
python main.py --target example.com --all --output json --file results.json

# Email-specific reconnaissance
python main.py --target user@domain.com --breach --social
```

#### Available Options
```
Target Options:
  --target TARGET        Domain, email, IP, or username to investigate

Reconnaissance Modules:
  --dns                  DNS enumeration (A, MX, NS, TXT records)
  --dorks               Google dorking for intelligence gathering
  --breach              Data breach and credential exposure checking
  --social              Social media profile discovery
  --emails              Email enumeration and validation techniques
  --all                 Run all available modules

Output Options:
  --output FORMAT       Output format: json, csv, terminal, all
  --file FILE           Save output to file

Performance Options:
  --rate-limit SECONDS  Delay between requests (default: 1.0)
  --timeout SECONDS     Request timeout (default: 30)
  --max-concurrent NUM  Max concurrent requests (default: 10)

Logging Options:
  --verbose             Enable detailed logging
  --quiet               Suppress output except results
  --log-file FILE       Save logs to file

Interface Options:
  --web                 Start web interface on port 5000
```

### Web Interface

1. **Start the web server**:
```bash
python main.py --web
```

2. **Open your browser** to http://localhost:5000

3. **Configure reconnaissance**:
   - Enter target (domain, email, username, or IP)
   - Select reconnaissance modules
   - Adjust advanced options if needed
   - Click "Start Reconnaissance"

4. **Monitor progress** in the Active Tasks section

5. **View results** when complete, with options to download in JSON or CSV format

## Configuration


### Environment Variables & Security

All sensitive configuration is managed via `.env` (never commit this file!). Example:
```env
# OSINT Reconnaissance Tool - Environment Configuration
RATE_LIMIT=1.0
TIMEOUT=30
MAX_CONCURRENT=10
GOOGLE_API_KEY=your-google-api-key
HIBP_API_KEY=your-hibp-api-key
LEAKLOOKUP_API_KEY=your-leaklookup-api-key
DEEPSEEK_API_KEY=your-deepseek-api-key
TWITTER_BEARER_TOKEN=your-twitter-token
LINKEDIN_SESSION=your-linkedin-session
SECRET_KEY=your-flask-secret
```
**Never share or commit your `.env` file.**

### Proxy Support

You can route all outbound HTTP(S) requests through a proxy or a pool of rotating proxies for anonymity and to avoid rate limiting.

**Single Proxy:**
Add to your `.env`:
```
HTTP_PROXY=http://your-proxy:port
HTTPS_PROXY=http://your-proxy:port
```

**Rotating Proxies:**
Add to your `.env`:
```
PROXY_LIST=http://proxy1:port,http://proxy2:port,http://proxy3:port
```
The tool will randomly select a proxy from this list for each request (supported in Google dorking, breach, and social modules).

> **Note:** For best results, use high-quality, reliable proxies. Free/public proxies may be slow or blocked by target services.

**Example Usage:**
```bash
export HTTP_PROXY=http://myproxy:port
python main.py --target example.com --dorks
```
Or for rotating proxies:
```bash
export PROXY_LIST=http://proxy1:port,http://proxy2:port
python main.py --target example.com --dorks
```

## .gitignore

This project includes a `.gitignore` that excludes:
- Python cache and build files
- Virtual environments
- .env and secrets
- Output/results
- VSCode and Jupyter files

### Rate Limiting

The tool includes built-in rate limiting to respect service terms:
- DNS queries: 0.1 seconds between requests
- Google searches: 1.0 seconds between requests
- Social media: 2.0 seconds between requests
- Breach API calls: 5.0 seconds between requests

Adjust with `--rate-limit` option or in the web interface.

## Examples

### CLI Examples

**Domain reconnaissance**:
```bash
python main.py --target example.com --dns --dorks --output json --file domain_recon.json
```

**Email investigation**:
```bash
python main.py --target john.doe@company.com --breach --social --verbose
```

**Comprehensive scan**:
```bash
python main.py --target example.com --all --rate-limit 0.5 --max-concurrent 15
```

**Quick DNS check**:
```bash
python main.py --target example.com --dns --quiet
```

### Expected Output

**Terminal Output**:
```
🔍 OSINT Reconnaissance Tool
==================================================
Target: example.com
Target Type: domain
Started: 2025-06-24 12:00:00

[DNS] RESULTS
----------------------------------------
DNS Records:
  A:
    • 93.184.216.34
  MX:
    • 10 mail.example.com
  NS:
    • ns1.example.com
    • ns2.example.com

Subdomains Found (3):
  • www.example.com
  • mail.example.com
  • blog.example.com

[DORKS] RESULTS
----------------------------------------
Total Results Found: 15

Top Results:
1. Example Domain Configuration
   URL: https://github.com/user/config
   Domain: github.com

EXECUTION SUMMARY
----------------------------------------
Total Findings: 25
Successful Modules: 2
Duration: 12.5 seconds
```

**JSON Output**:
```json
{
  "target": "example.com",
  "target_type": "domain",
  "start_time": "2025-06-24T12:00:00",
  "duration_seconds": 12.5,
  "results": {
    "dns": {
      "records": {
        "A": ["93.184.216.34"],
        "MX": ["10 mail.example.com"]
      },
      "subdomains": [...]
    },
    "dorks": {
      "total_results": 15,
      "results": [...]
    }
  },
  "summary": {
    "total_findings": 25,
    "modules_successful": 2
  }
}
```

## Project Structure

```
osint-reconnaissance-tool/
├── main.py                 # Main CLI entry point
├── web_interface.py        # Flask web application
├── config.py              # Configuration management
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── LICENSE                # MIT license
├── core/                  # Core system components
│   ├── orchestrator.py    # Main coordination system
│   ├── output_manager.py  # Result formatting
│   └── rate_limiter.py    # Request rate management
├── modules/               # Reconnaissance modules
│   ├── dns_enum.py        # DNS enumeration
│   ├── google_dorker.py   # Google dorking
│   ├── breach_checker.py  # Breach checking
│   └── social_scraper.py  # Social media reconnaissance
├── utils/                 # Utility functions
│   ├── logger.py          # Logging configuration
│   └── validators.py      # Input validation
├── templates/             # Web interface templates
│   └── index.html         # Main web page
└── static/                # Web assets
    ├── css/style.css      # Styles
    └── js/app.js          # Frontend JavaScript
```

## Security & Ethics

### Responsible Use
This tool is designed for legitimate security research, penetration testing, and OSINT gathering. Users must:
- Obtain proper authorization before testing systems
- Respect rate limits and terms of service
- Follow local laws and regulations
- Use findings responsibly


### Built-in Safeguards
- Rate limiting to prevent service overload
- No automated exploitation attempts
- Read-only reconnaissance operations
- Respectful API usage patterns
- Sensitive files excluded from git


### Privacy Considerations
- Only uses publicly available information
- No credential harvesting or unauthorized access
- Supports proxy configurations for anonymity
- Logs can be disabled for sensitive operations
- API keys and secrets are never logged

## Performance Optimization

### Concurrent Processing
- Async/await architecture for I/O operations
- Configurable concurrency limits
- Module-specific rate limiting
- Efficient resource utilization

### Memory Management
- Streaming result processing
- Configurable timeout values
- Garbage collection optimization
- Large dataset handling

## Troubleshooting

### Common Issues

**Import Errors**:
```bash
pip install -r requirements.txt
```

**Rate Limiting**:
```bash
# Increase delays between requests
python main.py --target example.com --dns --rate-limit 2.0
```

**DNS Resolution Issues**:
```bash
# Check network connectivity
nslookup example.com
```

**Web Interface Not Loading**:
```bash
# Check if port 5000 is available
netstat -an | grep 5000
```

### Debug Mode
```bash
# Enable verbose logging
python main.py --target example.com --dns --verbose --log-file debug.log
```

### Performance Issues
```bash
# Reduce concurrency for slower systems
python main.py --target example.com --all --max-concurrent 5 --rate-limit 2.0
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add your enhancements
4. Include tests for new functionality
5. Submit a pull request

### Adding New Modules
1. Create module in `modules/` directory
2. Implement async reconnaissance methods
3. Add to `core/orchestrator.py`
4. Update CLI arguments in `main.py`
5. Add web interface integration

## API Integration

### Adding API Keys

For enhanced functionality, configure these optional APIs:

**Have I Been Pwned**:
1. Visit https://haveibeenpwned.com/API/v3
2. Subscribe to API access
3. Set `HIBP_API_KEY` environment variable

**Google Custom Search**:
1. Create project in Google Cloud Console
2. Enable Custom Search API
3. Set `GOOGLE_API_KEY` environment variable

## Deployment

### Docker (Future Enhancement)
```bash
# Build container
docker build -t osint-recon .

# Run web interface
docker run -p 5000:5000 osint-recon --web

# Run CLI
docker run osint-recon --target example.com --dns
```

### Production Considerations
- Use production WSGI server (gunicorn/uwsgi)
- Configure reverse proxy (nginx/apache)
- Set up proper logging
- Implement authentication if needed
- Configure rate limiting at infrastructure level

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Changelog

### Version 1.0.0 (June 2025)
- Initial release
- CLI and web interfaces
- DNS enumeration module
- Google dorking module
- Breach checking integration
- Social media reconnaissance
- Rate limiting and ethical controls
- JSON/CSV output formats
- Real-time web monitoring

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Review the troubleshooting section
- Check existing documentation
- Follow responsible disclosure for security issues

## Acknowledgments

- Have I Been Pwned API for breach data
- DNS Python library for DNS operations
- aiohttp for async HTTP operations
- Flask for web interface
- Bootstrap for UI components
- Font Awesome for icons

---

**Disclaimer**: This tool is for authorized security testing and research only. Users are responsible for compliance with applicable laws and regulations.