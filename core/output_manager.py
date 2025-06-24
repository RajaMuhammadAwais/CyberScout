"""
Output Manager
Handles formatting and output of reconnaissance results
"""

import json
import csv
import io
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import logging

try:
    from colorama import init, Fore, Back, Style
    init()
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False
    # Fallback color constants
    Fore = type('Fore', (), {
        'RED': '', 'GREEN': '', 'YELLOW': '', 'BLUE': '', 
        'MAGENTA': '', 'CYAN': '', 'WHITE': '', 'RESET': ''
    })()
    Style = type('Style', (), {'BRIGHT': '', 'RESET_ALL': ''})()

logger = logging.getLogger(__name__)

class OutputManager:
    """Manages output formatting and display."""
    
    def __init__(self):
        self.colors_enabled = COLORAMA_AVAILABLE
    
    def print_terminal_results(self, results: Dict[str, Any], target: str):
        """Print results to terminal with formatting."""
        if not results:
            print(f"{Fore.RED}No results to display{Style.RESET_ALL}")
            return
        
        # Header
        self._print_header(results)
        
        # Module results
        for module_name, module_data in results.get('results', {}).items():
            self._print_module_results(module_name, module_data)
        
        # Summary
        self._print_summary(results.get('summary', {}))
    
    def _print_header(self, results: Dict[str, Any]):
        """Print results header."""
        print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{Style.BRIGHT}OSINT RECONNAISSANCE RESULTS{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        
        print(f"{Fore.WHITE}Target:{Style.RESET_ALL} {results.get('target', 'Unknown')}")
        print(f"{Fore.WHITE}Target Type:{Style.RESET_ALL} {results.get('target_type', 'Unknown')}")
        print(f"{Fore.WHITE}Start Time:{Style.RESET_ALL} {results.get('start_time', 'Unknown')}")
        print(f"{Fore.WHITE}Duration:{Style.RESET_ALL} {results.get('duration_seconds', 0):.2f} seconds")
        print()
    
    def _print_module_results(self, module_name: str, module_data: Dict[str, Any]):
        """Print results for a specific module."""
        if 'error' in module_data:
            print(f"{Fore.RED}[{module_name.upper()}] ERROR: {module_data['error']}{Style.RESET_ALL}")
            return
        
        print(f"{Fore.YELLOW}{Style.BRIGHT}[{module_name.upper()}] RESULTS{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}{'-'*40}{Style.RESET_ALL}")
        
        if module_name == 'dns':
            self._print_dns_results(module_data)
        elif module_name == 'dorks':
            self._print_dorks_results(module_data)
        elif module_name == 'ai_dorks':
            self._print_ai_dorks_results(module_data)
        elif module_name == 'breach':
            self._print_breach_results(module_data)
        elif module_name == 'social':
            self._print_social_results(module_data)
        elif module_name == 'emails':
            self._print_email_results(module_data)
        else:
            self._print_generic_results(module_data)
        
        print()
    
    def _print_dns_results(self, data: Dict[str, Any]):
        """Print DNS enumeration results."""
        # DNS Records
        records = data.get('records', {})
        if records:
            print(f"{Fore.GREEN}DNS Records:{Style.RESET_ALL}")
            for record_type, values in records.items():
                if values:
                    print(f"  {Fore.CYAN}{record_type}:{Style.RESET_ALL}")
                    for value in values:
                        print(f"    • {value}")
        
        # Subdomains
        subdomains = data.get('subdomains', [])
        if subdomains:
            print(f"\n{Fore.GREEN}Subdomains Found ({len(subdomains)}):{Style.RESET_ALL}")
            for subdomain in subdomains[:10]:  # Limit display
                print(f"  • {subdomain.get('full_domain', 'Unknown')}")
            if len(subdomains) > 10:
                print(f"  ... and {len(subdomains) - 10} more")
        
        # Reverse DNS
        reverse_dns = data.get('reverse_dns', {})
        if reverse_dns:
            print(f"\n{Fore.GREEN}Reverse DNS:{Style.RESET_ALL}")
            for ip, hostname in reverse_dns.items():
                print(f"  {ip} → {hostname}")
        
        # Errors
        errors = data.get('errors', [])
        if errors:
            print(f"\n{Fore.RED}Errors:{Style.RESET_ALL}")
            for error in errors[:3]:  # Limit error display
                print(f"  • {error}")
    
    def _print_dorks_results(self, data: Dict[str, Any]):
        """Print Google dorking results."""
        results = data.get('results', [])
        total_results = data.get('total_results', 0)
        
        print(f"{Fore.GREEN}Total Results Found: {total_results}{Style.RESET_ALL}")
        
        if results:
            print(f"\n{Fore.GREEN}Top Results:{Style.RESET_ALL}")
            for i, result in enumerate(results[:10], 1):
                print(f"{Fore.CYAN}{i}.{Style.RESET_ALL} {result.get('title', 'Unknown Title')}")
                print(f"   URL: {result.get('url', 'Unknown URL')}")
                print(f"   Domain: {result.get('domain', 'Unknown Domain')}")
                if result.get('snippet'):
                    snippet = result['snippet'][:100] + "..." if len(result['snippet']) > 100 else result['snippet']
                    print(f"   Snippet: {snippet}")
                print()
        
        # Dorks executed
        dorks_executed = data.get('dorks_executed', [])
        if dorks_executed:
            print(f"{Fore.GREEN}Dorks Executed ({len(dorks_executed)}):{Style.RESET_ALL}")
            for dork in dorks_executed[:5]:  # Show first 5
                print(f"  • {dork}")
            if len(dorks_executed) > 5:
                print(f"  ... and {len(dorks_executed) - 5} more")
    
    def _print_ai_dorks_results(self, data: Dict[str, Any]):
        """Print AI-powered Google dorking results."""
        total_results = data.get('total_results', 0)
        intelligence_score = data.get('intelligence_score', 0.0)
        
        print(f"{Fore.GREEN}AI Intelligence Score: {intelligence_score:.2f}/1.0{Style.RESET_ALL}")
        print(f"{Fore.GREEN}Total Results Found: {total_results}{Style.RESET_ALL}")
        
        # Show AI-generated query categories
        query_categories = data.get('query_categories', {})
        if query_categories:
            print(f"\n{Fore.CYAN}AI Query Categories:{Style.RESET_ALL}")
            for category, category_data in query_categories.items():
                results_count = category_data.get('total_results', 0)
                queries_count = len(category_data.get('queries', []))
                print(f"  • {category.replace('_', ' ').title()}: {queries_count} queries → {results_count} results")
        
        # Show top high-risk results
        results = data.get('results', [])
        high_risk_results = [r for r in results if r.get('risk_level') == 'high']
        
        if high_risk_results:
            print(f"\n{Fore.RED}High-Risk Findings ({len(high_risk_results)}):{Style.RESET_ALL}")
            for result in high_risk_results[:5]:
                print(f"  • {result.get('title', 'No title')}")
                print(f"    Risk: {result.get('risk_level', 'unknown').upper()} | Score: {result.get('composite_score', 0):.2f}")
                print(f"    URL: {result.get('url', 'No URL')}")
                indicators = result.get('intelligence_indicators', [])
                if indicators:
                    print(f"    Indicators: {', '.join(indicators[:3])}")
                print()
        
        # Show intelligence indicators summary
        all_indicators = []
        for result in results:
            all_indicators.extend(result.get('intelligence_indicators', []))
            all_indicators.extend(result.get('ml_intelligence_indicators', []))
        
        if all_indicators:
            from collections import Counter
            top_indicators = Counter(all_indicators).most_common(5)
            print(f"\n{Fore.YELLOW}Top Intelligence Indicators:{Style.RESET_ALL}")
            for indicator, count in top_indicators:
                print(f"  • {indicator.replace('_', ' ').title()}: {count} occurrences")
        
        # Show ML analysis summary
        ml_confidences = [r.get('ml_confidence', 0) for r in results if r.get('ml_confidence', 0) > 0]
        if ml_confidences:
            avg_ml_confidence = sum(ml_confidences) / len(ml_confidences)
            print(f"\n{Fore.CYAN}ML Analysis Summary:{Style.RESET_ALL}")
            print(f"  • Average ML Confidence: {avg_ml_confidence:.2f}")
            
            risk_categories = [r.get('ml_risk_category', 'unknown') for r in results]
            risk_counts = Counter(risk_categories)
            for risk, count in risk_counts.most_common():
                if risk != 'unknown':
                    color = Fore.RED if risk == 'critical' else Fore.YELLOW if risk == 'high' else Fore.GREEN
                    print(f"  • {color}{risk.title()} Risk Results: {count}{Style.RESET_ALL}")
    
    def _print_breach_results(self, data: Dict[str, Any]):
        """Print breach checking results."""
        if 'total_breaches' in data:
            # Single email check
            total_breaches = data.get('total_breaches', 0)
            paste_count = data.get('paste_count', 0)
            
            print(f"{Fore.GREEN}Breaches Found: {total_breaches}{Style.RESET_ALL}")
            print(f"{Fore.GREEN}Paste Appearances: {paste_count}{Style.RESET_ALL}")
            
            breaches = data.get('breaches', [])
            if breaches:
                print(f"\n{Fore.RED}Breach Details:{Style.RESET_ALL}")
                for breach in breaches[:5]:  # Limit display
                    if isinstance(breach, dict):
                        name = breach.get('Name', 'Unknown')
                        date = breach.get('BreachDate', 'Unknown')
                        pwn_count = breach.get('PwnCount', 'Unknown')
                        print(f"  • {name} ({date}) - {pwn_count:,} accounts affected")
        
        elif 'breaches_found' in data:
            # Multiple email checks
            breaches_found = data.get('breaches_found', [])
            attempted_emails = data.get('attempted_emails', [])
            
            print(f"{Fore.GREEN}Emails Checked: {len(attempted_emails)}{Style.RESET_ALL}")
            print(f"{Fore.GREEN}Breaches Found: {len(breaches_found)}{Style.RESET_ALL}")
            
            if breaches_found:
                print(f"\n{Fore.RED}Compromised Accounts:{Style.RESET_ALL}")
                for breach_data in breaches_found:
                    email = breach_data.get('email', 'Unknown')
                    total_breaches = breach_data.get('total_breaches', 0)
                    print(f"  • {email}: {total_breaches} breaches")
        
        elif 'affected_accounts' in data:
            # Domain breach check
            total_breaches = data.get('total_breaches', 0)
            affected_accounts = data.get('affected_accounts', 0)
            
            print(f"{Fore.GREEN}Domain Breaches: {total_breaches}{Style.RESET_ALL}")
            print(f"{Fore.GREEN}Total Affected Accounts: {affected_accounts:,}{Style.RESET_ALL}")
    
    def _print_social_results(self, data: Dict[str, Any]):
        """Print social media results."""
        profiles_found = data.get('profiles_found', [])
        mentions_found = data.get('mentions_found', [])
        platforms_searched = data.get('platforms_searched', [])
        
        print(f"{Fore.GREEN}Platforms Searched: {', '.join(platforms_searched)}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}Profiles Found: {len(profiles_found)}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}Mentions Found: {len(mentions_found)}{Style.RESET_ALL}")
        
        if profiles_found:
            print(f"\n{Fore.CYAN}Profiles:{Style.RESET_ALL}")
            for profile in profiles_found[:10]:
                platform = profile.get('platform', 'Unknown')
                username = profile.get('username', 'Unknown')
                url = profile.get('url', '')
                print(f"  • {platform.title()}: {username}")
                if url:
                    print(f"    URL: {url}")
        
        if mentions_found:
            print(f"\n{Fore.CYAN}Mentions:{Style.RESET_ALL}")
            for mention in mentions_found[:5]:
                platform = mention.get('platform', 'Unknown')
                url = mention.get('url', '')
                relevance = mention.get('relevance_score', 0)
                print(f"  • {platform.title()} (Score: {relevance:.1f})")
                if url:
                    print(f"    URL: {url}")
    
    def _print_email_results(self, data: Dict[str, Any]):
        """Print email enumeration results."""
        total_emails = data.get('total_emails', 0)
        emails_found = data.get('emails_found', [])
        common_patterns = data.get('common_patterns', [])
        
        print(f"{Fore.GREEN}Emails Found: {total_emails}{Style.RESET_ALL}")
        
        if emails_found:
            print(f"\n{Fore.CYAN}Email Addresses:{Style.RESET_ALL}")
            for email in emails_found[:10]:
                print(f"  • {email}")
        
        if common_patterns:
            print(f"\n{Fore.YELLOW}Common Email Patterns:{Style.RESET_ALL}")
            for pattern in common_patterns[:5]:
                print(f"  • {pattern}")
    
    def _print_generic_results(self, data: Dict[str, Any]):
        """Print generic results for unknown modules."""
        if isinstance(data, dict):
            for key, value in data.items():
                if key not in ['error', 'target', 'target_type']:
                    print(f"{Fore.CYAN}{key.title()}:{Style.RESET_ALL} {value}")
    
    def _print_summary(self, summary: Dict[str, Any]):
        """Print execution summary."""
        print(f"{Fore.MAGENTA}{Style.BRIGHT}EXECUTION SUMMARY{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}{'-'*40}{Style.RESET_ALL}")
        
        total_findings = summary.get('total_findings', 0)
        modules_successful = summary.get('modules_successful', 0)
        modules_failed = summary.get('modules_failed', 0)
        
        print(f"{Fore.GREEN}Total Findings: {total_findings}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}Successful Modules: {modules_successful}{Style.RESET_ALL}")
        if modules_failed > 0:
            print(f"{Fore.RED}Failed Modules: {modules_failed}{Style.RESET_ALL}")
        
        execution_summary = summary.get('execution_summary', '')
        if execution_summary:
            print(f"\n{execution_summary}")
        
        errors = summary.get('errors', [])
        if errors:
            print(f"\n{Fore.RED}Errors:{Style.RESET_ALL}")
            for error in errors:
                print(f"  • {error}")
    
    def format_json(self, results: Dict[str, Any], target: str) -> str:
        """Format results as JSON."""
        try:
            return json.dumps(results, indent=2, sort_keys=True, default=str)
        except Exception as e:
            logger.error(f"JSON formatting failed: {e}")
            return json.dumps({'error': f'JSON formatting failed: {str(e)}'}, indent=2)
    
    def format_csv(self, results: Dict[str, Any], target: str) -> str:
        """Format results as CSV."""
        output = io.StringIO()
        writer = csv.writer(output)
        
        try:
            # Write header
            writer.writerow(['Module', 'Type', 'Value', 'Details', 'Timestamp'])
            
            # Process each module's results
            for module_name, module_data in results.get('results', {}).items():
                if 'error' in module_data:
                    writer.writerow([module_name, 'error', module_data['error'], '', results.get('start_time', '')])
                    continue
                
                timestamp = results.get('start_time', '')
                
                if module_name == 'dns':
                    self._write_dns_csv(writer, module_data, timestamp)
                elif module_name == 'dorks':
                    self._write_dorks_csv(writer, module_data, timestamp)
                elif module_name == 'ai_dorks':
                    self._write_ai_dorks_csv(writer, module_data, timestamp)
                elif module_name == 'breach':
                    self._write_breach_csv(writer, module_data, timestamp)
                elif module_name == 'social':
                    self._write_social_csv(writer, module_data, timestamp)
                elif module_name == 'emails':
                    self._write_email_csv(writer, module_data, timestamp)
                else:
                    writer.writerow([module_name, 'result', str(module_data), '', timestamp])
            
            return output.getvalue()
            
        except Exception as e:
            logger.error(f"CSV formatting failed: {e}")
            return f"Error,CSV formatting failed: {str(e)}\n"
    
    def _write_dns_csv(self, writer, data: Dict[str, Any], timestamp: str):
        """Write DNS results to CSV."""
        # DNS records
        for record_type, values in data.get('records', {}).items():
            for value in values:
                writer.writerow(['dns', f'record_{record_type.lower()}', value, '', timestamp])
        
        # Subdomains
        for subdomain in data.get('subdomains', []):
            domain_name = subdomain.get('full_domain', '')
            records = subdomain.get('records', {})
            writer.writerow(['dns', 'subdomain', domain_name, json.dumps(records), timestamp])
        
        # Reverse DNS
        for ip, hostname in data.get('reverse_dns', {}).items():
            writer.writerow(['dns', 'reverse_dns', ip, hostname, timestamp])
    
    def _write_dorks_csv(self, writer, data: Dict[str, Any], timestamp: str):
        """Write Google dorking results to CSV."""
        for result in data.get('results', []):
            title = result.get('title', '')
            url = result.get('url', '')
            domain = result.get('domain', '')
            snippet = result.get('snippet', '')
            relevance = result.get('relevance_score', 0)
            
            details = json.dumps({
                'domain': domain,
                'snippet': snippet,
                'relevance_score': relevance
            })
            
            writer.writerow(['dorks', 'search_result', f'{title}|{url}', details, timestamp])
    
    def _write_breach_csv(self, writer, data: Dict[str, Any], timestamp: str):
        """Write breach results to CSV."""
        if 'breaches' in data:
            for breach in data.get('breaches', []):
                if isinstance(breach, dict):
                    name = breach.get('Name', '')
                    date = breach.get('BreachDate', '')
                    count = breach.get('PwnCount', 0)
                    
                    details = json.dumps({
                        'breach_date': date,
                        'pwn_count': count,
                        'data_classes': breach.get('DataClasses', [])
                    })
                    
                    writer.writerow(['breach', 'breach_record', name, details, timestamp])
        
        if 'breaches_found' in data:
            for breach_data in data.get('breaches_found', []):
                email = breach_data.get('email', '')
                total_breaches = breach_data.get('total_breaches', 0)
                writer.writerow(['breach', 'email_breach', email, f'breaches:{total_breaches}', timestamp])
    
    def _write_social_csv(self, writer, data: Dict[str, Any], timestamp: str):
        """Write social media results to CSV."""
        for profile in data.get('profiles_found', []):
            platform = profile.get('platform', '')
            username = profile.get('username', '')
            url = profile.get('url', '')
            
            details = json.dumps({
                'url': url,
                'exists': profile.get('exists', False),
                'relevance_score': profile.get('relevance_score', 0)
            })
            
            writer.writerow(['social', f'{platform}_profile', username, details, timestamp])
        
        for mention in data.get('mentions_found', []):
            platform = mention.get('platform', '')
            url = mention.get('url', '')
            relevance = mention.get('relevance_score', 0)
            
            writer.writerow(['social', f'{platform}_mention', url, f'relevance:{relevance}', timestamp])
    
    def _write_ai_dorks_csv(self, writer, data: Dict[str, Any], timestamp: str):
        """Write AI-powered Google dorking results to CSV."""
        for result in data.get('results', []):
            title = result.get('title', '')
            url = result.get('url', '')
            risk_level = result.get('risk_level', 'unknown')
            composite_score = result.get('composite_score', 0)
            intelligence_indicators = ', '.join(result.get('intelligence_indicators', []))
            
            details = json.dumps({
                'risk_level': risk_level,
                'composite_score': composite_score,
                'relevance_score': result.get('relevance_score', 0),
                'sensitivity_score': result.get('sensitivity_score', 0),
                'intelligence_indicators': intelligence_indicators,
                'intent_category': result.get('intent_category', ''),
                'ai_rank': result.get('ai_rank', 0)
            })
            
            writer.writerow(['ai_dorks', 'search_result', title, details, timestamp])
    
    def _write_email_csv(self, writer, data: Dict[str, Any], timestamp: str):
        """Write email results to CSV."""
        for email in data.get('emails_found', []):
            writer.writerow(['emails', 'email_address', email, '', timestamp])
        
        for pattern in data.get('common_patterns', []):
            writer.writerow(['emails', 'email_pattern', pattern, 'common_pattern', timestamp])
    
    def save_to_file(self, content: str, file_path: str, format_type: str):
        """Save content to file."""
        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Results saved to {file_path} ({format_type})")
            print(f"{Fore.GREEN}Results saved to: {file_path}{Style.RESET_ALL}")
            
        except Exception as e:
            error_msg = f"Failed to save results to {file_path}: {str(e)}"
            logger.error(error_msg)
            print(f"{Fore.RED}Error: {error_msg}{Style.RESET_ALL}")
