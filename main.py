def threat_score_cli():
    parser = argparse.ArgumentParser(description='DeepSeek AI-powered threat scoring CLI')
    parser.add_argument('--threat-score', type=str, help='Task ID to get AI threat score')
    args, unknown = parser.parse_known_args()
    if args.threat_score:
        from core.output_manager import OutputManager
        from web_interface import active_tasks, task_results
        task_id = args.threat_score
        task = active_tasks.get(task_id)
        if not task:
            print(f"[Error] Task {task_id} not found.")
            exit(1)
        if task.status != 'completed':
            print(f"[Error] Task {task_id} not completed yet.")
            exit(1)
        results = task_results.get(task_id)
        if not results:
            print(f"[Error] Results for task {task_id} not found.")
            exit(1)
        prompt = (
            f"Analyze the following OSINT reconnaissance results for target '{task.target}'. "
            f"Assign a risk/threat score from 1 (low) to 10 (critical) and provide a short justification.\nResults: {results}"
        )
        try:
            ai_response = deepseek_complete(prompt, max_tokens=256)
            print("\n[DeepSeek AI Threat Score]\n" + ai_response)
        except Exception as e:
            print(f"[DeepSeek AI Error] {e}")
        exit(0)
import argparse
from utils.deepseek import deepseek_complete

# Legacy CLI entrypoints for DeepSeek AI features (for backward compatibility)
def threat_score_cli():
    parser = argparse.ArgumentParser(description='DeepSeek AI-powered threat scoring CLI')
    parser.add_argument('--threat-score', type=str, help='Task ID to get AI threat score')
    args, unknown = parser.parse_known_args()
    if args.threat_score:
        from core.output_manager import OutputManager
        from web_interface import active_tasks, task_results
        task_id = args.threat_score
        task = active_tasks.get(task_id)
        if not task:
            print(f"[Error] Task {task_id} not found.")
            exit(1)
        if task.status != 'completed':
            print(f"[Error] Task {task_id} not completed yet.")
            exit(1)
        results = task_results.get(task_id)
        if not results:
            print(f"[Error] Results for task {task_id} not found.")
            exit(1)
        prompt = (
            f"Analyze the following OSINT reconnaissance results for target '{task.target}'. "
            f"Assign a risk/threat score from 1 (low) to 10 (critical) and provide a short justification.\nResults: {results}"
        )
        try:
            ai_response = deepseek_complete(prompt, max_tokens=256)
            print("\n[DeepSeek AI Threat Score]\n" + ai_response)
        except Exception as e:
            print(f"[DeepSeek AI Error] {e}")
        exit(0)

def summarize_task_cli():
    parser = argparse.ArgumentParser(description='DeepSeek AI-powered task summary CLI')
    parser.add_argument('--summarize-task', type=str, help='Task ID to summarize with DeepSeek AI')
    args, unknown = parser.parse_known_args()
    if args.summarize_task:
        from core.output_manager import OutputManager
        from web_interface import active_tasks, task_results
        task_id = args.summarize_task
        task = active_tasks.get(task_id)
        if not task:
            print(f"[Error] Task {task_id} not found.")
            exit(1)
        if task.status != 'completed':
            print(f"[Error] Task {task_id} not completed yet.")
            exit(1)
        results = task_results.get(task_id)
        if not results:
            print(f"[Error] Results for task {task_id} not found.")
            exit(1)
        prompt = f"Summarize the following OSINT reconnaissance results for target '{task.target}'. Highlight key findings, risks, and recommended actions.\nResults: {results}"
        try:
            summary = deepseek_complete(prompt, max_tokens=512)
            print("\n[DeepSeek AI Summary]\n" + summary)
        except Exception as e:
            print(f"[DeepSeek AI Error] {e}")
        exit(0)

def deepseek_enrich_cli():
    parser = argparse.ArgumentParser(description='DeepSeek AI enrichment CLI')
    parser.add_argument('--deepseek-enrich', type=str, help='Text to enrich/summarize with DeepSeek AI')
    args, unknown = parser.parse_known_args()
    if args.deepseek_enrich:
        try:
            result = deepseek_complete(args.deepseek_enrich)
            print("\n[DeepSeek AI Result]\n" + result)
        except Exception as e:
            print(f"[DeepSeek AI Error] {e}")
        exit(0)
def summarize_task_cli():
    parser = argparse.ArgumentParser(description='DeepSeek AI-powered task summary CLI')
    parser.add_argument('--summarize-task', type=str, help='Task ID to summarize with DeepSeek AI')
    args, unknown = parser.parse_known_args()
    if args.summarize_task:
        from core.output_manager import OutputManager
        from web_interface import active_tasks, task_results
        task_id = args.summarize_task
        task = active_tasks.get(task_id)
        if not task:
            print(f"[Error] Task {task_id} not found.")
            exit(1)
        if task.status != 'completed':
            print(f"[Error] Task {task_id} not completed yet.")
            exit(1)
        results = task_results.get(task_id)
        if not results:
            print(f"[Error] Results for task {task_id} not found.")
            exit(1)
        prompt = f"Summarize the following OSINT reconnaissance results for target '{task.target}'. Highlight key findings, risks, and recommended actions.\nResults: {results}"
        try:
            summary = deepseek_complete(prompt, max_tokens=512)
            print("\n[DeepSeek AI Summary]\n" + summary)
        except Exception as e:
            print(f"[DeepSeek AI Error] {e}")
        exit(0)
if __name__ == '__main__':
    summarize_task_cli()
    # ...existing code...
from utils.deepseek import deepseek_complete
import argparse
def deepseek_enrich_cli():
    parser = argparse.ArgumentParser(description='DeepSeek AI enrichment CLI')
    parser.add_argument('--deepseek-enrich', type=str, help='Text to enrich/summarize with DeepSeek AI')
    args, unknown = parser.parse_known_args()
    if args.deepseek_enrich:
        try:
            result = deepseek_complete(args.deepseek_enrich)
            print("\n[DeepSeek AI Result]\n" + result)
        except Exception as e:
            print(f"[DeepSeek AI Error] {e}")
        exit(0)
if __name__ == '__main__':
    deepseek_enrich_cli()
    # ...existing code...
#!/usr/bin/env python3
"""
OSINT Reconnaissance Tool
A high-performance automated OSINT reconnaissance tool for cybersecurity professionals.
"""

import argparse
import asyncio
import sys
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

from core.orchestrator import ReconOrchestrator
from core.output_manager import OutputManager
from utils.logger import setup_logger
from utils.validators import validate_target, validate_output_format
from config import Config

def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Advanced OSINT Reconnaissance Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --target example.com --dns --emails
  %(prog)s --target example.com --all --output json --file results.json
  %(prog)s --target example.com --dorks --social --verbose
  %(prog)s --target john.doe --emails --breach --rate-limit 2
        """
    )
    
    # Target specification
    parser.add_argument(
        '--target', '-t',
        help='Target domain, email, or username for reconnaissance'
    )
    
    # Module selection (chainable)
    parser.add_argument(
        '--dns',
        action='store_true',
        help='Perform DNS enumeration (A, MX, NS, TXT records)'
    )
    
    parser.add_argument(
        '--emails',
        action='store_true',
        help='Search for email addresses related to target'
    )
    
    parser.add_argument(
        '--dorks',
        action='store_true',
        help='Execute Google dorks for intelligence gathering'
    )
    
    parser.add_argument(
        '--ai-dorks',
        action='store_true',
        help='Execute AI-powered intelligent Google dorking'
    )
    
    parser.add_argument(
        '--social',
        action='store_true',
        help='Scrape social media for target information'
    )
    
    parser.add_argument(
        '--breach',
        action='store_true',
        help='Check for data breaches and exposed credentials'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all available reconnaissance modules'
    )
    
    # Output configuration
    parser.add_argument(
        '--output', '-o',
        choices=['json', 'csv', 'terminal', 'all'],
        default='terminal',
        help='Output format (default: terminal)'
    )
    
    parser.add_argument(
        '--file', '-f',
        help='Output file path (optional)'
    )
    
    # Performance and behavior
    parser.add_argument(
        '--rate-limit',
        type=float,
        default=1.0,
        help='Rate limit in seconds between requests (default: 1.0)'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=30,
        help='Request timeout in seconds (default: 30)'
    )
    
    parser.add_argument(
        '--max-concurrent',
        type=int,
        default=10,
        help='Maximum concurrent requests (default: 10)'
    )
    
    # Logging and debug
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress all output except results'
    )
    
    parser.add_argument(
        '--log-file',
        help='Log file path (optional)'
    )
    
    # Web interface
    parser.add_argument(
        '--web',
        action='store_true',
        help='Start web interface on port 5000'
    )
    
    return parser

def validate_arguments(args: argparse.Namespace) -> bool:
    """Validate command line arguments."""
    # Validate target
    if not validate_target(args.target):
        print(f"Error: Invalid target format: {args.target}")
        return False
    
    # Validate output format
    if not validate_output_format(args.output):
        print(f"Error: Invalid output format: {args.output}")
        return False
    
    # Check if at least one module is selected
    modules_selected = any([
        args.dns, args.emails, args.dorks, getattr(args, 'ai_dorks', False),
        args.social, args.breach, args.all
    ])
    
    if not modules_selected and not args.web:
        print("Error: At least one reconnaissance module must be selected")
        return False
    
    # Validate file path if specified
    if args.file:
        try:
            Path(args.file).parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Error: Cannot create output directory: {e}")
            return False
    
    return True

async def run_reconnaissance(args: argparse.Namespace) -> Dict[str, Any]:
    """Run the reconnaissance process."""
    # Setup configuration
    config = Config(
        rate_limit=args.rate_limit,
        timeout=args.timeout,
        max_concurrent=args.max_concurrent,
        verbose=args.verbose
    )
    
    # Initialize orchestrator
    orchestrator = ReconOrchestrator(config)
    
    # Determine which modules to run
    modules_to_run = []
    if args.all:
        modules_to_run = ['dns', 'emails', 'dorks', 'ai_dorks', 'social', 'breach']
    else:
        if args.dns:
            modules_to_run.append('dns')
        if args.emails:
            modules_to_run.append('emails')
        if args.dorks:
            modules_to_run.append('dorks')
        if getattr(args, 'ai_dorks', False):
            modules_to_run.append('ai_dorks')
        if args.social:
            modules_to_run.append('social')
        if args.breach:
            modules_to_run.append('breach')
    
    # Execute reconnaissance
    results = await orchestrator.run_reconnaissance(
        target=args.target,
        modules=modules_to_run
    )
    
    return results

def start_web_interface():
    """Start the web interface."""
    from web_interface import app
    
    config = Config()
    print("Starting OSINT Reconnaissance Tool Web Interface...")
    print(f"Access at: http://localhost:{config.web_port}")
    app.run(host=config.web_host, port=config.web_port, debug=config.debug)

async def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Handle web interface
    if args.web:
        start_web_interface()
        return
    
    # Validate arguments
    if not validate_arguments(args):
        sys.exit(1)
    
    # Setup logging
    logger = setup_logger(
        verbose=args.verbose,
        quiet=args.quiet,
        log_file=args.log_file
    )
    
    try:
        if not args.quiet:
            print("🔍 OSINT Reconnaissance Tool")
            print("=" * 50)
            print(f"Target: {args.target}")
            print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print()
        
        # Run reconnaissance
        results = await run_reconnaissance(args)
        
        # Handle output
        output_manager = OutputManager()
        
        if args.output in ['terminal', 'all']:
            output_manager.print_terminal_results(results, args.target)
        
        if args.output in ['json', 'all']:
            json_output = output_manager.format_json(results, args.target)
            if args.file:
                output_manager.save_to_file(json_output, args.file, 'json')
            elif args.output == 'json':
                print(json_output)
        
        if args.output in ['csv', 'all']:
            csv_output = output_manager.format_csv(results, args.target)
            if args.file:
                csv_file = args.file.replace('.json', '.csv') if args.file.endswith('.json') else f"{args.file}.csv"
                output_manager.save_to_file(csv_output, csv_file, 'csv')
            elif args.output == 'csv':
                print(csv_output)
        
        if not args.quiet:
            print(f"\n✅ Reconnaissance completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
    except KeyboardInterrupt:
        if not args.quiet:
            print("\n❌ Reconnaissance interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Reconnaissance failed: {e}")
        if not args.quiet:
            print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n❌ Program interrupted")
        sys.exit(1)
