#!/bin/bash

# OSINT Reconnaissance Tool - CLI Demo Script
# Demonstrates various CLI usage patterns

echo "🔍 OSINT Reconnaissance Tool - CLI Demo"
echo "======================================"
echo ""

# Basic DNS enumeration
echo "1. Basic DNS Enumeration:"
echo "Command: python main.py --target google.com --dns"
echo ""
python main.py --target google.com --dns
echo ""
echo "Press Enter to continue..."
read

# Multiple modules with JSON output
echo "2. Multiple Modules with JSON Output:"
echo "Command: python main.py --target github.com --dns --dorks --output json"
echo ""
python main.py --target github.com --dns --dorks --output json
echo ""
echo "Press Enter to continue..."
read

# Email investigation
echo "3. Email Investigation:"
echo "Command: python main.py --target test@gmail.com --breach --social"
echo ""
python main.py --target test@gmail.com --breach --social
echo ""
echo "Press Enter to continue..."
read

# All modules with rate limiting
echo "4. Comprehensive Scan with Rate Limiting:"
echo "Command: python main.py --target example.com --all --rate-limit 0.5 --verbose"
echo ""
python main.py --target example.com --all --rate-limit 0.5 --verbose
echo ""
echo "Demo completed!"
echo ""
echo "Try your own commands:"
echo "- python main.py --target [YOUR_TARGET] --dns"
echo "- python main.py --target [YOUR_TARGET] --all --output json --file results.json"
echo "- python main.py --web  # For web interface"