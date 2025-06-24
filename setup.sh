#!/bin/bash

# OSINT Reconnaissance Tool - Setup Script
# This script sets up the OSINT reconnaissance tool on your system

set -e

echo "🔍 OSINT Reconnaissance Tool Setup"
echo "=================================="

# Check Python version
echo "Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    echo "Please install Python 3.11 or higher and try again."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "✓ Python $PYTHON_VERSION detected"

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "❌ main.py not found. Please run this script from the project root directory."
    exit 1
fi

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip3 install aiohttp beautifulsoup4 dnspython flask werkzeug colorama

echo "✓ Dependencies installed successfully"

# Make the main script executable
chmod +x main.py

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "Usage Options:"
echo ""
echo "1. Web Interface (Recommended for beginners):"
echo "   python3 main.py --web"
echo "   Then open http://localhost:5000 in your browser"
echo ""
echo "2. Command Line Interface:"
echo "   python3 main.py --target example.com --dns --dorks"
echo ""
echo "3. View all options:"
echo "   python3 main.py --help"
echo ""
echo "📖 For detailed usage instructions, see README.md"
echo ""
echo "⚠️  Remember to use this tool ethically and legally!"
echo "   - Only test systems you own or have permission to test"
echo "   - Respect rate limits and terms of service"
echo "   - Follow local laws and regulations"