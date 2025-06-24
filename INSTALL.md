# Quick Installation Guide

## One-Line Install (Linux/macOS)

```bash
chmod +x setup.sh && ./setup.sh
```

## Manual Installation

### 1. Install Python Dependencies
```bash
pip install aiohttp beautifulsoup4 dnspython flask werkzeug colorama
```

### 2. Start the Tool

**Web Interface** (Recommended):
```bash
python main.py --web
```
Open http://localhost:5000

**Command Line**:
```bash
python main.py --target example.com --dns --dorks
```

## Quick Test

```bash
# Test DNS enumeration
python main.py --target google.com --dns

# Test with web interface
python main.py --web
```

## Troubleshooting

**Permission Error**:
```bash
chmod +x main.py setup.sh
```

**Module Not Found**:
```bash
pip install --upgrade pip
pip install -r requirements.txt  # If file exists
```

**Port Already in Use**:
```bash
# Check what's using port 5000
lsof -i :5000
```

For detailed documentation, see [README.md](README.md)