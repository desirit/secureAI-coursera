#!/bin/bash
# SecureAI Course - Complete Setup Script
# Run this to set up your environment from scratch

set -e  # Exit on error

echo "========================================="
echo "SecureAI Course - Environment Setup"
echo "========================================="
echo ""

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

if ! python3 -c 'import sys; assert sys.version_info >= (3, 8)' 2>/dev/null; then
    echo "❌ Error: Python 3.8+ required"
    exit 1
fi
echo "✅ Python version OK"
echo ""

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ -d "venv" ]; then
    echo "⚠️  venv directory already exists. Remove it? (y/n)"
    read -r response
    if [ "$response" = "y" ]; then
        rm -rf venv
        echo "Removed old venv"
    else
        echo "Keeping existing venv"
    fi
fi

if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Using existing virtual environment"
fi
echo ""

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate
echo "✅ Virtual environment activated"
echo ""

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo "✅ pip upgraded"
echo ""

# Install requirements
echo "📥 Installing requirements..."
pip install -r requirements.txt
echo "✅ Requirements installed"
echo ""

# Check Ollama
echo "🤖 Checking Ollama installation..."
if command -v ollama &> /dev/null; then
    echo "✅ Ollama is installed"
    
    # Check if llama2-uncensored is available
    if ollama list | grep -q "llama2-uncensored"; then
        echo "✅ llama2-uncensored model is installed"
    else
        echo "⚠️  llama2-uncensored not found"
        echo "   Run: ollama pull llama2-uncensored"
    fi
    
    # Check if Ollama is running
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✅ Ollama service is running"
    else
        echo "⚠️  Ollama service not running"
        echo "   Run: ollama serve"
    fi
else
    echo "⚠️  Ollama not installed"
    echo "   Install from: https://ollama.ai"
    echo "   Required for: m1-v2-1.py (Prompt Injection demo)"
    echo "   Other demos work without Ollama"
fi
echo ""

# Test imports
echo "🧪 Testing Python imports..."
python3 << EOF
try:
    import streamlit
    import requests
    import pandas
    import numpy
    import plotly
    from PIL import Image
    print("✅ All core dependencies import successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)
EOF
echo ""

# Summary
echo "========================================="
echo "✅ Setup Complete!"
echo "========================================="
echo ""
echo "📁 Project structure:"
echo "   venv/              - Virtual environment"
echo "   m1-v2-1.py         - Prompt Injection (needs Ollama)"
echo "   m1-v2-2.py         - Model Extraction"
echo "   m1-v3-1.py         - STRIDE Threat Modeling"
echo "   m1-v3-2.py         - MITRE ATLAS Framework"
echo "   m2-v2.py           - Integration Testing"
echo "   m2-v3.py           - Adversarial Testing"
echo "   m3-v2.py           - CI/CD Security Gates"
echo "   m3-v3.py           - Monitoring Dashboard"
echo ""
echo "🚀 Quick start:"
echo "   source venv/bin/activate"
echo "   streamlit run m1-v2-1.py"
echo ""
echo "📚 For more info, see README.md"
echo ""
