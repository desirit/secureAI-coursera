#!/bin/bash
# SecureAI Course - Demo Launcher
# Quick way to run any demo

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Activate venv if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo -e "${GREEN}✅ Virtual environment activated${NC}"
else
    echo -e "${YELLOW}⚠️  No virtual environment found. Run ./setup.sh first${NC}"
    exit 1
fi

# Display menu
echo ""
echo "========================================="
echo "   SecureAI Course - Demo Launcher"
echo "========================================="
echo ""
echo "Module 1: Understanding AI Threat Models"
echo "  ${BLUE}1${NC}) m1-v2-1.py - Prompt Injection (needs Ollama)"
echo "  ${BLUE}2${NC}) m1-v2-2.py - Model Extraction"
echo "  ${BLUE}3${NC}) m1-v3-1.py - STRIDE Threat Modeling"
echo "  ${BLUE}4${NC}) m1-v3-2.py - MITRE ATLAS Framework"
echo ""
echo "Module 2: Creating Security Test Cases"
echo "  ${BLUE}5${NC}) m2-v2.py - Integration Testing"
echo "  ${BLUE}6${NC}) m2-v3.py - Adversarial Testing"
echo ""
echo "Module 3: CI/CD Integration"
echo "  ${BLUE}7${NC}) m3-v2.py - CI/CD Security Gates"
echo "  ${BLUE}8${NC}) m3-v3.py - Monitoring Dashboard"
echo ""
echo "  ${BLUE}9${NC}) Run all demos (one after another)"
echo "  ${BLUE}0${NC}) Exit"
echo ""
echo -n "Select demo to run (1-9, 0 to exit): "
read -r choice

echo ""

case $choice in
    1)
        echo -e "${GREEN}🚀 Launching Prompt Injection Demo...${NC}"
        # Check if Ollama is running
        if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo -e "${YELLOW}⚠️  Ollama not running. Starting it...${NC}"
            echo -e "${YELLOW}   (Open another terminal and run: ollama serve)${NC}"
            sleep 2
        fi
        streamlit run m1-v2-1.py
        ;;
    2)
        echo -e "${GREEN}🚀 Launching Model Extraction Demo...${NC}"
        streamlit run m1-v2-2.py
        ;;
    3)
        echo -e "${GREEN}🚀 Launching STRIDE Demo...${NC}"
        streamlit run m1-v3-1.py
        ;;
    4)
        echo -e "${GREEN}🚀 Launching MITRE ATLAS Demo...${NC}"
        streamlit run m1-v3-2.py
        ;;
    5)
        echo -e "${GREEN}🚀 Launching Integration Testing Demo...${NC}"
        streamlit run m2-v2.py
        ;;
    6)
        echo -e "${GREEN}🚀 Launching Adversarial Testing Demo...${NC}"
        streamlit run m2-v3.py
        ;;
    7)
        echo -e "${GREEN}🚀 Launching CI/CD Gates Demo...${NC}"
        streamlit run m3-v2.py
        ;;
    8)
        echo -e "${GREEN}🚀 Launching Monitoring Dashboard...${NC}"
        streamlit run m3-v3.py
        ;;
    9)
        echo -e "${GREEN}🚀 Running all demos...${NC}"
        echo -e "${YELLOW}⚠️  This will open each demo in sequence${NC}"
        echo -e "${YELLOW}   Close each browser window to proceed to next${NC}"
        echo ""
        sleep 2
        
        demos=(
            "m1-v2-1.py:Prompt Injection"
            "m1-v2-2.py:Model Extraction"
            "m1-v3-1.py:STRIDE"
            "m1-v3-2.py:MITRE ATLAS"
            "m2-v2.py:Integration Testing"
            "m2-v3.py:Adversarial Testing"
            "m3-v2.py:CI/CD Gates"
            "m3-v3.py:Monitoring"
        )
        
        for demo in "${demos[@]}"; do
            file="${demo%%:*}"
            name="${demo##*:}"
            echo -e "${BLUE}📊 Running: $name${NC}"
            streamlit run "$file"
            echo ""
        done
        
        echo -e "${GREEN}✅ All demos completed!${NC}"
        ;;
    0)
        echo -e "${BLUE}👋 Goodbye!${NC}"
        exit 0
        ;;
    *)
        echo -e "${RED}❌ Invalid choice${NC}"
        exit 1
        ;;
esac
