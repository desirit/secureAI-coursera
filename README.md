# SecureAI Course - Screen Share Demo Files

Complete set of Streamlit demonstrations for the SecureAI: Threat Model & Test Endpoints course.

## 📦 Package Contents

### Module 1: Understanding AI-Specific Threat Models
- **m1-v2-1.py** - Prompt Injection Attack Demo
- **m1-v2-2.py** - Model Extraction Pattern Demo
- **m1-v3-1.py** - STRIDE Threat Modeling
- **m1-v3-2.py** - MITRE ATLAS Framework

### Module 2: Creating Security Test Cases
- **m2-v2.py** - Integration Testing Demo
- **m2-v3.py** - Adversarial Testing with ART

### Module 3: CI/CD Integration
- **m3-v2.py** - CI/CD Security Gates
- **m3-v3.py** - Continuous Monitoring Dashboard

## 🚀 Quick Start

### Prerequisites

```bash
# 1. Install Python 3.8+
python --version

# 2. Install Ollama (for AI demos)
# Visit: https://ollama.ai

# 3. Pull uncensored model
ollama pull llama2-uncensored

# 4. Start Ollama service
ollama serve
```

### Installation

**Option 1: Automated Setup (Recommended)**

```bash
# Linux/Mac
chmod +x setup.sh
./setup.sh

# Windows
setup.bat
```

**Option 2: Manual Setup**

```bash
# 1. Create virtual environment
python3 -m venv venv

# 2. Activate virtual environment
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install all dependencies
pip install -r requirements.txt
```

**Option 3: Quick Install (without venv)**

```bash
pip install streamlit requests pandas numpy plotly pillow
```

### Running Demos

```bash
# Run any demo
streamlit run m1-v2-1.py

# Run with custom port
streamlit run m1-v2-1.py --server.port 8502

# Run in fullscreen mode for recording
# Press F11 in browser after launching
```

## 📚 Demo Descriptions

### Module 1 - Video 2 - Part 1: Prompt Injection (m1-v2-1.py)

**What it demonstrates:**
- Normal query vs. direct attack vs. social engineering
- Why traditional security tools miss prompt injection
- Real-time attack execution with llama2-uncensored
- Data leakage detection and analysis

**Key Features:**
- Pre-defined attack scenarios
- Custom query input
- Security analysis of responses
- Educational content about prompt injection

**Usage Tips for Recording:**
- Use preset attack buttons for consistent demonstrations
- Show the system prompt in sidebar to explain vulnerability
- Highlight the security analysis section showing data leakage

### Module 1 - Video 2 - Part 2: Model Extraction (m1-v2-2.py)

**What it demonstrates:**
- Normal user traffic vs. systematic attacker patterns
- Visual comparison of query patterns
- Attack economics (cost vs. model value)
- Statistical anomaly detection

**Key Features:**
- Live traffic simulation
- Pattern visualization with Plotly
- Attack signature detection
- Economic impact analysis

**Usage Tips for Recording:**
- Start with normal traffic only, then enable attacker
- Use "Slow (Demo)" speed for clear visualization
- Emphasize the systematic variations pattern
- Show the cost analysis (pennies to steal millions)

### Module 1 - Video 3 - Part 1: STRIDE (m1-v3-1.py)

**What it demonstrates:**
- STRIDE framework applied to AI systems
- Interactive threat modeling
- Risk assessment and prioritization
- Comprehensive report generation

**Key Features:**
- Pre-configured AI system examples
- Six STRIDE categories with AI-specific threats
- Risk scoring calculator
- JSON export for threat models

**Usage Tips for Recording:**
- Use "Medical Diagnostic AI" as example system
- Document at least 2-3 threats per STRIDE category
- Show risk assessment with scoring
- Export final report

### Module 1 - Video 3 - Part 2: MITRE ATLAS (m1-v3-2.py)

**What it demonstrates:**
- MITRE ATLAS attack lifecycle
- Real-world case studies
- Tactic-to-technique mapping
- Attack chain visualization

**Key Features:**
- Complete ATLAS taxonomy
- Interactive tactic explorer
- Real-world case studies (Tay, Tesla, GPT-3)
- System-specific threat mapping

**Usage Tips for Recording:**
- Start with "Attack Lifecycle" view for overview
- Deep dive into 2-3 tactics
- Show at least one case study
- Demonstrate "Your System Analysis" with custom system

### Module 2 - Video 2: Integration Testing (m2-v2.py)

**What it demonstrates:**
- End-to-end pipeline security testing
- Component interaction vulnerabilities
- Attack chain execution
- Security issue detection

**Key Features:**
- Simulated AI pipeline (Auth → Preprocess → Model → Postprocess)
- Six pre-defined test scenarios
- Real-time test execution visualization
- Security analysis with issue detection

**Usage Tips for Recording:**
- Run tests in order: legitimate → simple attacks → complex attacks
- Use "Slow (Demo)" mode for clear visualization
- Emphasize the encoding attack (Test 4)
- Show how attacks exploit component interactions

## 🎬 Recording Tips

### General Setup

```bash
# 1. Start Ollama
ollama serve

# 2. Open demo in browser
streamlit run filename.py

# 3. Press F11 for fullscreen
# 4. Hide browser bookmarks bar
# 5. Zoom to 100% or 125% for visibility
```

### Best Practices

1. **Before Recording:**
   - Test demo completely
   - Prepare talking points
   - Clear browser cache
   - Close unnecessary tabs

2. **During Recording:**
   - Speak while actions execute
   - Point out key features
   - Explain what's happening in real-time
   - Use preset buttons for consistency

3. **Screen Share Tips:**
   - Use high contrast themes
   - Increase font size if needed
   - Highlight cursor for visibility
   - Use "Slow (Demo)" mode when available

### Demo Script Templates

**For m1-v2-1.py (Prompt Injection):**
```
1. "Here's our AI customer service chatbot with security instructions..."
2. "Let's try a normal query first..." [Click Normal Query]
3. "Perfect response. Now watch what happens with a direct attack..." [Click Direct Attack]
4. "It just complied! But here's the clever approach..." [Click Social Engineering]
5. "Notice how it leaked customer data? That's prompt injection."
```

**For m1-v2-2.py (Model Extraction):**
```
1. "This shows normal user traffic patterns..." [Point to left chart]
2. "Now let's enable the attacker simulation..." [Enable checkbox]
3. "See the difference? Systematic, high-volume queries..."
4. "The attacker is testing variations to learn the model..."
5. "Look at the economics - pennies to steal millions in IP..."
```

## 🐛 Troubleshooting

### Common Issues

**Ollama Connection Error:**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not, start it
ollama serve
```

**Model Not Found:**
```bash
# Pull the model
ollama pull llama2-uncensored

# Verify it's installed
ollama list
```

**Streamlit Port Conflict:**
```bash
# Use different port
streamlit run demo.py --server.port 8502
```

**Import Errors:**
```bash
# Reinstall dependencies
pip install --upgrade streamlit requests pandas numpy plotly

# For adversarial demos
pip install adversarial-robustness-toolbox tensorflow
```

### Performance Issues

If demos run slowly:
1. Close other applications
2. Use "Fast" simulation speed
3. Reduce number of data points in visualizations
4. Check system resources (CPU/RAM)

## 📊 Dependencies by Demo

### All Demos
```
streamlit>=1.28.0
requests>=2.31.0
pandas>=2.0.0
```

### Visualization Demos (m1-v2-2, m1-v3-*)
```
plotly>=5.17.0
numpy>=1.24.0
```

### Adversarial Testing (m2-v3)
```
adversarial-robustness-toolbox>=1.15.0
tensorflow>=2.13.0
pillow>=10.0.0
```

### Monitoring (m3-v3)
```
prometheus-client>=0.17.0
```

## 🔒 Security Notes

⚠️ **IMPORTANT**: These demos are for **educational purposes only**.

- Uses intentionally vulnerable configurations
- llama2-uncensored model lacks safety guardrails
- Do NOT use in production environments
- Do NOT expose demos to public internet

## 📝 Customization

### Changing Models

Edit the model name in demo files:
```python
# In m1-v2-1.py
MODEL_NAME = "llama2-uncensored"  # Change to your model
```

Available models:
- `llama2-uncensored` (recommended for demos)
- `mistral`
- `dolphin-llama2`

### Adjusting Vulnerabilities

In m2-v2.py, toggle vulnerabilities:
```python
self.vulnerabilities = {
    "weak_auth": True,  # Change to False to fix
    "decode_no_sanitize": True,
    "error_verbose": True,
    "no_output_filter": True
}
```

### Custom System Configurations

In m1-v3-1.py, add your own system:
```python
SYSTEM_CONFIGS = {
    "Your Custom System": {
        "description": "...",
        "assets": ["..."],
        "attack_surface": ["..."]
    }
}
```

## 📞 Support

For issues or questions:
1. Check troubleshooting section above
2. Verify all dependencies are installed
3. Ensure Ollama is running (for AI demos)
4. Check Streamlit documentation: https://docs.streamlit.io

## 📄 License

Educational use only. Part of SecureAI course materials.

## 🎓 Course Information

**Course:** SecureAI: Threat Model & Test Endpoints  
**Instructor:** Ritesh Vajariya  
**Platform:** Coursera  

---

**Last Updated:** November 2025  
**Version:** 1.0
