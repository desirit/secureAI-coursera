# SecureAI: Threat Model & Test Endpoints

Welcome to the course repository! This contains all the interactive demonstrations and monitoring tools you'll use throughout the course.

## 🎯 What's Inside

This repository includes hands-on demos for learning AI security concepts:

### Module 1: Understanding AI-Specific Threat Models
- **Prompt Injection Attacks** - See how attackers manipulate AI responses
- **Model Extraction** - Understand how models can be stolen
- **STRIDE Threat Modeling** - Apply systematic threat analysis to AI systems
- **MITRE ATLAS Framework** - Explore real-world AI attack patterns

### Module 2: Creating Security Test Cases
- **Integration Testing** - Test security across AI pipeline components
- **Adversarial Testing** - Generate adversarial examples and test model robustness

### Module 3: CI/CD Integration & Monitoring
- **CI/CD Security Gates** - Automate security checks in deployment pipelines
- **Continuous Monitoring** - Real-time monitoring with Prometheus and Grafana

## 🚀 Getting Started

### Prerequisites

You'll need:
1. **Python 3.8 or higher** - [Download Python](https://www.python.org/downloads/)
2. **Ollama** - For running local AI models - [Install Ollama](https://ollama.ai)

### Step 1: Install Ollama and Model

```bash
# Install Ollama from https://ollama.ai

# Pull the uncensored model (needed for security demos)
ollama pull llama2-uncensored

# Start Ollama service
ollama serve
```

Keep this terminal window open while running demos.

### Step 2: Set Up Python Environment

**Option A: Automated Setup (Recommended)**

```bash
# On Linux/Mac:
chmod +x setup.sh
./setup.sh

# On Windows:
setup.bat
```

**Option B: Manual Setup**

```bash
# 1. Create virtual environment
python3 -m venv venv

# 2. Activate it
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Step 3: Run Your First Demo

```bash
# Start any demo (example: Prompt Injection)
streamlit run m1-v2-1.py

# Your browser will open automatically at http://localhost:8501
```

## 📚 Demo Guide

### Module 1 Demos

#### m1-v2-1.py: Prompt Injection Attack
**What you'll learn:**
- How prompt injection bypasses AI safety measures
- Different attack techniques (direct, social engineering)
- Why traditional security tools miss these attacks

**How to use:**
1. Read the system prompt in the sidebar
2. Try the "Normal Query" button - see safe behavior
3. Click "Direct Attack" - see how AI instructions are bypassed
4. Try "Social Engineering" - see a more sophisticated attack
5. Review the security analysis showing data leakage

#### m1-v2-2.py: Model Extraction
**What you'll learn:**
- How attackers systematically query models to steal them
- Visual patterns that indicate model extraction attempts
- Economic impact of model theft

**How to use:**
1. Start with normal traffic simulation
2. Enable "Simulate Attacker" checkbox
3. Watch the different patterns emerge
4. Review the statistics and cost analysis

#### m1-v3-1.py: STRIDE Threat Modeling
**What you'll learn:**
- Apply STRIDE framework to AI systems
- Identify threats across six categories
- Assess and prioritize security risks

**How to use:**
1. Select a system type (or describe your own)
2. Work through each STRIDE category
3. Document threats you identify
4. Review the risk assessment and export report

#### m1-v3-2.py: MITRE ATLAS Framework
**What you'll learn:**
- Comprehensive AI attack lifecycle
- Real-world case studies (Tay, Tesla autopilot, GPT-3)
- Mapping attacks to tactics and techniques

**How to use:**
1. Explore the "Attack Lifecycle Overview"
2. Deep dive into specific tactics
3. Read case studies to see real applications
4. Analyze your own system for vulnerabilities

### Module 2 Demos

#### m2-v2.py: Integration Testing
**What you'll learn:**
- How vulnerabilities span multiple components
- Testing entire AI pipelines, not just models
- Detection of complex attack chains

**How to use:**
1. Review the pipeline architecture diagram
2. Run pre-defined test scenarios
3. Watch how attacks exploit component interactions
4. Review security analysis for each test

#### m2-v3.py: Adversarial Testing with ART
**What you'll learn:**
- Generate adversarial examples against real models
- Test model robustness with various attacks
- Understand model vulnerabilities

**How to use:**
1. Select an attack type (FGSM, PGD, etc.)
2. Choose target images from MNIST dataset
3. Generate adversarial examples
4. Compare original vs adversarial predictions
5. Review attack success rates

### Module 3 Demos

#### m3-v2.py: CI/CD Security Gates
**What you'll learn:**
- Automate security testing in deployment pipelines
- Configure quality gates and thresholds
- Interpret security scan results

**How to use:**
1. Configure security thresholds for your needs
2. Run simulated CI/CD pipeline
3. Review security scan results
4. See how gates block vulnerable deployments

#### m3-v3.py: Continuous Monitoring Dashboard
**What you'll learn:**
- Monitor AI systems in production
- Detect anomalies and security events in real-time
- Configure alerts for security incidents

**How to use:**
1. Start the monitoring stack (see below)
2. Run the Streamlit dashboard
3. Watch real-time metrics
4. Simulate attacks to see detection

**Monitoring Stack Setup:**
```bash
cd monitoring-stack
docker compose up -d

# Access tools:
# Grafana: http://localhost:3000 (admin / secureai123)
# Prometheus: http://localhost:9090
# Metrics: http://localhost:8004/metrics
```

## 🔧 Troubleshooting

### Ollama Issues

**Problem:** "Connection refused" or "Model not found"
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not running, start it
ollama serve

# Verify model is installed
ollama list

# If model missing, pull it
ollama pull llama2-uncensored
```

### Python/Streamlit Issues

**Problem:** Port already in use
```bash
# Use a different port
streamlit run m1-v2-1.py --server.port 8502
```

**Problem:** Missing dependencies
```bash
# Reinstall all requirements
pip install -r requirements.txt

# Or install individually:
pip install streamlit requests pandas numpy plotly
```

### Docker Issues

**Problem:** Port conflicts in monitoring stack
```bash
# Check what's using the ports
netstat -tuln | grep -E ':(3000|8004|9090)'

# Stop conflicting services or edit docker-compose.yml ports
```

## 📖 Learning Path

We recommend this sequence:

1. **Start with Module 1** - Understand the threats
   - Run m1-v2-1.py to see prompt injection
   - Run m1-v2-2.py to understand model extraction
   - Complete threat modeling with m1-v3-*.py

2. **Move to Module 2** - Learn testing techniques
   - Practice integration testing with m2-v2.py
   - Generate adversarial examples with m2-v3.py

3. **Finish with Module 3** - Implement security at scale
   - Set up CI/CD gates with m3-v2.py
   - Deploy monitoring with m3-v3.py

## 🔒 Important Security Notes

⚠️ **For Educational Use Only**

- These demos intentionally contain vulnerabilities for learning
- The `llama2-uncensored` model lacks safety guardrails by design
- **Never** deploy these demos to production
- **Never** expose these demos to the public internet
- Only run in safe, isolated learning environments

## 💡 Tips for Success

1. **Run demos alongside course videos** - Pause videos and try the demos yourself
2. **Experiment with inputs** - Don't just use preset buttons, try your own queries
3. **Read the code** - All demos are well-commented, learn from the implementations
4. **Take notes** - Document interesting findings in your own threat models
5. **Join discussions** - Share your discoveries in course forums

## 📦 What's Included

```
secureAI-coursera/
├── m1-v2-1.py              # Prompt injection demo
├── m1-v2-2.py              # Model extraction demo
├── m1-v3-1.py              # STRIDE threat modeling
├── m1-v3-2.py              # MITRE ATLAS framework
├── m2-v2.py                # Integration testing
├── m2-v3.py                # Adversarial testing
├── m3-v2.py                # CI/CD security gates
├── m3-v3.py                # Monitoring dashboard
├── monitoring-stack/       # Prometheus + Grafana setup
│   ├── docker-compose.yml
│   ├── prometheus.yml
│   ├── alerts.yml
│   └── grafana/           # Dashboard configs
├── requirements.txt        # Python dependencies
├── setup.sh               # Linux/Mac setup script
└── setup.bat              # Windows setup script
```

## 🎓 Course Information

**Course:** SecureAI: Threat Model & Test Endpoints
**Instructor:** Ritesh Vajariya
**Platform:** Coursera

## 🤝 Getting Help

If you run into issues:

1. Check the **Troubleshooting** section above
2. Review the course discussion forums
3. Verify all prerequisites are installed correctly
4. Make sure Ollama is running for AI demos
5. Check that all Python dependencies are installed

## 📚 Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io)
- [Ollama Documentation](https://github.com/ollama/ollama)
- [MITRE ATLAS](https://atlas.mitre.org/)
- [OWASP Top 10 for LLMs](https://owasp.org/www-project-top-10-for-large-language-model-applications/)

---

**Happy Learning! 🚀**

Start with `streamlit run m1-v2-1.py` and explore AI security!
