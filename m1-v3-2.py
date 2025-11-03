"""
Module 1 Video 3 Part 2: MITRE ATLAS Framework for AI Systems
Mapping real-world attack techniques across the AI attack lifecycle
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="MITRE ATLAS Framework",
    page_icon="🗺️",
    layout="wide"
)

st.title("🗺️ MITRE ATLAS Attack Framework")
st.caption("Module 1 - Video 3 - Part 2: Mapping Real-World AI Adversary Tactics")

# Sidebar
with st.sidebar:
    st.header("🎯 Analysis Mode")
    
    view_mode = st.radio(
        "Select View",
        ["Attack Lifecycle", "Tactic Deep Dive", "Case Studies", "Your System Analysis"]
    )
    
    st.divider()
    
    st.header("📊 Filters")
    show_examples = st.checkbox("Show Real Examples", value=True)
    show_mitigations = st.checkbox("Show Countermeasures", value=True)

# MITRE ATLAS Framework Data
ATLAS_TACTICS = {
    "Reconnaissance": {
        "description": "Adversary gathers information about the target ML system",
        "techniques": {
            "Discover ML Model Ontology": {
                "id": "AML.T0002",
                "description": "Identify model architecture, training data, and capabilities",
                "examples": [
                    "Analyzing API documentation for model details",
                    "Probing with test inputs to infer model type",
                    "Studying error messages for architecture hints"
                ],
                "real_case": "Researchers analyzed OpenAI's GPT-3 API to understand model capabilities before launching attacks",
                "detection": ["Monitor reconnaissance patterns", "Limit information in errors", "Track systematic API exploration"],
                "mitigation": ["Minimize public model information", "Sanitize error messages", "Implement rate limiting"]
            },
            "Discover ML Artifacts": {
                "id": "AML.T0003",
                "description": "Find training data, model files, or other ML artifacts",
                "examples": [
                    "Searching GitHub for accidentally committed models",
                    "Finding exposed S3 buckets with training data",
                    "Discovering model weights in public registries"
                ],
                "real_case": "Multiple companies found with ML models leaked on public GitHub repositories",
                "detection": ["Monitor for unauthorized access", "Track artifact downloads", "Scan public repositories"],
                "mitigation": ["Secret scanning", "Access control", "Data loss prevention"]
            }
        }
    },
    "Resource Development": {
        "description": "Adversary develops resources to support attacks",
        "techniques": {
            "Develop Adversarial Dataset": {
                "id": "AML.T0008",
                "description": "Create dataset designed to fool the target model",
                "examples": [
                    "Generating adversarial images that fool classifiers",
                    "Creating poisoned training samples",
                    "Developing trigger patterns for backdoors"
                ],
                "real_case": "Researchers created adversarial stop signs that fooled Tesla Autopilot",
                "detection": ["Analyze training data for anomalies", "Validate data sources", "Statistical testing"],
                "mitigation": ["Adversarial training", "Data validation", "Multi-source verification"]
            },
            "Obtain Capability": {
                "id": "AML.T0007",
                "description": "Acquire tools or knowledge to attack ML systems",
                "examples": [
                    "Using open-source adversarial attack libraries",
                    "Purchasing similar models for testing",
                    "Learning from published attack papers"
                ],
                "real_case": "Adversarial Robustness Toolbox (ART) provides ready-to-use attack implementations",
                "detection": ["Monitor for known attack tool signatures", "Track related research activity"],
                "mitigation": ["Stay updated on attack research", "Test against known tools", "Red team exercises"]
            }
        }
    },
    "Initial Access": {
        "description": "Adversary gains initial access to the ML system",
        "techniques": {
            "ML Supply Chain Compromise": {
                "id": "AML.T0011",
                "description": "Compromise ML pipeline components or dependencies",
                "examples": [
                    "Poisoning pip packages used in ML",
                    "Compromising pre-trained models on public repositories",
                    "Backdooring ML frameworks"
                ],
                "real_case": "Malicious Python packages targeting ML developers discovered on PyPI",
                "detection": ["Dependency scanning", "Integrity verification", "Supply chain monitoring"],
                "mitigation": ["Vendor verification", "Dependency pinning", "Private repositories"]
            },
            "Valid Accounts": {
                "id": "AML.T0012",
                "description": "Use legitimate credentials to access ML systems",
                "examples": [
                    "Stolen API keys for model access",
                    "Compromised data scientist accounts",
                    "Phished ML platform credentials"
                ],
                "real_case": "Compromised AWS credentials led to unauthorized model access in several incidents",
                "detection": ["Unusual access patterns", "Geographic anomalies", "Behavioral analysis"],
                "mitigation": ["MFA enforcement", "Credential rotation", "Least privilege access"]
            }
        }
    },
    "ML Attack Staging": {
        "description": "Adversary prepares for the actual attack on the ML model",
        "techniques": {
            "Craft Adversarial Data": {
                "id": "AML.T0015",
                "description": "Create inputs designed to fool the model",
                "examples": [
                    "Adding imperceptible perturbations to images",
                    "Creating adversarial text prompts",
                    "Modifying audio to evade detection"
                ],
                "real_case": "Adversarial patches on stop signs caused misclassification in vision systems",
                "detection": ["Input anomaly detection", "Adversarial input classifiers", "Statistical analysis"],
                "mitigation": ["Adversarial training", "Input preprocessing", "Ensemble methods"]
            },
            "Poison Training Data": {
                "id": "AML.T0018",
                "description": "Inject malicious data into training pipeline",
                "examples": [
                    "Submitting mislabeled data via crowdsourcing",
                    "Injecting backdoor triggers",
                    "Corrupting public datasets"
                ],
                "real_case": "Researchers showed 3% poisoned data could compromise medical AI",
                "detection": ["Data validation", "Outlier detection", "Provenance tracking"],
                "mitigation": ["Data sanitization", "Multi-source validation", "Robust training methods"]
            }
        }
    },
    "Persistence": {
        "description": "Adversary maintains presence in the ML system",
        "techniques": {
            "Backdoor ML Model": {
                "id": "AML.T0024",
                "description": "Embed hidden functionality in the model",
                "examples": [
                    "Trojan triggers in image classifiers",
                    "Hidden commands in NLP models",
                    "Conditional behaviors in models"
                ],
                "real_case": "BadNets research demonstrated backdoors in neural networks",
                "detection": ["Model behavior monitoring", "Trigger analysis", "Activation analysis"],
                "mitigation": ["Model validation", "Regular retraining", "Input sanitization"]
            }
        }
    },
    "Defense Evasion": {
        "description": "Adversary avoids detection of attacks",
        "techniques": {
            "Evade ML Model": {
                "id": "AML.T0026",
                "description": "Craft inputs that bypass model defenses",
                "examples": [
                    "Adversarial examples that look normal",
                    "Prompt injection that appears benign",
                    "Evasion through input transformation"
                ],
                "real_case": "Adversarial examples bypassing spam filters and malware detectors",
                "detection": ["Multiple detection layers", "Ensemble defenses", "Behavioral analysis"],
                "mitigation": ["Defense in depth", "Diverse model architectures", "Human oversight"]
            }
        }
    },
    "Exfiltration": {
        "description": "Adversary steals information from the ML system",
        "techniques": {
            "Infer Training Data Membership": {
                "id": "AML.T0035",
                "description": "Determine if specific data was in training set",
                "examples": [
                    "Membership inference attacks",
                    "Querying for memorized data",
                    "Statistical inference"
                ],
                "real_case": "Researchers extracted training data from GPT-2 model",
                "detection": ["Query pattern monitoring", "Output filtering", "Anomaly detection"],
                "mitigation": ["Differential privacy", "Output sanitization", "Query limits"]
            },
            "Exfiltrate ML Model": {
                "id": "AML.T0036",
                "description": "Steal the model through systematic querying",
                "examples": [
                    "Model extraction via API abuse",
                    "Copying model architecture",
                    "Stealing model weights"
                ],
                "real_case": "Commercial image classifiers extracted through systematic API queries",
                "detection": ["Query volume monitoring", "Pattern detection", "Rate anomalies"],
                "mitigation": ["Intelligent rate limiting", "Output perturbation", "Watermarking"]
            }
        }
    },
    "Impact": {
        "description": "Adversary achieves their objective",
        "techniques": {
            "Erode ML Model Integrity": {
                "id": "AML.T0040",
                "description": "Degrade model performance or introduce bias",
                "examples": [
                    "Causing misclassifications in production",
                    "Introducing discriminatory bias",
                    "Breaking safety guardrails"
                ],
                "real_case": "Tay chatbot manipulated to output offensive content",
                "detection": ["Performance monitoring", "Bias testing", "Output validation"],
                "mitigation": ["Robust training", "Continuous validation", "Human oversight"]
            }
        }
    }
}

# Main content based on view mode
if view_mode == "Attack Lifecycle":
    st.header("MITRE ATLAS Attack Lifecycle")
    
    st.info("""
    MITRE ATLAS extends the MITRE ATT&CK framework for machine learning systems. 
    It maps adversary tactics and techniques across the ML attack lifecycle.
    """)
    
    # Create attack chain visualization
    st.subheader("📊 Attack Chain Flow")
    
    tactics_list = list(ATLAS_TACTICS.keys())
    
    fig = go.Figure(data=[go.Sankey(
        node = dict(
            pad = 15,
            thickness = 20,
            line = dict(color = "white", width = 2),
            label = tactics_list,
            color = ["#ff6b6b", "#ffa07a", "#ffd700", "#ffeb3b", "#66bb6a", "#42a5f5", "#ba68c8", "#b0bec5"],
            # Add font configuration for better readability
        ),
        link = dict(
            source = [0, 1, 2, 3, 3, 4, 5, 6],
            target = [1, 2, 3, 4, 5, 6, 7, 7],
            value = [1, 1, 1, 1, 1, 1, 1, 1],
            color = ["rgba(255, 107, 107, 0.3)", "rgba(255, 160, 122, 0.3)", "rgba(255, 215, 0, 0.3)",
                     "rgba(255, 235, 59, 0.3)", "rgba(255, 235, 59, 0.3)", "rgba(102, 187, 106, 0.3)",
                     "rgba(66, 165, 245, 0.3)", "rgba(186, 104, 200, 0.3)"]
        ),
        textfont = dict(
            size = 14,
            color = "white",
            family = "Arial, sans-serif"
        )
    )])

    fig.update_layout(
        title_text="ATLAS Tactic Flow",
        font_size=14,
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Tactic overview
    st.subheader("🎯 Tactic Overview")
    
    for tactic, details in ATLAS_TACTICS.items():
        with st.expander(f"**{tactic}** - {len(details['techniques'])} Techniques"):
            st.write(details['description'])
            st.write(f"\n**Techniques:**")
            for tech_name in details['techniques'].keys():
                tech_id = details['techniques'][tech_name]['id']
                st.write(f"• [{tech_id}] {tech_name}")

elif view_mode == "Tactic Deep Dive":
    st.header("Tactic Deep Dive")
    
    # Select tactic
    selected_tactic = st.selectbox("Select Tactic to Explore:", list(ATLAS_TACTICS.keys()))
    
    tactic_info = ATLAS_TACTICS[selected_tactic]
    
    st.subheader(f"{selected_tactic}")
    st.write(f"**Description:** {tactic_info['description']}")
    
    st.divider()
    
    # Display techniques
    for tech_name, tech_details in tactic_info['techniques'].items():
        st.subheader(f"🔹 {tech_name}")
        st.code(tech_details['id'], language=None)
        
        st.write(f"**Description:** {tech_details['description']}")
        
        if show_examples:
            st.write("\n**Attack Examples:**")
            for i, example in enumerate(tech_details['examples'], 1):
                st.write(f"{i}. {example}")
        
        if show_examples and 'real_case' in tech_details:
            st.warning(f"**🔍 Real-World Case:** {tech_details['real_case']}")
        
        if show_mitigations:
            col1, col2 = st.columns(2)
            
            with col1:
                st.success("**Detection Methods:**")
                for detection in tech_details['detection']:
                    st.write(f"• {detection}")
            
            with col2:
                st.info("**Mitigation Strategies:**")
                for mitigation in tech_details['mitigation']:
                    st.write(f"• {mitigation}")
        
        st.divider()

elif view_mode == "Case Studies":
    st.header("Real-World Attack Case Studies")
    
    # Case study 1
    with st.expander("🔴 Case Study 1: Microsoft Tay Chatbot Manipulation", expanded=True):
        st.subheader("Attack Timeline")
        
        timeline_data = {
            "Time": ["March 23, 2016 - 9:00 AM", "11:00 AM", "2:00 PM", "4:00 PM", "11:00 PM"],
            "Event": [
                "Tay launched on Twitter",
                "Users discover Tay learns from interactions",
                "Coordinated attack begins - users feed offensive content",
                "Tay starts outputting racist/offensive tweets",
                "Microsoft takes Tay offline"
            ],
            "ATLAS Tactic": [
                "Initial Access",
                "Reconnaissance",
                "ML Attack Staging (Poison Training)",
                "Impact (Erode Model Integrity)",
                "Incident Response"
            ]
        }
        
        df_timeline = pd.DataFrame(timeline_data)
        st.table(df_timeline)
        
        st.write("\n**Attack Analysis:**")
        st.write("- **Technique Used:** Data Poisoning through user interaction")
        st.write("- **Vulnerability:** Real-time learning without content filtering")
        st.write("- **Impact:** Reputational damage, service shutdown")
        st.write("- **Lesson:** Need for robust content filtering and adversarial training")
    
    # Case study 2
    with st.expander("🟠 Case Study 2: Adversarial Stop Sign Attack (Tesla Autopilot)"):
        st.subheader("Attack Details")
        
        st.write("""
        **Attack Vector:** Physical adversarial perturbations
        
        **Technique:** Researchers placed small stickers on stop signs that:
        - Appeared innocuous to human drivers
        - Caused AI vision system to misclassify as speed limit signs
        - Demonstrated real-world applicability of adversarial examples
        
        **ATLAS Mapping:**
        """)
        
        attack_mapping = pd.DataFrame({
            "Phase": ["Resource Development", "ML Attack Staging", "Defense Evasion", "Impact"],
            "Technique": [
                "Develop Adversarial Dataset",
                "Craft Adversarial Data",
                "Evade ML Model",
                "Erode ML Model Integrity"
            ],
            "Action": [
                "Create adversarial sticker designs",
                "Test designs on similar vision models",
                "Ensure stickers look innocuous",
                "Successfully fool production system"
            ]
        })
        
        st.table(attack_mapping)
        
        st.error("**Safety Impact:** Potential for accidents if exploited maliciously")
    
    # Case study 3
    with st.expander("🟡 Case Study 3: GPT-3 Model Extraction Attempts"):
        st.write("""
        **Attack Scenario:** Researchers attempted to extract OpenAI's GPT-3 model
        
        **Method:**
        1. **Reconnaissance:** Analyzed API documentation and pricing
        2. **Resource Development:** Designed systematic query strategy
        3. **Exfiltration Attempt:** Sent thousands of carefully crafted queries
        4. **Detection:** OpenAI's monitoring systems detected anomalous patterns
        
        **Outcome:**
        - Attack detected before significant model theft
        - Led to improved rate limiting and query monitoring
        - Demonstrated importance of behavioral analysis
        
        **Defensive Measures Implemented:**
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("**Detection:**")
            st.write("• Query pattern analysis")
            st.write("• Volume anomaly detection")
            st.write("• User behavior profiling")
        
        with col2:
            st.info("**Prevention:**")
            st.write("• Dynamic rate limiting")
            st.write("• Output perturbation")
            st.write("• API usage auditing")

elif view_mode == "Your System Analysis":
    st.header("Map ATLAS to Your System")
    
    st.write("""
    Use this tool to map ATLAS tactics and techniques to your specific AI system.
    This helps identify which attacks are most relevant to your threat model.
    """)
    
    # System input
    system_name = st.text_input("Your AI System Name:")
    system_desc = st.text_area("System Description:", height=100)
    
    if system_name and system_desc:
        st.subheader(f"Threat Mapping for: {system_name}")
        
        # Allow users to select relevant tactics
        st.write("**Select applicable tactics for your system:**")
        
        selected_tactics = []
        for tactic in ATLAS_TACTICS.keys():
            if st.checkbox(f"{tactic} - {ATLAS_TACTICS[tactic]['description']}", key=f"tactic_{tactic}"):
                selected_tactics.append(tactic)
        
        if selected_tactics:
            st.divider()
            st.subheader("📋 Your System's Threat Landscape")
            
            threat_count = 0
            for tactic in selected_tactics:
                st.write(f"\n**{tactic}:**")
                for tech_name, tech_details in ATLAS_TACTICS[tactic]['techniques'].items():
                    threat_count += 1
                    st.write(f"• [{tech_details['id']}] {tech_name}")
                    st.caption(tech_details['description'])
            
            st.success(f"**Total Relevant Threats Identified:** {threat_count}")
            
            # Export
            if st.button("📥 Export Threat Analysis"):
                report = {
                    "system": system_name,
                    "description": system_desc,
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "tactics": selected_tactics,
                    "total_threats": threat_count
                }
                
                st.download_button(
                    "Download Analysis",
                    data=str(report),
                    file_name=f"atlas_analysis_{system_name.replace(' ', '_')}.txt"
                )

# Footer
st.divider()
col1, col2 = st.columns([3, 1])

with col1:
    st.caption("🗺️ **MITRE ATLAS** - Adversarial Threat Landscape for Artificial-Intelligence Systems")
    st.caption("Learn more: https://atlas.mitre.org")

with col2:
    if st.button("🔗 Visit MITRE ATLAS"):
        st.write("https://atlas.mitre.org")
