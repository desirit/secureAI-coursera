"""
Module 1 Video 3 Part 1: STRIDE Threat Modeling for AI Systems
Interactive threat modeling framework demonstration
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import json

# Page config
st.set_page_config(
    page_title="STRIDE Threat Modeling",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ STRIDE Threat Modeling for AI Systems")
st.caption("Module 1 - Video 3 - Part 1: Building Comprehensive AI Threat Models")

# Sidebar - System Selection
with st.sidebar:
    st.header("🎯 Target System")
    
    system_type = st.selectbox(
        "Select AI System to Model",
        ["Medical Diagnostic AI (X-Ray Analysis)",
         "Financial Loan Approval System",
         "Autonomous Vehicle Vision System",
         "Custom AI System"]
    )
    
    if system_type == "Custom AI System":
        custom_name = st.text_input("System Name")
        custom_desc = st.text_area("System Description")
    
    st.divider()
    
    st.header("⚙️ Assessment Settings")
    show_examples = st.checkbox("Show Example Threats", value=True)
    show_mitigations = st.checkbox("Show Mitigation Strategies", value=True)
    risk_calculation = st.checkbox("Enable Risk Scoring", value=True)

# System descriptions
SYSTEM_CONFIGS = {
    "Medical Diagnostic AI (X-Ray Analysis)": {
        "description": "AI system that analyzes chest X-rays to detect pneumonia, tuberculosis, and other respiratory conditions",
        "assets": ["Patient health data", "Diagnostic model", "Medical images", "Doctor recommendations"],
        "attack_surface": ["API endpoints", "Training data pipeline", "Model inference engine", "Output interface"]
    },
    "Financial Loan Approval System": {
        "description": "AI system that evaluates loan applications and makes approval/denial decisions",
        "assets": ["Applicant financial data", "Credit histories", "Approval model", "Decision logs"],
        "attack_surface": ["Application submission API", "Data preprocessing", "Model scoring", "Decision reporting"]
    },
    "Autonomous Vehicle Vision System": {
        "description": "AI system that processes camera feeds to detect objects, signs, and navigate safely",
        "assets": ["Real-time camera feeds", "Object detection model", "Navigation decisions", "Safety protocols"],
        "attack_surface": ["Camera inputs", "Sensor fusion", "Model inference", "Control systems"]
    }
}

# STRIDE Categories and AI-specific threats
STRIDE_CATEGORIES = {
    "Spoofing": {
        "definition": "Pretending to be something or someone other than yourself",
        "ai_threats": [
            "Feeding fake training data through crowdsourcing platforms",
            "Spoofing medical image origins to poison training data",
            "Impersonating legitimate data sources",
            "Submitting adversarial examples disguised as normal inputs"
        ],
        "impact": "Model learns incorrect patterns, makes wrong decisions",
        "mitigations": [
            "Data provenance tracking",
            "Multi-source validation",
            "Cryptographic signatures on data",
            "Input authenticity verification"
        ]
    },
    "Tampering": {
        "definition": "Modifying data or code without authorization",
        "ai_threats": [
            "Modifying model weights in production",
            "Corrupting training datasets",
            "Tampering with inference parameters",
            "Manipulating feedback loops"
        ],
        "impact": "Compromised model behavior, incorrect predictions",
        "mitigations": [
            "Model integrity checks (hashing)",
            "Immutable training data storage",
            "Access control on model parameters",
            "Version control and change tracking"
        ]
    },
    "Repudiation": {
        "definition": "Claiming you didn't do something or that something didn't happen",
        "ai_threats": [
            "No audit trail of model decisions",
            "Cannot prove which training data influenced output",
            "Lack of versioning for model changes",
            "Missing logs of data processing steps"
        ],
        "impact": "Cannot investigate errors, prove compliance, or defend decisions",
        "mitigations": [
            "Comprehensive audit logging",
            "Model versioning and lineage tracking",
            "Input/output logging",
            "Explainability mechanisms"
        ]
    },
    "Information Disclosure": {
        "definition": "Exposing information to unauthorized individuals",
        "ai_threats": [
            "Model memorizes and leaks training data",
            "Model inversion attacks extract sensitive information",
            "Membership inference reveals if data was in training set",
            "Model extraction through systematic querying"
        ],
        "impact": "Privacy violations, intellectual property theft, competitive loss",
        "mitigations": [
            "Differential privacy in training",
            "Output filtering for sensitive data",
            "Rate limiting on API queries",
            "Model watermarking"
        ]
    },
    "Denial of Service": {
        "definition": "Denying or degrading service to legitimate users",
        "ai_threats": [
            "Crafted inputs causing excessive computation",
            "Triggering worst-case model complexity",
            "Resource exhaustion through query flooding",
            "Adversarial examples requiring re-processing"
        ],
        "impact": "Service unavailability, degraded performance, increased costs",
        "mitigations": [
            "Input validation and complexity limits",
            "Query rate limiting",
            "Resource monitoring and throttling",
            "Graceful degradation strategies"
        ]
    },
    "Elevation of Privilege": {
        "definition": "Gaining capabilities beyond authorized level",
        "ai_threats": [
            "Prompt injection to bypass restrictions",
            "Model manipulation to access unauthorized data",
            "Exploiting model to perform unintended actions",
            "Circumventing safety guardrails"
        ],
        "impact": "Unauthorized data access, system manipulation, safety violations",
        "mitigations": [
            "Strict input/output validation",
            "Principle of least privilege for model access",
            "Separate system instructions from user inputs",
            "Multi-layer security controls"
        ]
    }
}

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📋 System Overview", "🔍 STRIDE Analysis", "📊 Risk Assessment", "📄 Export Report"])

with tab1:
    st.header("System Configuration")
    
    if system_type != "Custom AI System":
        config = SYSTEM_CONFIGS[system_type]
        
        st.subheader(system_type)
        st.write(config["description"])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Critical Assets:**")
            for asset in config["assets"]:
                st.write(f"• {asset}")
        
        with col2:
            st.write("**Attack Surface:**")
            for surface in config["attack_surface"]:
                st.write(f"• {surface}")
    else:
        st.info("Configure your custom system in the sidebar")
    
    st.divider()
    st.subheader("🎯 Threat Modeling Approach")
    st.write("""
    We'll apply the STRIDE framework to systematically identify threats across six categories:
    
    1. **S**poofing - Identity and authenticity threats
    2. **T**ampering - Data and model integrity threats  
    3. **R**epudiation - Audit and accountability threats
    4. **I**nformation Disclosure - Privacy and confidentiality threats
    5. **D**enial of Service - Availability threats
    6. **E**levation of Privilege - Authorization and access control threats
    
    For each category, we'll identify AI-specific threats, assess their impact, and define mitigations.
    """)

with tab2:
    st.header("STRIDE Threat Analysis")
    
    # Allow users to work through each STRIDE category
    for category, details in STRIDE_CATEGORIES.items():
        with st.expander(f"**{category}** - {details['definition']}", expanded=(category == "Spoofing")):
            
            st.write(f"**Definition:** {details['definition']}")
            
            st.divider()
            
            if show_examples:
                st.subheader("🚨 AI-Specific Threats")
                for i, threat in enumerate(details['ai_threats'], 1):
                    st.write(f"{i}. {threat}")
                
                st.error(f"**Impact:** {details['impact']}")
            
            st.divider()
            
            # Interactive threat documentation
            st.subheader("📝 Document Your Threats")
            
            threat_key = f"{category.lower()}_threat"
            if threat_key not in st.session_state:
                st.session_state[threat_key] = ""
            
            user_threat = st.text_area(
                f"Describe specific {category} threats for your system:",
                value=st.session_state[threat_key],
                key=f"input_{threat_key}",
                height=100,
                placeholder=f"Example: {details['ai_threats'][0]}"
            )
            st.session_state[threat_key] = user_threat
            
            if show_mitigations and user_threat:
                st.success("**Recommended Mitigations:**")
                for i, mitigation in enumerate(details['mitigations'], 1):
                    st.write(f"{i}. {mitigation}")
            
            if risk_calculation and user_threat:
                st.divider()
                st.subheader("⚖️ Risk Assessment")
                
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    likelihood = st.select_slider(
                        "Likelihood",
                        options=["Very Low", "Low", "Medium", "High", "Very High"],
                        key=f"likelihood_{category}"
                    )
                
                with col_b:
                    impact = st.select_slider(
                        "Impact",
                        options=["Very Low", "Low", "Medium", "High", "Very High"],
                        key=f"impact_{category}"
                    )
                
                # Calculate risk score
                likelihood_score = {"Very Low": 1, "Low": 2, "Medium": 3, "High": 4, "Very High": 5}[likelihood]
                impact_score = {"Very Low": 1, "Low": 2, "Medium": 3, "High": 4, "Very High": 5}[impact]
                risk_score = likelihood_score * impact_score
                
                with col_c:
                    if risk_score >= 16:
                        risk_level = "🔴 Critical"
                        risk_color = "red"
                    elif risk_score >= 9:
                        risk_level = "🟠 High"
                        risk_color = "orange"
                    elif risk_score >= 4:
                        risk_level = "🟡 Medium"
                        risk_color = "yellow"
                    else:
                        risk_level = "🟢 Low"
                        risk_color = "green"
                    
                    st.metric("Risk Level", risk_level)
                    st.caption(f"Score: {risk_score}/25")
                
                # Store risk assessment
                risk_key = f"{category}_risk"
                st.session_state[risk_key] = {
                    "likelihood": likelihood,
                    "impact": impact,
                    "score": risk_score,
                    "level": risk_level
                }

with tab3:
    st.header("Risk Assessment Summary")
    
    # Collect all documented threats and risks
    threat_data = []
    for category in STRIDE_CATEGORIES.keys():
        threat_key = f"{category.lower()}_threat"
        risk_key = f"{category}_risk"
        
        if threat_key in st.session_state and st.session_state[threat_key]:
            threat_text = st.session_state[threat_key]
            
            if risk_key in st.session_state:
                risk_info = st.session_state[risk_key]
                threat_data.append({
                    "Category": category,
                    "Threat": threat_text[:100] + "..." if len(threat_text) > 100 else threat_text,
                    "Likelihood": risk_info["likelihood"],
                    "Impact": risk_info["impact"],
                    "Risk Score": risk_info["score"],
                    "Risk Level": risk_info["level"]
                })
    
    if threat_data:
        df = pd.DataFrame(threat_data)
        
        # Display summary table
        st.dataframe(df, use_container_width=True)
        
        # Risk distribution
        st.subheader("Risk Distribution")
        risk_counts = df["Risk Level"].value_counts()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            critical = risk_counts.get("🔴 Critical", 0)
            st.metric("Critical Risks", critical)
        
        with col2:
            high = risk_counts.get("🟠 High", 0)
            st.metric("High Risks", high)
        
        with col3:
            medium = risk_counts.get("🟡 Medium", 0)
            st.metric("Medium Risks", medium)
        
        with col4:
            low = risk_counts.get("🟢 Low", 0)
            st.metric("Low Risks", low)
        
        # Prioritization
        st.subheader("📌 Prioritized Action Items")
        
        # Sort by risk score
        df_sorted = df.sort_values("Risk Score", ascending=False)
        
        st.write("**Immediate Attention Required:**")
        critical_high = df_sorted[df_sorted["Risk Score"] >= 9]
        
        if len(critical_high) > 0:
            for idx, row in critical_high.iterrows():
                st.error(f"**{row['Category']}** - {row['Risk Level']} (Score: {row['Risk Score']})")
                st.write(row['Threat'])
                st.write("")
        else:
            st.success("No critical or high-risk threats identified!")
    else:
        st.info("Document threats in the STRIDE Analysis tab to see risk assessment summary")

with tab4:
    st.header("Export Threat Model Report")
    
    if threat_data:
        # Generate report
        report = {
            "system": system_type,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "threats": []
        }
        
        for category in STRIDE_CATEGORIES.keys():
            threat_key = f"{category.lower()}_threat"
            risk_key = f"{category}_risk"
            
            if threat_key in st.session_state and st.session_state[threat_key]:
                threat_entry = {
                    "category": category,
                    "threat_description": st.session_state[threat_key],
                    "recommended_mitigations": STRIDE_CATEGORIES[category]["mitigations"]
                }
                
                if risk_key in st.session_state:
                    threat_entry["risk_assessment"] = st.session_state[risk_key]
                
                report["threats"].append(threat_entry)
        
        # Display formatted report
        st.subheader("Threat Model Report")
        
        st.write(f"**System:** {report['system']}")
        st.write(f"**Date:** {report['date']}")
        st.write(f"**Total Threats Identified:** {len(report['threats'])}")
        
        st.divider()
        
        for threat in report["threats"]:
            st.write(f"### {threat['category']}")
            st.write(f"**Threat:** {threat['threat_description']}")
            
            if 'risk_assessment' in threat:
                risk = threat['risk_assessment']
                st.write(f"**Risk Level:** {risk['level']} (Score: {risk['score']}/25)")
                st.write(f"**Likelihood:** {risk['likelihood']} | **Impact:** {risk['impact']}")
            
            st.write("**Recommended Mitigations:**")
            for i, mit in enumerate(threat['recommended_mitigations'], 1):
                st.write(f"{i}. {mit}")
            
            st.write("")
        
        # Download button
        json_report = json.dumps(report, indent=2)
        st.download_button(
            label="📥 Download JSON Report",
            data=json_report,
            file_name=f"stride_threat_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    else:
        st.info("Complete the STRIDE Analysis to generate a report")

# Footer
st.divider()
st.caption("🛡️ **STRIDE Framework** - Developed by Microsoft for systematic threat modeling")
