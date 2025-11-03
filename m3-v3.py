"""
Module 3 Video 3: Continuous Monitoring and Incident Response
Demonstrates security monitoring for production AI systems
Simulates Prometheus/Grafana-style monitoring dashboards
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import time
import random

# Page config
st.set_page_config(
    page_title="AI Security Monitoring",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI Security Monitoring Dashboard")
st.caption("Module 3 - Video 3: Continuous Security Monitoring and Incident Response")

# Sidebar
with st.sidebar:
    st.header("⚙️ Monitoring Configuration")
    
    st.subheader("Time Range")
    time_range = st.selectbox(
        "Select Time Window",
        ["Last 5 minutes", "Last 15 minutes", "Last 1 hour", "Last 24 hours"]
    )
    
    st.divider()
    
    st.subheader("Alert Settings")
    enable_alerts = st.checkbox("Enable Alerts", value=True)
    alert_threshold_queries = st.slider("Query Rate Alert (per min)", 10, 500, 100)
    alert_threshold_errors = st.slider("Error Rate Alert (%)", 1, 20, 5)
    
    st.divider()
    
    st.subheader("Simulation")
    simulate_attack = st.checkbox("Simulate Attack", value=False)
    attack_type = st.selectbox(
        "Attack Type",
        ["Model Extraction", "Prompt Injection", "Brute Force Auth"]
    )
    
    auto_refresh = st.checkbox("Auto Refresh", value=False)
    if auto_refresh:
        refresh_interval = st.slider("Refresh Interval (sec)", 1, 10, 3)

# Generate monitoring data
def generate_monitoring_data(minutes=60, simulate_attack_pattern=False, attack_type_sim="Model Extraction"):
    """Generate realistic monitoring data"""
    timestamps = pd.date_range(end=datetime.now(), periods=minutes, freq='1min')
    
    data = {
        'timestamp': timestamps,
        'query_rate': [],
        'auth_failures': [],
        'model_accuracy': [],
        'response_time': [],
        'adversarial_detected': [],
        'prompt_injection_attempts': [],
        'data_leakage_events': []
    }
    
    for i in range(minutes):
        # Normal baseline with some randomness
        if simulate_attack_pattern and attack_type_sim == "Model Extraction":
            # Sudden spike in query rate
            if i > minutes * 0.6 and i < minutes * 0.9:
                query_rate = random.randint(200, 400)  # Attack pattern
            else:
                query_rate = random.randint(20, 50)  # Normal
        else:
            query_rate = random.randint(20, 50)
        
        if simulate_attack_pattern and attack_type_sim == "Brute Force Auth":
            # Spike in auth failures
            if i > minutes * 0.5 and i < minutes * 0.8:
                auth_failures = random.randint(15, 30)
            else:
                auth_failures = random.randint(0, 3)
        else:
            auth_failures = random.randint(0, 3)
        
        # Model accuracy (drops if under attack)
        if simulate_attack_pattern and i > minutes * 0.7:
            accuracy = random.uniform(0.75, 0.85)
        else:
            accuracy = random.uniform(0.92, 0.96)
        
        # Response time (increases under load)
        if query_rate > 100:
            response_time = random.uniform(0.3, 0.8)
        else:
            response_time = random.uniform(0.1, 0.3)
        
        # Security events
        if simulate_attack_pattern and attack_type_sim == "Prompt Injection":
            if i > minutes * 0.6:
                prompt_injection = random.randint(5, 15)
            else:
                prompt_injection = random.randint(0, 2)
        else:
            prompt_injection = random.randint(0, 2)
        
        adversarial = random.randint(0, 5) if simulate_attack_pattern else random.randint(0, 1)
        data_leakage = random.randint(0, 1)
        
        data['query_rate'].append(query_rate)
        data['auth_failures'].append(auth_failures)
        data['model_accuracy'].append(accuracy)
        data['response_time'].append(response_time)
        data['adversarial_detected'].append(adversarial)
        data['prompt_injection_attempts'].append(prompt_injection)
        data['data_leakage_events'].append(data_leakage)
    
    return pd.DataFrame(data)

# Alert detection
def check_alerts(df, enable_alerts, query_threshold, error_threshold):
    """Check for alert conditions"""
    alerts = []
    
    if not enable_alerts:
        return alerts
    
    # Check last 5 minutes
    recent_data = df.tail(5)
    
    # Query rate alert
    avg_query_rate = recent_data['query_rate'].mean()
    if avg_query_rate > query_threshold:
        alerts.append({
            'severity': 'high',
            'type': 'Query Rate',
            'message': f'High query rate detected: {avg_query_rate:.0f} queries/min (threshold: {query_threshold})',
            'time': datetime.now().strftime('%H:%M:%S')
        })
    
    # Auth failure alert
    auth_failures = recent_data['auth_failures'].sum()
    if auth_failures > 20:
        alerts.append({
            'severity': 'critical',
            'type': 'Authentication',
            'message': f'Multiple authentication failures: {auth_failures} in 5 minutes',
            'time': datetime.now().strftime('%H:%M:%S')
        })
    
    # Accuracy drop alert
    current_accuracy = recent_data['model_accuracy'].mean()
    if current_accuracy < 0.85:
        alerts.append({
            'severity': 'high',
            'type': 'Model Performance',
            'message': f'Model accuracy dropped to {current_accuracy:.1%}',
            'time': datetime.now().strftime('%H:%M:%S')
        })
    
    # Prompt injection alert
    prompt_attempts = recent_data['prompt_injection_attempts'].sum()
    if prompt_attempts > 10:
        alerts.append({
            'severity': 'high',
            'type': 'Prompt Injection',
            'message': f'{prompt_attempts} prompt injection attempts detected',
            'time': datetime.now().strftime('%H:%M:%S')
        })
    
    return alerts

# Main content
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Live Metrics",
    "🚨 Alerts & Incidents",
    "📊 Analytics",
    "📚 Playbooks"
])

with tab1:
    st.header("Real-Time Security Metrics")
    
    # Generate data
    time_map = {
        "Last 5 minutes": 5,
        "Last 15 minutes": 15,
        "Last 1 hour": 60,
        "Last 24 hours": 1440
    }
    
    minutes = time_map[time_range]
    df = generate_monitoring_data(
        minutes=minutes,
        simulate_attack_pattern=simulate_attack,
        attack_type_sim=attack_type if simulate_attack else None
    )
    
    # Check for alerts
    alerts = check_alerts(df, enable_alerts, alert_threshold_queries, alert_threshold_errors)
    
    # Show alerts banner
    if alerts:
        st.error(f"🚨 **{len(alerts)} Active Alerts**")
        for alert in alerts[:3]:  # Show top 3
            severity_color = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
            st.warning(f"{severity_color[alert['severity']]} **{alert['type']}**: {alert['message']} ({alert['time']})")
    
    # Key metrics row
    st.subheader("📊 Current Status")
    col1, col2, col3, col4 = st.columns(4)
    
    current_query_rate = df['query_rate'].iloc[-1]
    current_accuracy = df['model_accuracy'].iloc[-1]
    current_response_time = df['response_time'].iloc[-1]
    total_security_events = df['prompt_injection_attempts'].sum() + df['adversarial_detected'].sum()
    
    with col1:
        st.metric(
            "Query Rate",
            f"{current_query_rate}/min",
            delta=f"{current_query_rate - df['query_rate'].iloc[-2]}" if len(df) > 1 else None
        )
    
    with col2:
        st.metric(
            "Model Accuracy",
            f"{current_accuracy:.1%}",
            delta=f"{(current_accuracy - df['model_accuracy'].iloc[-2]):.1%}" if len(df) > 1 else None
        )
    
    with col3:
        st.metric(
            "Avg Response Time",
            f"{current_response_time:.2f}s",
            delta=f"{(current_response_time - df['response_time'].iloc[-2]):.2f}s" if len(df) > 1 else None,
            delta_color="inverse"
        )
    
    with col4:
        st.metric(
            "Security Events",
            total_security_events,
            delta=f"+{df['prompt_injection_attempts'].iloc[-1] + df['adversarial_detected'].iloc[-1]}"
        )
    
    st.divider()
    
    # Time series charts
    st.subheader("📈 Query Pattern Analysis")
    
    fig_queries = go.Figure()
    fig_queries.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['query_rate'],
        mode='lines',
        name='Query Rate',
        line=dict(color='#3498db', width=2),
        fill='tozeroy',
        fillcolor='rgba(52, 152, 219, 0.1)'
    ))
    
    # Add alert threshold line
    fig_queries.add_hline(
        y=alert_threshold_queries,
        line_dash="dash",
        line_color="red",
        annotation_text="Alert Threshold"
    )
    
    fig_queries.update_layout(
        title="Query Rate Over Time",
        xaxis_title="Time",
        yaxis_title="Queries per Minute",
        height=300,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_queries, use_container_width=True)
    
    # Model performance
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("🎯 Model Performance")
        
        fig_accuracy = go.Figure()
        fig_accuracy.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['model_accuracy'],
            mode='lines',
            name='Accuracy',
            line=dict(color='#2ecc71', width=2)
        ))
        
        fig_accuracy.add_hline(
            y=0.85,
            line_dash="dash",
            line_color="red",
            annotation_text="Min Threshold"
        )
        
        fig_accuracy.update_layout(
            title="Model Accuracy",
            xaxis_title="Time",
            yaxis_title="Accuracy",
            height=300,
            yaxis=dict(range=[0.7, 1.0])
        )
        
        st.plotly_chart(fig_accuracy, use_container_width=True)
    
    with col_b:
        st.subheader("⚡ Response Time")
        
        fig_response = go.Figure()
        fig_response.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['response_time'],
            mode='lines',
            name='Response Time',
            line=dict(color='#e67e22', width=2)
        ))
        
        fig_response.update_layout(
            title="API Response Time",
            xaxis_title="Time",
            yaxis_title="Seconds",
            height=300
        )
        
        st.plotly_chart(fig_response, use_container_width=True)
    
    # Security events
    st.subheader("🔒 Security Events")
    
    fig_security = go.Figure()
    
    fig_security.add_trace(go.Bar(
        x=df['timestamp'],
        y=df['prompt_injection_attempts'],
        name='Prompt Injection',
        marker_color='#e74c3c'
    ))
    
    fig_security.add_trace(go.Bar(
        x=df['timestamp'],
        y=df['adversarial_detected'],
        name='Adversarial Inputs',
        marker_color='#f39c12'
    ))
    
    fig_security.add_trace(go.Bar(
        x=df['timestamp'],
        y=df['auth_failures'],
        name='Auth Failures',
        marker_color='#95a5a6'
    ))
    
    fig_security.update_layout(
        title="Security Events by Type",
        xaxis_title="Time",
        yaxis_title="Count",
        barmode='stack',
        height=300
    )
    
    st.plotly_chart(fig_security, use_container_width=True)

with tab2:
    st.header("Alert Management & Incident Response")
    
    if alerts:
        st.subheader(f"🚨 Active Alerts ({len(alerts)})")
        
        for alert in alerts:
            severity_colors = {
                "critical": ("🔴", "error"),
                "high": ("🟠", "warning"),
                "medium": ("🟡", "info"),
                "low": ("🟢", "success")
            }
            
            icon, method = severity_colors[alert['severity']]
            
            with st.expander(f"{icon} [{alert['severity'].upper()}] {alert['type']} - {alert['time']}", expanded=True):
                getattr(st, method)(alert['message'])
                
                st.write("**Recommended Actions:**")
                
                if alert['type'] == 'Query Rate':
                    st.write("""
                    1. **Immediate:** Check if this is legitimate traffic or attack
                    2. **Investigate:** Review query patterns for systematic variations
                    3. **Action:** Enable rate limiting if attack detected
                    4. **Monitor:** Watch for distributed attack patterns
                    """)
                    
                    if st.button(f"Block High-Volume Users", key=f"block_{alert['time']}"):
                        st.success("✅ Rate limiting activated for suspicious accounts")
                
                elif alert['type'] == 'Authentication':
                    st.write("""
                    1. **Immediate:** Block source IPs with failed attempts
                    2. **Investigate:** Check for brute force patterns
                    3. **Action:** Temporarily lock affected accounts
                    4. **Notify:** Alert security team and affected users
                    """)
                    
                    if st.button(f"Initiate Security Lockdown", key=f"lockdown_{alert['time']}"):
                        st.success("✅ Security lockdown initiated - suspicious IPs blocked")
                
                elif alert['type'] == 'Model Performance':
                    st.write("""
                    1. **Immediate:** Check if under adversarial attack
                    2. **Investigate:** Review recent inputs for anomalies
                    3. **Action:** Consider rolling back to previous model version
                    4. **Escalate:** Notify ML team for investigation
                    """)
                    
                    col_x, col_y = st.columns(2)
                    with col_x:
                        if st.button(f"Rollback Model", key=f"rollback_{alert['time']}"):
                            st.success("✅ Model rolled back to previous stable version")
                    with col_y:
                        if st.button(f"Enable Defensive Mode", key=f"defense_{alert['time']}"):
                            st.success("✅ Defensive filters activated")
                
                elif alert['type'] == 'Prompt Injection':
                    st.write("""
                    1. **Immediate:** Enable enhanced input filtering
                    2. **Investigate:** Analyze injection patterns
                    3. **Action:** Update prompt injection defenses
                    4. **Log:** Record all attempts for pattern analysis
                    """)
                    
                    if st.button(f"Strengthen Filters", key=f"filter_{alert['time']}"):
                        st.success("✅ Enhanced input filtering activated")
    else:
        st.success("✅ No active alerts - All systems normal")
    
    st.divider()
    
    st.subheader("📋 Incident Response Workflow")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Level 1: Automated**")
        st.write("- Rate limiting kicks in")
        st.write("- Block malicious IPs")
        st.write("- Log security events")
        st.write("- Send alerts")
    
    with col2:
        st.write("**Level 2: Investigation**")
        st.write("- Security team notified")
        st.write("- Gather forensic data")
        st.write("- Assess attack severity")
        st.write("- Determine response")
    
    with col3:
        st.write("**Level 3: Escalation**")
        st.write("- Incident commander assigned")
        st.write("- War room activated")
        st.write("- Execute response plan")
        st.write("- Stakeholder communication")

with tab3:
    st.header("Security Analytics")
    
    # User behavior analysis
    st.subheader("👥 User Behavior Analysis")
    
    # Simulated user data
    users = ['user_001', 'user_002', 'user_003', 'DataResearch_Co', 'user_005']
    user_data = []
    
    for user in users:
        if user == 'DataResearch_Co' and simulate_attack:
            queries = random.randint(5000, 10000)
            systematic = True
        else:
            queries = random.randint(10, 100)
            systematic = False
        
        user_data.append({
            'User': user,
            'Total Queries': queries,
            'Avg per Minute': queries / minutes,
            'Systematic Pattern': 'Yes ⚠️' if systematic else 'No',
            'Risk Level': 'High 🔴' if systematic else 'Low 🟢'
        })
    
    df_users = pd.DataFrame(user_data)
    st.dataframe(df_users, use_container_width=True)
    
    if simulate_attack:
        st.error("""
        ⚠️ **Anomaly Detected: DataResearch_Co**
        
        - Query volume: {:.0f}x above normal
        - Pattern: Systematic variations detected
        - Risk: Possible model extraction attempt
        - Recommendation: Block account and investigate
        """.format(df_users[df_users['User']=='DataResearch_Co']['Total Queries'].iloc[0] / 
                   df_users[df_users['User']!='DataResearch_Co']['Total Queries'].mean()))
    
    st.divider()
    
    # Attack pattern analysis
    st.subheader("🎯 Attack Pattern Distribution")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        attack_types = {
            'Prompt Injection': df['prompt_injection_attempts'].sum(),
            'Adversarial Inputs': df['adversarial_detected'].sum(),
            'Auth Failures': df['auth_failures'].sum(),
            'Data Leakage': df['data_leakage_events'].sum()
        }
        
        fig_pie = px.pie(
            values=list(attack_types.values()),
            names=list(attack_types.keys()),
            title="Security Events by Type"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col_b:
        # Time distribution
        hourly_events = df.groupby(df['timestamp'].dt.hour)['prompt_injection_attempts'].sum()
        
        fig_hour = px.bar(
            x=hourly_events.index,
            y=hourly_events.values,
            title="Security Events by Hour",
            labels={'x': 'Hour', 'y': 'Events'}
        )
        st.plotly_chart(fig_hour, use_container_width=True)

with tab4:
    st.header("Incident Response Playbooks")
    
    st.write("""
    Pre-defined response procedures for common security incidents.
    These playbooks ensure consistent, effective responses to threats.
    """)
    
    with st.expander("🔴 Playbook 1: Model Extraction Attack", expanded=True):
        st.write("""
        **Detection Indicators:**
        - High query volume from single account
        - Systematic input variations
        - Queries distributed over time to evade rate limits
        - Multiple accounts from similar IPs
        
        **Response Procedure:**
        
        **IMMEDIATE (0-5 minutes):**
        1. ✅ Block suspicious accounts automatically
        2. ✅ Enable stricter rate limiting
        3. ✅ Log all queries for forensic analysis
        4. ✅ Alert security team via Slack/PagerDuty
        
        **SHORT-TERM (5-30 minutes):**
        1. 🔍 Analyze query patterns to confirm attack
        2. 🔍 Identify all related accounts/IPs
        3. 🔍 Assess if any model data was leaked
        4. 📧 Notify incident commander
        
        **MEDIUM-TERM (30 min - 24 hours):**
        1. 🛡️ Implement additional protections:
           - Output perturbation
           - Query similarity detection
           - IP reputation checks
        2. 📊 Generate incident report
        3. 🔒 Review and update rate limiting rules
        4. 📝 Document lessons learned
        
        **POST-INCIDENT:**
        1. Conduct root cause analysis
        2. Update detection rules
        3. Enhance monitoring
        4. Train team on new patterns
        """)
    
    with st.expander("🟠 Playbook 2: Prompt Injection Campaign"):
        st.write("""
        **Detection Indicators:**
        - Spike in prompt injection attempts
        - Common patterns across multiple users
        - Attempts to bypass safety filters
        - Unusual command-like inputs
        
        **Response Procedure:**
        
        **IMMEDIATE:**
        1. ✅ Enable enhanced input filtering
        2. ✅ Log all injection attempts
        3. ✅ Increase prompt validation strictness
        
        **SHORT-TERM:**
        1. 🔍 Analyze injection techniques used
        2. 🔍 Test if any bypassed filters
        3. 🔍 Check if any data was leaked
        4. 🔍 Identify attack source
        
        **MEDIUM-TERM:**
        1. 🛡️ Update prompt injection defenses
        2. 🛡️ Retrain input classifiers
        3. 🛡️ Add new attack patterns to test suite
        4. 📊 Update security documentation
        
        **POST-INCIDENT:**
        1. Add test cases for new patterns
        2. Update adversarial training data
        3. Review system prompt security
        4. Share findings with AI security community
        """)
    
    with st.expander("🟡 Playbook 3: Authentication Brute Force"):
        st.write("""
        **Detection Indicators:**
        - Multiple failed login attempts
        - Attempts from same IP or IP range
        - Sequential or dictionary-based credentials
        - Rapid-fire authentication requests
        
        **Response Procedure:**
        
        **IMMEDIATE:**
        1. ✅ Block source IPs after threshold
        2. ✅ Implement exponential backoff
        3. ✅ Require CAPTCHA for affected accounts
        4. ✅ Alert affected users
        
        **SHORT-TERM:**
        1. 🔍 Check if any accounts compromised
        2. 🔍 Analyze attack patterns
        3. 🔍 Identify targeted accounts
        4. 🔒 Force password reset for at-risk accounts
        
        **MEDIUM-TERM:**
        1. 🛡️ Enhance authentication security:
           - Require MFA for all users
           - Implement device fingerprinting
           - Add behavioral analysis
        2. 📊 Review access logs
        3. 📧 User security awareness campaign
        
        **POST-INCIDENT:**
        1. Update password policies
        2. Review authentication infrastructure
        3. Implement additional fraud detection
        4. Test incident response time
        """)
    
    with st.expander("🟢 Playbook 4: Data Leakage Event"):
        st.write("""
        **Detection Indicators:**
        - Sensitive data in model outputs
        - PII exposure
        - Training data memorization
        - Customer data in responses
        
        **Response Procedure:**
        
        **IMMEDIATE (CRITICAL):**
        1. 🚨 Take affected model offline
        2. 🚨 Block all API access to affected endpoints
        3. 🚨 Alert legal and compliance teams
        4. 🚨 Preserve all logs for investigation
        
        **SHORT-TERM:**
        1. 🔍 Identify all exposed data
        2. 🔍 Determine scope of breach
        3. 🔍 Track who accessed leaked data
        4. 📧 Notify affected parties (GDPR/CCPA requirements)
        
        **MEDIUM-TERM:**
        1. 🛡️ Fix data leakage vulnerability
        2. 🛡️ Implement output filtering
        3. 🛡️ Retrain model with differential privacy
        4. 🛡️ Add leakage detection to CI/CD
        5. 📊 Regulatory reporting
        
        **POST-INCIDENT:**
        1. Conduct data privacy audit
        2. Review training data handling
        3. Implement data minimization
        4. Update privacy policies
        5. Third-party security review
        """)

# Auto-refresh
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()

# Footer
st.divider()
st.caption("📊 **Security Monitoring Dashboard** - Continuous validation of AI systems in production")
st.caption("Monitor, detect, and respond to security threats in real-time")
