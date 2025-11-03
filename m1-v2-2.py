"""
Module 1 Video 2 Part 2: Model Extraction Attack Pattern Demo
Demonstrates how model extraction works through systematic API queries
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

# Page config
st.set_page_config(
    page_title="Model Extraction Demo",
    page_icon="🕵️",
    layout="wide"
)

st.title("🕵️ Model Extraction Attack Pattern Demonstration")
st.caption("Module 1 - Video 2 - Part 2: Understanding Model Extraction Through API Abuse")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Simulation Settings")
    
    st.subheader("Normal User Pattern")
    normal_users = st.slider("Number of Normal Users", 10, 100, 50)
    normal_query_rate = st.slider("Normal Query Rate (per minute)", 1, 10, 3)
    
    st.divider()
    
    st.subheader("Attacker Pattern")
    attacker_enabled = st.checkbox("Enable Attacker Simulation", value=True)
    attacker_query_rate = st.slider("Attacker Query Rate (per minute)", 10, 200, 100)
    systematic_variation = st.checkbox("Show Systematic Variations", value=True)
    
    st.divider()
    
    simulation_speed = st.selectbox("Simulation Speed", ["Slow", "Medium", "Fast"], index=1)
    speed_map = {"Slow": 2, "Medium": 1, "Fast": 0.5}

# Generate sample data
def generate_normal_traffic(num_users, duration_minutes=30):
    """Generate realistic normal user traffic"""
    data = []
    start_time = datetime.now() - timedelta(minutes=duration_minutes)
    
    for user_id in range(num_users):
        # Random number of queries per user (1-10)
        num_queries = np.random.randint(1, 11)
        
        for _ in range(num_queries):
            # Random time within the duration
            query_time = start_time + timedelta(
                minutes=np.random.randint(0, duration_minutes),
                seconds=np.random.randint(0, 60)
            )
            
            data.append({
                'timestamp': query_time,
                'user_id': f'USER_{user_id:04d}',
                'query_type': np.random.choice(['product_info', 'pricing', 'support', 'general']),
                'response_time': np.random.uniform(0.1, 0.5),
                'user_type': 'normal'
            })
    
    return pd.DataFrame(data)

def generate_attacker_traffic(duration_minutes=30):
    """Generate systematic attacker traffic pattern"""
    data = []
    start_time = datetime.now() - timedelta(minutes=duration_minutes)
    
    # Attacker uses multiple accounts to avoid detection
    attacker_accounts = ['DataResearch_Co', 'ML_Analysis_Lab', 'AI_Consulting_Inc']
    
    # Systematic queries every few seconds
    num_queries = duration_minutes * 100  # 100 queries per minute
    
    for i in range(num_queries):
        query_time = start_time + timedelta(seconds=i * 0.6)  # Every 0.6 seconds
        
        # Rotate through accounts
        account = attacker_accounts[i % len(attacker_accounts)]
        
        # Systematic variation in queries
        variation_type = ['brightness_+5%', 'rotation_10deg', 'crop_top_left', 
                         'contrast_+10%', 'scale_0.95x', 'blur_gaussian'][i % 6]
        
        data.append({
            'timestamp': query_time,
            'user_id': account,
            'query_type': f'systematic_{variation_type}',
            'response_time': np.random.uniform(0.1, 0.3),
            'user_type': 'attacker',
            'variation_type': variation_type
        })
    
    return pd.DataFrame(data)

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📊 Live Pattern Analysis", "🔍 Attack Signatures", "📈 Statistical Analysis", "🎓 Learn More"])

with tab1:
    st.header("Query Pattern Comparison")
    
    # Generate traffic
    normal_df = generate_normal_traffic(normal_users)
    attacker_df = generate_attacker_traffic() if attacker_enabled else pd.DataFrame()
    
    # Combine data
    if not attacker_df.empty:
        all_traffic = pd.concat([normal_df, attacker_df])
    else:
        all_traffic = normal_df
    
    # Time series visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Normal User Traffic")
        
        # Aggregate by minute
        normal_by_minute = normal_df.groupby(normal_df['timestamp'].dt.floor('1min')).size().reset_index(name='queries')
        
        fig_normal = go.Figure()
        fig_normal.add_trace(go.Scatter(
            x=normal_by_minute['timestamp'],
            y=normal_by_minute['queries'],
            mode='lines+markers',
            name='Normal Queries',
            line=dict(color='#2ecc71', width=2),
            marker=dict(size=6)
        ))
        
        fig_normal.update_layout(
            title="Normal Query Pattern (Random)",
            xaxis_title="Time",
            yaxis_title="Queries per Minute",
            height=300,
            showlegend=False
        )
        
        st.plotly_chart(fig_normal, use_container_width=True)
        
        st.success("✅ **Characteristics:**")
        st.write("- Random timing")
        st.write("- Variable query types")
        st.write("- Low, inconsistent volume")
        st.write("- Multiple different users")
    
    with col2:
        st.subheader("Suspicious Traffic")
        
        if not attacker_df.empty:
            # Aggregate by minute
            attacker_by_minute = attacker_df.groupby(attacker_df['timestamp'].dt.floor('1min')).size().reset_index(name='queries')
            
            fig_attacker = go.Figure()
            fig_attacker.add_trace(go.Scatter(
                x=attacker_by_minute['timestamp'],
                y=attacker_by_minute['queries'],
                mode='lines+markers',
                name='Suspicious Queries',
                line=dict(color='#e74c3c', width=2),
                marker=dict(size=6)
            ))
            
            fig_attacker.update_layout(
                title="Attacker Query Pattern (Systematic)",
                xaxis_title="Time",
                yaxis_title="Queries per Minute",
                height=300,
                showlegend=False
            )
            
            st.plotly_chart(fig_attacker, use_container_width=True)
            
            st.error("⚠️ **Red Flags:**")
            st.write("- Consistent, high-volume")
            st.write("- Systematic variations")
            st.write("- Even time distribution")
            st.write("- Few accounts, many queries")
        else:
            st.info("Enable attacker simulation in sidebar to see suspicious patterns")

with tab2:
    st.header("Attack Signature Detection")
    
    if not attacker_df.empty:
        st.subheader("🔴 Detected Attack Patterns")
        
        # Show systematic variations
        if systematic_variation:
            st.write("**Systematic Input Variations Detected:**")
            
            variation_counts = attacker_df['variation_type'].value_counts()
            
            fig_variations = px.bar(
                x=variation_counts.index,
                y=variation_counts.values,
                labels={'x': 'Variation Type', 'y': 'Number of Queries'},
                title="Systematic Query Variations (Model Extraction Signature)"
            )
            fig_variations.update_traces(marker_color='#e74c3c')
            st.plotly_chart(fig_variations, use_container_width=True)
            
            st.error("""
            **🚨 ALERT: Model Extraction Attack Detected**
            
            The attacker is systematically varying inputs to probe decision boundaries:
            - Testing same input with brightness variations
            - Testing rotations to understand rotation invariance
            - Testing crops to identify important image regions
            - Testing contrast to learn feature sensitivity
            
            **Attacker Goal:** Collect enough input-output pairs to replicate the model
            """)
        
        # Show query concentration
        st.subheader("Query Volume by Account")
        
        account_queries = attacker_df.groupby('user_id').size().reset_index(name='total_queries')
        
        fig_accounts = px.bar(
            account_queries,
            x='user_id',
            y='total_queries',
            title="Queries per Account (30-minute window)",
            labels={'total_queries': 'Number of Queries', 'user_id': 'Account'}
        )
        fig_accounts.update_traces(marker_color='#e67e22')
        st.plotly_chart(fig_accounts, use_container_width=True)
        
        # Calculate cost vs value
        st.subheader("💰 Attack Economics")
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            queries_total = len(attacker_df)
            api_cost = queries_total * 0.002  # $0.002 per query
            st.metric("Total Queries", f"{queries_total:,}")
            st.caption(f"API Cost: ${api_cost:.2f}")
        
        with col_b:
            # Assuming model training cost $5M
            st.metric("Your Model Value", "$5,000,000")
            st.caption("2 years development")
        
        with col_c:
            theft_percentage = (api_cost / 5000000) * 100
            st.metric("Attack Cost", f"{theft_percentage:.4f}%")
            st.caption("of model value")
        
        st.error(f"""
        **Attack Cost Analysis:**
        - Attacker spent: **${api_cost:.2f}** in API costs
        - Model value: **$5,000,000**
        - Cost to steal: **{theft_percentage:.4f}%** of model value
        - ROI for attacker: **{(5000000/api_cost):.0f}x return**
        """)
    else:
        st.info("Enable attacker simulation to see attack signatures")

with tab3:
    st.header("Statistical Anomaly Detection")
    
    if len(all_traffic) > 0:
        # Query rate distribution
        st.subheader("Query Rate Distribution by User Type")
        
        # Calculate queries per user
        user_query_counts = all_traffic.groupby(['user_id', 'user_type']).size().reset_index(name='query_count')
        
        fig_dist = px.box(
            user_query_counts,
            x='user_type',
            y='query_count',
            color='user_type',
            title="Query Count Distribution",
            labels={'query_count': 'Queries per User', 'user_type': 'User Type'}
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Statistical summary
        st.subheader("Statistical Summary")
        
        normal_stats = user_query_counts[user_query_counts['user_type'] == 'normal']['query_count']
        
        if not attacker_df.empty:
            attacker_stats = user_query_counts[user_query_counts['user_type'] == 'attacker']['query_count']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Normal Users:**")
                st.write(f"- Mean queries: {normal_stats.mean():.1f}")
                st.write(f"- Median queries: {normal_stats.median():.1f}")
                st.write(f"- Std deviation: {normal_stats.std():.1f}")
                st.write(f"- Max queries: {normal_stats.max()}")
            
            with col2:
                st.write("**Attacker Accounts:**")
                st.write(f"- Mean queries: {attacker_stats.mean():.1f}")
                st.write(f"- Median queries: {attacker_stats.median():.1f}")
                st.write(f"- Std deviation: {attacker_stats.std():.1f}")
                st.write(f"- Max queries: {attacker_stats.max()}")
            
            # Z-score analysis
            mean_normal = normal_stats.mean()
            std_normal = normal_stats.std()
            attacker_mean = attacker_stats.mean()
            
            z_score = (attacker_mean - mean_normal) / std_normal
            
            st.error(f"""
            **Anomaly Score:**
            - Attacker query rate is **{z_score:.1f} standard deviations** above normal
            - Probability this is legitimate: **< 0.001%**
            - **RECOMMENDATION: Block accounts and investigate**
            """)

with tab4:
    st.header("Understanding Model Extraction")
    
    with st.expander("What is Model Extraction?"):
        st.write("""
        Model extraction (also called model stealing) is an attack where adversaries query a machine learning 
        model systematically to learn its behavior and create a functionally equivalent copy.
        
        **How it works:**
        1. **Probing**: Attacker sends systematic queries to the model API
        2. **Learning**: Attacker collects input-output pairs
        3. **Replication**: Attacker trains their own model on the collected data
        4. **Theft Complete**: Attacker has a working copy without development costs
        """)
    
    with st.expander("Why is this Dangerous?"):
        st.write("""
        **Intellectual Property Theft:**
        - Your model represents years of work and millions in investment
        - Competitors can copy it for pennies on the dollar
        - No code access needed - just API access
        
        **Business Impact:**
        - Loss of competitive advantage
        - Revenue loss from competitors using your model
        - Potential regulatory violations (if model was trained on licensed data)
        
        **Real-World Examples:**
        - Image classification models extracted via API
        - Speech recognition systems replicated
        - Recommendation algorithms copied
        """)
    
    with st.expander("Detection Strategies"):
        st.write("""
        **1. Query Pattern Analysis:**
        - Monitor for systematic variations in inputs
        - Detect unusual query volumes per user
        - Identify repeated queries with minor modifications
        
        **2. Statistical Anomaly Detection:**
        - Establish baseline query rates
        - Alert on outliers (>3 standard deviations)
        - Track query diversity per account
        
        **3. Rate Limiting (Smart):**
        - Not just volume-based
        - Consider query similarity
        - Account for systematic patterns
        
        **4. Watermarking:**
        - Embed signatures in model outputs
        - Detect if extracted models carry watermarks
        - Prove theft in court if needed
        """)
    
    with st.expander("Prevention Measures"):
        st.write("""
        **API-Level Defenses:**
        - Implement intelligent rate limiting
        - Add output perturbation (noise injection)
        - Require authentication and authorization
        - Monitor query patterns in real-time
        
        **Model-Level Defenses:**
        - Add prediction uncertainty to outputs
        - Limit output precision
        - Use model watermarking
        - Implement query auditing
        
        **Business-Level Defenses:**
        - Tiered API access with monitoring
        - Legal agreements with harsh penalties
        - Regular security audits
        - Incident response plans
        """)

# Footer
st.divider()
st.caption("⚠️ **Educational Demo** - This simulation demonstrates model extraction patterns for security training purposes.")
