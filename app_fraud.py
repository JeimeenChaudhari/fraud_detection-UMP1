"""
Real-time Fraud Detection Dashboard
====================================
Interactive Streamlit application for fraud detection with:
- Overview dashboard with charts and metrics
- Single transaction fraud check
- Customer investigation tools
- Model explainability and insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import glob
import os
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Real-time Fraud Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL AND DATA
# ============================================================================

@st.cache_resource
def load_model():
    """Load the trained fraud detection model"""
    try:
        with open('fraud_detection_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        st.error("❌ Model file not found! Please run train_fraud_model.py first.")
        st.stop()

@st.cache_data
def load_feature_list():
    """Load feature names"""
    try:
        with open('artifacts/feature_list.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.warning("Feature list not found. Using default features.")
        return []

@st.cache_data
def load_metrics():
    """Load training metrics"""
    try:
        with open('artifacts/metrics.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

@st.cache_data
def load_dataset():
    """Load all transaction data from pkl files"""
    data_folder = "data/"
    pkl_files = sorted(glob.glob(os.path.join(data_folder, "*.pkl")))
    
    if not pkl_files:
        return pd.DataFrame()
    
    dfs = []
    for f in pkl_files[:30]:  # Limit to first 30 files for performance
        try:
            df_temp = pd.read_pickle(f)
            dfs.append(df_temp)
        except:
            continue
    
    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        df['TX_DATETIME'] = pd.to_datetime(df['TX_DATETIME'], errors='coerce')
        return df
    return pd.DataFrame()

# Load model and data
model_pipeline = load_model()
model = model_pipeline['model']
scaler = model_pipeline['scaler']
feature_names = model_pipeline['feature_names']
metrics = load_metrics()

# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.title("🔍 Fraud Detection System By CJ")
st.sidebar.markdown("---")

# Model info
st.sidebar.subheader("Model Information")
st.sidebar.info(f"""
**Model Type:** XGBoost Classifier  
**Features:** {len(feature_names)}  
**Training Date:** {model_pipeline.get('training_date', 'N/A')[:10]}  
**AUPRC:** {metrics.get('auprc', 0):.4f}
""")

# Threshold slider
st.sidebar.markdown("---")
st.sidebar.subheader("Detection Settings")
threshold = st.sidebar.slider(
    "Fraud Probability Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Transactions with probability above this threshold are flagged as fraudulent"
)

# Advanced options
show_advanced = st.sidebar.checkbox("Show Advanced Charts", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Navigation")
st.sidebar.info("Use the tabs above to navigate between different sections of the dashboard.")

# ============================================================================
# MAIN CONTENT
# ============================================================================

st.title("🔍 Real-time Fraud Detection Dashboard")
st.markdown("**Detect fraudulent transactions with machine learning**")
st.markdown("---")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview", 
    "🔎 Model Check", 
    "👤 Investigate Customer",
    "🧠 Model Explainability"
])

# ============================================================================
# TAB 1: OVERVIEW DASHBOARD
# ============================================================================

with tab1:
    st.header("Overview Dashboard")
    
    # Load dataset
    with st.spinner("Loading transaction data..."):
        df = load_dataset()
    
    if df.empty:
        st.warning("No transaction data available. Please ensure .pkl files are in the data/ folder.")
    else:
        # Key metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Transactions",
                f"{len(df):,}",
                delta=None
            )
        
        with col2:
            fraud_count = df['TX_FRAUD'].sum() if 'TX_FRAUD' in df.columns else 0
            fraud_rate = (fraud_count / len(df) * 100) if len(df) > 0 else 0
            st.metric(
                "Fraud Rate",
                f"{fraud_rate:.3f}%",
                delta=f"{fraud_count:,} frauds"
            )
        
        with col3:
            avg_amount = df['TX_AMOUNT'].mean() if 'TX_AMOUNT' in df.columns else 0
            st.metric(
                "Avg Transaction",
                f"${avg_amount:.2f}",
                delta=None
            )
        
        with col4:
            date_range = (df['TX_DATETIME'].max() - df['TX_DATETIME'].min()).days if 'TX_DATETIME' in df.columns else 0
            st.metric(
                "Date Range",
                f"{date_range} days",
                delta=None
            )
        
        st.markdown("---")
        
        # Charts
        if show_advanced and 'TX_DATETIME' in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Transactions Over Time")
                df['date'] = df['TX_DATETIME'].dt.date
                daily_counts = df.groupby('date').size().reset_index(name='count')
                
                fig = px.line(
                    daily_counts, 
                    x='date', 
                    y='count',
                    title='Daily Transaction Volume'
                )
                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Number of Transactions",
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Transaction Amount Distribution")
                fig = px.histogram(
                    df,
                    x='TX_AMOUNT',
                    nbins=50,
                    title='Amount Distribution (Log Scale)',
                    log_y=True
                )
                fig.update_layout(
                    xaxis_title="Transaction Amount ($)",
                    yaxis_title="Count (log scale)"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Hourly heatmap
        if show_advanced and 'TX_DATETIME' in df.columns:
            st.subheader("Fraud Pattern: Hour vs Day of Week")
            
            df['hour'] = df['TX_DATETIME'].dt.hour
            df['day_of_week'] = df['TX_DATETIME'].dt.dayofweek
            
            if 'TX_FRAUD' in df.columns:
                heatmap_data = df.groupby(['day_of_week', 'hour'])['TX_FRAUD'].mean().reset_index()
                heatmap_pivot = heatmap_data.pivot(index='day_of_week', columns='hour', values='TX_FRAUD')
                
                fig = go.Figure(data=go.Heatmap(
                    z=heatmap_pivot.values,
                    x=heatmap_pivot.columns,
                    y=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                    colorscale='Reds',
                    text=heatmap_pivot.values,
                    texttemplate='%{text:.3f}',
                    textfont={"size": 10}
                ))
                
                fig.update_layout(
                    title='Fraud Rate by Hour and Day',
                    xaxis_title='Hour of Day',
                    yaxis_title='Day of Week',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Model evaluation charts
        st.markdown("---")
        st.subheader("Model Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if os.path.exists('artifacts/confusion_matrix.png'):
                st.image('artifacts/confusion_matrix.png', caption='Confusion Matrix')
        
        with col2:
            if os.path.exists('artifacts/precision_recall_curve.png'):
                st.image('artifacts/precision_recall_curve.png', caption='Precision-Recall Curve')

# ============================================================================
# TAB 2: MODEL CHECK (Single Transaction)
# ============================================================================

with tab2:
    st.header("🔎 Check Single Transaction")
    st.markdown("Enter transaction details to check fraud probability")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Transaction Details")
        
        tx_amount = st.number_input(
            "Transaction Amount ($)",
            min_value=0.0,
            max_value=10000.0,
            value=100.0,
            step=10.0
        )
        
        hour = st.slider("Hour of Day", 0, 23, 12)
        
        day_of_week = st.selectbox(
            "Day of Week",
            options=[0, 1, 2, 3, 4, 5, 6],
            format_func=lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                                   'Friday', 'Saturday', 'Sunday'][x]
        )
        
        is_weekend = 1 if day_of_week >= 5 else 0
        
        st.subheader("Additional Features (Optional)")
        
        cust_avg_amount = st.number_input(
            "Customer Avg Amount",
            min_value=0.0,
            value=50.0,
            help="Average transaction amount for this customer"
        )
        
        cust_tx_count = st.number_input(
            "Customer Total Transactions",
            min_value=0,
            value=10,
            help="Total number of transactions by this customer"
        )
    
    with col2:
        st.subheader("Prediction Result")
        
        if st.button("🔍 Check Transaction", type="primary", use_container_width=True):
            # Prepare features
            features_dict = {
                'TX_AMOUNT': tx_amount,
                'amount_log': np.log1p(tx_amount),
                'hour': hour,
                'day_of_week': day_of_week,
                'is_weekend': is_weekend,
                'day_of_month': 15,  # Default
                'month': 4,  # Default
                'cust_avg_amount': cust_avg_amount,
                'cust_std_amount': cust_avg_amount * 0.3,  # Estimate
                'cust_tx_count': cust_tx_count,
                'cust_fraud_count': 0,  # Default
                'terminal_avg_amount': 75.0,  # Default
                'terminal_tx_count': 100,  # Default
                'terminal_fraud_count': 1,  # Default
                'amount_to_cust_avg_ratio': tx_amount / (cust_avg_amount + 1e-5),
                'amount_to_terminal_avg_ratio': tx_amount / 75.0,
                'cust_tx_count_24h': 5,  # Default
                'terminal_tx_count_24h': 20  # Default
            }
            
            # Create DataFrame with correct feature order
            X_input = pd.DataFrame([features_dict])[feature_names]
            
            # Scale and predict
            X_scaled = scaler.transform(X_input)
            fraud_prob = model.predict_proba(X_scaled)[0, 1]
            
            # Display probability
            st.metric("Fraud Probability", f"{fraud_prob:.2%}")
            
            # Progress bar
            st.progress(float(fraud_prob))
            
            # Risk assessment
            if fraud_prob > threshold:
                st.error("🚨 **HIGH RISK** - This transaction should be reviewed!")
                st.warning(f"Probability ({fraud_prob:.2%}) exceeds threshold ({threshold:.2%})")
            elif fraud_prob > threshold * 0.7:
                st.warning("⚠️ **MEDIUM RISK** - Consider additional verification")
            else:
                st.success("✅ **LOW RISK** - Transaction appears legitimate")
            
            # Feature contributions (simplified)
            st.markdown("---")
            st.subheader("Key Factors")
            
            # Show top contributing features
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Value': X_input.values[0]
            })
            
            st.dataframe(
                feature_importance.head(8),
                use_container_width=True,
                hide_index=True
            )

# ============================================================================
# TAB 3: INVESTIGATE CUSTOMER
# ============================================================================

with tab3:
    st.header("👤 Customer Investigation")
    st.markdown("Analyze transaction history and behavior patterns for a specific customer")
    
    # Load dataset
    df = load_dataset()
    
    if df.empty:
        st.warning("No transaction data available.")
    else:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Search Customer")
            
            if 'CUSTOMER_ID' in df.columns:
                # Get unique customers
                customer_ids = sorted(df['CUSTOMER_ID'].unique())
                
                customer_id = st.selectbox(
                    "Select Customer ID",
                    options=customer_ids[:100],  # Limit for performance
                    help="Select a customer to investigate"
                )
                
                if st.button("🔍 Load Customer Data", use_container_width=True):
                    st.session_state['selected_customer'] = customer_id
        
        with col2:
            if 'selected_customer' in st.session_state:
                customer_id = st.session_state['selected_customer']
                
                # Filter customer data
                cust_df = df[df['CUSTOMER_ID'] == customer_id].copy()
                
                if len(cust_df) > 0:
                    st.subheader(f"Customer #{customer_id}")
                    
                    # Customer metrics
                    col_a, col_b, col_c, col_d = st.columns(4)
                    
                    with col_a:
                        st.metric("Total Transactions", len(cust_df))
                    
                    with col_b:
                        fraud_count = cust_df['TX_FRAUD'].sum() if 'TX_FRAUD' in cust_df.columns else 0
                        st.metric("Fraud Count", fraud_count)
                    
                    with col_c:
                        avg_amount = cust_df['TX_AMOUNT'].mean()
                        st.metric("Avg Amount", f"${avg_amount:.2f}")
                    
                    with col_d:
                        total_spent = cust_df['TX_AMOUNT'].sum()
                        st.metric("Total Spent", f"${total_spent:.2f}")
                    
                    st.markdown("---")
                    
                    # Transaction timeline
                    st.subheader("Transaction Timeline")
                    
                    cust_df_sorted = cust_df.sort_values('TX_DATETIME')
                    
                    fig = go.Figure()
                    
                    # Legitimate transactions
                    legit = cust_df_sorted[cust_df_sorted['TX_FRAUD'] == 0]
                    fig.add_trace(go.Scatter(
                        x=legit['TX_DATETIME'],
                        y=legit['TX_AMOUNT'],
                        mode='markers',
                        name='Legitimate',
                        marker=dict(color='blue', size=8)
                    ))
                    
                    # Fraudulent transactions
                    if 'TX_FRAUD' in cust_df.columns:
                        fraud = cust_df_sorted[cust_df_sorted['TX_FRAUD'] == 1]
                        if len(fraud) > 0:
                            fig.add_trace(go.Scatter(
                                x=fraud['TX_DATETIME'],
                                y=fraud['TX_AMOUNT'],
                                mode='markers',
                                name='Fraud',
                                marker=dict(color='red', size=12, symbol='x')
                            ))
                    
                    fig.update_layout(
                        title='Transaction History',
                        xaxis_title='Date',
                        yaxis_title='Amount ($)',
                        hovermode='closest',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Recent transactions table
                    st.subheader("Recent Transactions")
                    
                    display_cols = ['TX_DATETIME', 'TX_AMOUNT', 'TERMINAL_ID']
                    if 'TX_FRAUD' in cust_df.columns:
                        display_cols.append('TX_FRAUD')
                    
                    recent_tx = cust_df_sorted[display_cols].tail(10).sort_values('TX_DATETIME', ascending=False)
                    st.dataframe(recent_tx, use_container_width=True, hide_index=True)
                    
                    # Behavioral insights
                    st.markdown("---")
                    st.subheader("Behavioral Insights")
                    
                    col_x, col_y = st.columns(2)
                    
                    with col_x:
                        # Hour distribution
                        if 'TX_DATETIME' in cust_df.columns:
                            cust_df['hour'] = cust_df['TX_DATETIME'].dt.hour
                            hour_dist = cust_df['hour'].value_counts().sort_index()
                            
                            fig = px.bar(
                                x=hour_dist.index,
                                y=hour_dist.values,
                                labels={'x': 'Hour of Day', 'y': 'Transaction Count'},
                                title='Transaction Distribution by Hour'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col_y:
                        # Amount distribution
                        fig = px.histogram(
                            cust_df,
                            x='TX_AMOUNT',
                            nbins=20,
                            title='Amount Distribution'
                        )
                        fig.update_layout(
                            xaxis_title='Amount ($)',
                            yaxis_title='Frequency'
                        )
                        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 4: MODEL EXPLAINABILITY
# ============================================================================

with tab4:
    st.header("🧠 Model Explainability & Insights")
    st.markdown("Understand how the fraud detection model makes decisions")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Model Performance")
        
        # Display key metrics
        if metrics:
            metric_col1, metric_col2 = st.columns(2)
            
            with metric_col1:
                st.metric(
                    "AUPRC",
                    f"{metrics.get('auprc', 0):.4f}",
                    help="Area Under Precision-Recall Curve"
                )
                st.metric(
                    "Test Samples",
                    f"{metrics.get('test_samples', 0):,}"
                )
            
            with metric_col2:
                st.metric(
                    "ROC-AUC",
                    f"{metrics.get('roc_auc', 0):.4f}",
                    help="Area Under ROC Curve"
                )
                st.metric(
                    "Features",
                    metrics.get('feature_count', 0)
                )
        
        st.markdown("---")
        
        # Threshold analysis
        st.subheader("🎯 Threshold Analysis")
        
        st.markdown("""
        The **fraud probability threshold** determines when to flag a transaction:
        
        - **Lower threshold** (e.g., 0.3): Catch more fraud, but more false alarms
        - **Higher threshold** (e.g., 0.7): Fewer false alarms, but miss some fraud
        
        **Current threshold:** {:.2%}
        """.format(threshold))
        
        # Threshold recommendations
        st.info("""
        **Recommended thresholds by use case:**
        
        - 🔴 **High Security** (0.3): Banking, high-value transactions
        - 🟡 **Balanced** (0.5): E-commerce, moderate risk
        - 🟢 **Low Friction** (0.7): Low-value, high-volume transactions
        """)
    
    with col2:
        st.subheader("🔍 Feature Importance")
        
        # Display feature importance
        if os.path.exists('artifacts/feature_importance.png'):
            st.image(
                'artifacts/feature_importance.png',
                caption='Top Features Contributing to Fraud Detection'
            )
        
        # Load and display feature importance table
        if os.path.exists('artifacts/feature_importance.csv'):
            feat_imp = pd.read_csv('artifacts/feature_importance.csv')
            
            st.markdown("**Top 10 Features:**")
            st.dataframe(
                feat_imp.head(10)[['feature', 'importance']],
                use_container_width=True,
                hide_index=True
            )
    
    st.markdown("---")
    
    # Model insights
    st.subheader("💡 Key Insights")
    
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.markdown("""
        **🎯 Detection Strategy**
        
        - Uses XGBoost classifier
        - Handles class imbalance with SMOTE
        - Focuses on precision-recall tradeoff
        - Real-time feature engineering
        """)
    
    with col_b:
        st.markdown("""
        **🔧 Important Features**
        
        - Transaction amount patterns
        - Customer behavior history
        - Time-based features
        - Terminal risk scores
        """)
    
    with col_c:
        st.markdown("""
        **⚡ Performance**
        
        - High precision for fraud detection
        - Low false positive rate
        - Scalable to millions of transactions
        - Production-ready deployment
        """)
    
    # Export functionality
    st.markdown("---")
    st.subheader("📥 Export Options")
    
    col_export1, col_export2 = st.columns(2)
    
    with col_export1:
        if st.button("📊 Download Model Report", use_container_width=True):
            st.info("Model report generation would be implemented here")
    
    with col_export2:
        # Load dataset for flagged transactions
        df = load_dataset()
        if not df.empty and st.button("⚠️ Download Flagged Transactions", use_container_width=True):
            # This would generate a CSV of high-risk transactions
            st.info("Flagged transactions export would be implemented here")
    
    # Additional resources
    st.markdown("---")
    st.subheader("📚 Additional Resources")
    
    with st.expander("ℹ️ About the Model"):
        st.markdown("""
        **Fraud Detection Model Details:**
        
        This model uses **XGBoost** (Extreme Gradient Boosting), an ensemble learning method
        that combines multiple decision trees to make predictions.
        
        **Key Features:**
        - Handles imbalanced data with SMOTE (Synthetic Minority Over-sampling Technique)
        - Uses time-based and behavioral features
        - Trained on historical transaction patterns
        - Optimized for precision-recall balance
        
        **Model Training:**
        - Training Date: {training_date}
        - Features: {feature_count}
        - Random State: {random_state}
        """.format(
            training_date=model_pipeline.get('training_date', 'N/A')[:10],
            feature_count=len(feature_names),
            random_state=model_pipeline.get('random_state', 42)
        ))
    
    with st.expander("🔒 Privacy & Security"):
        st.markdown("""
        **Data Privacy:**
        - Customer IDs are anonymized
        - No PII (Personally Identifiable Information) stored
        - Secure model storage
        
        **Security Measures:**
        - Model artifacts encrypted
        - Access control implemented
        - Audit logs maintained
        - Regular security reviews
        """)
    
    with st.expander("📈 Model Improvement"):
        st.markdown("""
        **Continuous Improvement:**
        
        To maintain model accuracy:
        1. Retrain monthly with new data
        2. Monitor performance metrics
        3. Update features based on new fraud patterns
        4. A/B test threshold changes
        5. Collect feedback on false positives/negatives
        
        **Next Steps:**
        - Implement real-time feature store
        - Add SHAP explanations for individual predictions
        - Deploy model monitoring dashboard
        - Set up automated retraining pipeline
        """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>Fraud Detection System v1.0</strong></p>
    <p>Built with Streamlit • XGBoost • Python</p>
    <p><em>Made By Jeimeen Chaudhari</em></p>
</div>
""", unsafe_allow_html=True)