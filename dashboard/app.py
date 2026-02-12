import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import threading
import uvicorn
import sys
import time
from pathlib import Path

# Add apps directory to sys.path to allow importing src
apps_dir = Path(__file__).resolve().parent.parent
if str(apps_dir) not in sys.path:
    sys.path.append(str(apps_dir))

# Import FastAPI app (must be after sys.path update)
try:
    from src.api.main import app as fastapi_app
except ImportError as e:
    st.error(f"Failed to import API: {e}")
    st.stop()

# --- API Background Server ---
def run_api():
    """Run FastAPI in a separate thread"""
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000, log_level="error")

@st.cache_resource
def start_api_server():
    """Start API server in background (singleton)"""
    thread = threading.Thread(target=run_api, daemon=True)
    thread.start()
    # Give it a moment to start
    time.sleep(2)
    return thread

# Start API automatically
try:
    start_api_server()
except Exception as e:
    st.error(f"Failed to start background API: {e}")

# Page config
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API endpoint
API_URL = "http://localhost:8001"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🎭 Sentiment Analysis Dashboard</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    st.success("✅ Prediction API Running")
    
    # Model selection
    model = st.selectbox(
        "Select Model",
        ["bert", "logistic_regression", "naive_bayes", "xgboost", "ensemble"],
        help="Choose the ML model for prediction"
    )
    
    # API health check
    st.subheader("🏥 API Status")
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            st.success("✅ API is healthy")
            st.json(data)
        else:
            st.error("❌ API is unhealthy")
    except Exception as e:
        st.error(f"❌ Cannot connect to API: {str(e)}")
        st.info("Make sure the API is running on http://localhost:8000")

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(["📝 Single Prediction", "📊 Batch Analysis", "📈 Analytics", "ℹ️ About"])

# Tab 1: Single Prediction
with tab1:
    st.header("Single Text Analysis")
    
    text_input = st.text_area(
        "Enter text to analyze:",
        height=150,
        placeholder="Type or paste your text here..."
    )
    
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        predict_button = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)
    with col2:
        clear_button = st.button("🗑️ Clear", use_container_width=True)
    
    if clear_button:
        st.rerun()
    
    if predict_button and text_input:
        with st.spinner("Analyzing..."):
            try:
                response = requests.post(
                    f"{API_URL}/api/v1/predict",
                    json={"text": text_input, "model": model},
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display results
                    st.success("Analysis Complete!")
                    
                    # Sentiment result
                    sentiment = result['sentiment']
                    confidence = result['confidence']
                    
                    # Color coding
                    color_map = {
                        'positive': '🟢',
                        'neutral': '🟡',
                        'negative': '🔴'
                    }
                    
                    st.markdown(f"### {color_map[sentiment]} Sentiment: **{sentiment.upper()}**")
                    st.markdown(f"**Confidence:** {confidence:.2%}")
                    
                    # Probability distribution
                    probs = result['probabilities']
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("😊 Positive", f"{probs['positive']:.2%}")
                    with col2:
                        st.metric("😐 Neutral", f"{probs['neutral']:.2%}")
                    with col3:
                        st.metric("😞 Negative", f"{probs['negative']:.2%}")
                    
                    # Probability chart
                    fig = go.Figure(data=[
                        go.Bar(
                            x=['Positive', 'Neutral', 'Negative'],
                            y=[probs['positive'], probs['neutral'], probs['negative']],
                            marker_color=['#2ecc71', '#f39c12', '#e74c3c']
                        )
                    ])
                    fig.update_layout(
                        title="Probability Distribution",
                        yaxis_title="Probability",
                        yaxis_tickformat='.0%',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Model info
                    with st.expander("📊 Detailed Results"):
                        st.json(result)
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    elif predict_button:
        st.warning("⚠️ Please enter some text to analyze")

# Tab 2: Batch Analysis
with tab2:
    st.header("Batch Text Analysis")
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Text Area", "Upload CSV"],
        horizontal=True
    )
    
    texts = []
    
    if input_method == "Text Area":
        batch_input = st.text_area(
            "Enter multiple texts (one per line):",
            height=200,
            placeholder="Enter each text on a new line..."
        )
        if batch_input:
            texts = [line.strip() for line in batch_input.split('\n') if line.strip()]
    
    else:  # Upload CSV
        uploaded_file = st.file_uploader(
            "Upload CSV file (must have a 'text' column)",
            type=['csv']
        )
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            if 'text' in df.columns:
                texts = df['text'].dropna().tolist()
                st.success(f"✅ Loaded {len(texts)} texts from CSV")
                with st.expander("Preview data"):
                    st.dataframe(df.head())
            else:
                st.error("❌ CSV must contain a 'text' column")
    
    if texts:
        st.info(f"📝 {len(texts)} texts ready for analysis")
        
        if st.button("🚀 Analyze Batch", type="primary"):
            with st.spinner(f"Analyzing {len(texts)} texts..."):
                try:
                    response = requests.post(
                        f"{API_URL}/api/v1/predict/batch",
                        json={"texts": texts, "model": model},
                        timeout=120
                    )
                    
                    if response.status_code == 200:
                        batch_result = response.json()
                        results = batch_result['results']
                        
                        st.success(f"✅ Analyzed {batch_result['total']} texts in {batch_result['processing_time_ms']:.0f}ms")
                        
                        # Create results dataframe
                        df_results = pd.DataFrame([
                            {
                                'Text': r['text'][:50] + '...' if len(r['text']) > 50 else r['text'],
                                'Sentiment': r['sentiment'],
                                'Confidence': r['confidence'],
                                'Positive': r['probabilities']['positive'],
                                'Neutral': r['probabilities']['neutral'],
                                'Negative': r['probabilities']['negative']
                            }
                            for r in results
                        ])
                        
                        # Summary statistics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Texts", len(results))
                        with col2:
                            positive_count = sum(1 for r in results if r['sentiment'] == 'positive')
                            st.metric("😊 Positive", positive_count)
                        with col3:
                            neutral_count = sum(1 for r in results if r['sentiment'] == 'neutral')
                            st.metric("😐 Neutral", neutral_count)
                        with col4:
                            negative_count = sum(1 for r in results if r['sentiment'] == 'negative')
                            st.metric("😞 Negative", negative_count)
                        
                        # Sentiment distribution pie chart
                        sentiment_counts = df_results['Sentiment'].value_counts()
                        fig = px.pie(
                            values=sentiment_counts.values,
                            names=sentiment_counts.index,
                            title="Sentiment Distribution",
                            color=sentiment_counts.index,
                            color_discrete_map={
                                'positive': '#2ecc71',
                                'neutral': '#f39c12',
                                'negative': '#e74c3c'
                            }
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Results table
                        st.subheader("📋 Detailed Results")
                        st.dataframe(
                            df_results.style.background_gradient(subset=['Confidence'], cmap='RdYlGn'),
                            use_container_width=True
                        )
                        
                        # Download results
                        csv = df_results.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name="sentiment_analysis_results.csv",
                            mime="text/csv"
                        )
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# Tab 3: Analytics
with tab3:
    st.header("Analytics & Visualization")
    
    st.info("💡 Analyze batch results to see visualizations here")
    
    # Sample data for demonstration
    if st.checkbox("Show sample analytics"):
        sample_data = {
            'sentiment': ['positive'] * 45 + ['neutral'] * 30 + ['negative'] * 25,
            'confidence': [0.85, 0.92, 0.78, 0.95, 0.88] * 20
        }
        df_sample = pd.DataFrame(sample_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment distribution
            fig1 = px.histogram(
                df_sample,
                x='sentiment',
                title="Sentiment Distribution",
                color='sentiment',
                color_discrete_map={
                    'positive': '#2ecc71',
                    'neutral': '#f39c12',
                    'negative': '#e74c3c'
                }
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Confidence distribution
            fig2 = px.box(
                df_sample,
                x='sentiment',
                y='confidence',
                title="Confidence by Sentiment",
                color='sentiment',
                color_discrete_map={
                    'positive': '#2ecc71',
                    'neutral': '#f39c12',
                    'negative': '#e74c3c'
                }
            )
            st.plotly_chart(fig2, use_container_width=True)

# Tab 4: About
with tab4:
    st.header("About This Dashboard")
    
    st.markdown("""
    ### 🎭 Sentiment Analysis System
    
    This dashboard provides an interactive interface for analyzing text sentiment using multiple machine learning models.
    
    #### 🤖 Available Models:
    - **BERT**: State-of-the-art transformer model (best accuracy)
    - **Logistic Regression**: Fast and reliable traditional ML
    - **Naive Bayes**: Lightweight probabilistic classifier
    
    #### 📊 Features:
    - Single text analysis with detailed probabilities
    - Batch processing for multiple texts
    - CSV file upload support
    - Interactive visualizations
    - Downloadable results
    
    #### 🔧 Technology Stack:
    - **Frontend**: Streamlit
    - **Backend API**: FastAPI
    - **ML Models**: scikit-learn, PyTorch, Transformers
    - **Visualization**: Plotly, Matplotlib
    
    #### 📖 Usage:
    1. Ensure the API is running on `http://localhost:8000`
    2. Select your preferred model from the sidebar
    3. Enter text or upload a CSV file
    4. Click "Analyze" to get predictions
    5. Download results if needed
    
    ---
    
    **Version:** 1.0.0  
    **Last Updated:** February 2026
    """)
    
    # System info
    with st.expander("🔍 System Information"):
        st.code(f"""
API Endpoint: {API_URL}
Selected Model: {model}
Streamlit Version: {st.__version__}
        """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>Built with ❤️ using Streamlit | Sentiment Analysis Dashboard v1.0</div>",
    unsafe_allow_html=True
)