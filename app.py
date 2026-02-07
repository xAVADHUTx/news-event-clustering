import streamlit as st
import pandas as pd
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from datetime import datetime, timedelta
import re
from collections import Counter
import plotly.express as px
import io
import os
import tempfile
import base64
import zipfile
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="AI News Clustering Dashboard",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    
    .cluster-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #667eea;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .article-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid #e0e0e0;
        transition: all 0.3s ease;
    }
    
    .article-card:hover {
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        transform: translateY(-2px);
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
    }
    
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .upload-section {
        background: linear-gradient(135deg, #f0f4ff 0%, #e6f7ff 100%);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        border: 2px dashed #667eea;
    }
    
    .file-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid #e0e0e0;
    }
    
    .highlight-box {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'articles' not in st.session_state:
    st.session_state.articles = []
if 'clustered' not in st.session_state:
    st.session_state.clustered = False
if 'clusters' not in st.session_state:
    st.session_state.clusters = {}
if 'selected_article' not in st.session_state:
    st.session_state.selected_article = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'dashboard'
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'file_data' not in st.session_state:
    st.session_state.file_data = {}
if 'active_dataset' not in st.session_state:
    st.session_state.active_dataset = None

# Sample news data
def fetch_sample_news():
    """Fetch sample news articles"""
    sample_articles = [
        {
            'title': 'AI Breakthrough: New Model Achieves Human-Level Reasoning',
            'content': 'Researchers at DeepMind have developed a new AI model that demonstrates human-level reasoning capabilities in complex problem-solving tasks. The model, called AlphaReason, can solve logical puzzles that were previously only solvable by humans.',
            'source': 'Tech News',
            'date': '2024-01-15',
            'url': '#',
            'category': 'AI Research',
            'sentiment': 'positive',
            'read_time': '5 min'
        },
        {
            'title': 'Climate Change: AI Predicts Rising Sea Levels',
            'content': 'Using advanced machine learning algorithms, scientists have created more accurate models for predicting sea-level rise. The AI system analyzes satellite data and ocean temperature readings to provide 10-year forecasts.',
            'source': 'Science Daily',
            'date': '2024-01-14',
            'url': '#',
            'category': 'Climate Tech',
            'sentiment': 'neutral',
            'read_time': '4 min'
        },
        {
            'title': 'Quantum Computing Meets AI: New Frontiers',
            'content': 'The intersection of quantum computing and artificial intelligence promises to revolutionize computational capabilities. Quantum neural networks could solve optimization problems thousands of times faster.',
            'source': 'Quantum News',
            'date': '2024-01-13',
            'url': '#',
            'category': 'Quantum AI',
            'sentiment': 'positive',
            'read_time': '6 min'
        },
        {
            'title': 'Healthcare AI: Diagnosing Diseases with 99% Accuracy',
            'content': 'A new AI system can diagnose rare diseases from medical images with unprecedented accuracy. The system was trained on over 1 million medical images and shows promise for early detection.',
            'source': 'MedTech',
            'date': '2024-01-12',
            'url': '#',
            'category': 'Health Tech',
            'sentiment': 'positive',
            'read_time': '3 min'
        },
        {
            'title': 'Autonomous Vehicles: AI Navigation Breakthrough',
            'content': 'Tesla announces new AI-powered navigation system that improves autonomous driving safety by 40%. The system uses reinforcement learning to adapt to complex traffic conditions.',
            'source': 'Auto Tech',
            'date': '2024-01-11',
            'url': '#',
            'category': 'Autonomous Vehicles',
            'sentiment': 'positive',
            'read_time': '5 min'
        },
        {
            'title': 'LLM Security: New Vulnerabilities Discovered',
            'content': 'Researchers have identified critical security vulnerabilities in popular large language models. The findings highlight the need for improved security protocols in AI systems.',
            'source': 'Security Weekly',
            'date': '2024-01-10',
            'url': '#',
            'category': 'AI Security',
            'sentiment': 'negative',
            'read_time': '4 min'
        },
        {
            'title': 'Robotics: AI-Powered Humanoid Robots',
            'content': 'Boston Dynamics unveils new generation of humanoid robots with advanced AI capabilities. The robots can now perform complex tasks in unstructured environments.',
            'source': 'Robotics Today',
            'date': '2024-01-09',
            'url': '#',
            'category': 'Robotics',
            'sentiment': 'positive',
            'read_time': '5 min'
        },
        {
            'title': 'AI Ethics: New Framework Proposed',
            'content': 'International consortium releases new ethical framework for AI development and deployment. The framework emphasizes transparency, accountability, and fairness.',
            'source': 'Ethics in Tech',
            'date': '2024-01-08',
            'url': '#',
            'category': 'AI Ethics',
            'sentiment': 'neutral',
            'read_time': '6 min'
        },
        {
            'title': 'Financial AI: Predicting Market Trends',
            'content': 'AI algorithms are now outperforming human traders in predicting stock market movements. Hedge funds are increasingly adopting AI-driven trading strategies.',
            'source': 'Finance Tech',
            'date': '2024-01-07',
            'url': '#',
            'category': 'FinTech',
            'sentiment': 'positive',
            'read_time': '4 min'
        },
        {
            'title': 'Edge AI: Running Models on Mobile Devices',
            'content': 'New compression techniques allow complex AI models to run efficiently on mobile devices. This enables real-time AI applications without cloud connectivity.',
            'source': 'Mobile Tech',
            'date': '2024-01-06',
            'url': '#',
            'category': 'Edge Computing',
            'sentiment': 'positive',
            'read_time': '3 min'
        }
    ]
    return sample_articles

def process_uploaded_file(file, file_name):
    """Process uploaded file and extract news articles"""
    try:
        # Reset file_data
        st.session_state.file_data[file_name] = {
            'articles': [],
            'metadata': {
                'file_name': file_name,
                'upload_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'file_size': f"{len(file.getvalue()) / 1024:.1f} KB"
            }
        }
        
        # Try to read as CSV
        try:
            if file_name.endswith('.csv'):
                df = pd.read_csv(file)
                
                # Map columns to expected format
                for _, row in df.iterrows():
                    article = {
                        'title': str(row.get('title', row.get('headline', 'Untitled'))),
                        'content': str(row.get('content', row.get('text', row.get('description', '')))),
                        'source': str(row.get('source', row.get('publisher', 'Unknown'))),
                        'date': str(row.get('date', row.get('published_date', datetime.now().strftime("%Y-%m-%d")))),
                        'url': str(row.get('url', row.get('link', '#'))),
                        'category': str(row.get('category', row.get('topic', 'General'))),
                        'sentiment': str(row.get('sentiment', 'neutral')),
                        'read_time': str(row.get('read_time', '3 min'))
                    }
                    st.session_state.file_data[file_name]['articles'].append(article)
                
                st.session_state.file_data[file_name]['metadata']['row_count'] = len(df)
                st.session_state.file_data[file_name]['metadata']['columns'] = list(df.columns)
                
            elif file_name.endswith('.json'):
                data = json.load(file)
                
                # Handle different JSON structures
                if isinstance(data, list):
                    for item in data:
                        article = {
                            'title': str(item.get('title', 'Untitled')),
                            'content': str(item.get('content', item.get('text', ''))),
                            'source': str(item.get('source', 'Unknown')),
                            'date': str(item.get('date', datetime.now().strftime("%Y-%m-%d"))),
                            'url': str(item.get('url', '#')),
                            'category': str(item.get('category', 'General')),
                            'sentiment': str(item.get('sentiment', 'neutral')),
                            'read_time': str(item.get('read_time', '3 min'))
                        }
                        st.session_state.file_data[file_name]['articles'].append(article)
                
                st.session_state.file_data[file_name]['metadata']['row_count'] = len(st.session_state.file_data[file_name]['articles'])
                
            elif file_name.endswith('.txt'):
                text = file.read().decode('utf-8')
                # Simple text parsing - split by lines or paragraphs
                lines = text.split('\n')
                for line in lines:
                    if len(line.strip()) > 20:  # Only consider substantial lines
                        article = {
                            'title': line[:100] + '...' if len(line) > 100 else line,
                            'content': line,
                            'source': 'Text File',
                            'date': datetime.now().strftime("%Y-%m-%d"),
                            'url': '#',
                            'category': 'Text Import',
                            'sentiment': 'neutral',
                            'read_time': '2 min'
                        }
                        st.session_state.file_data[file_name]['articles'].append(article)
                
                st.session_state.file_data[file_name]['metadata']['row_count'] = len(st.session_state.file_data[file_name]['articles'])
        
        except Exception as e:
            st.error(f"Error processing {file_name}: {str(e)}")
            return False
        
        return True
    
    except Exception as e:
        st.error(f"Failed to process file: {str(e)}")
        return False

def perform_clustering(articles, n_clusters=4):
    """Perform clustering on news articles"""
    try:
        # Prepare text data
        texts = [f"{art['title']} {art['content']} {art['category']}" for art in articles]
        
        # Vectorize text
        vectorizer = TfidfVectorizer(
            stop_words='english', 
            max_features=1000,
            min_df=2,
            max_df=0.8
        )
        X = vectorizer.fit_transform(texts)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        clusters = kmeans.fit_predict(X)
        
        # Get cluster labels (top words)
        feature_names = vectorizer.get_feature_names_out()
        cluster_labels = {}
        for i in range(n_clusters):
            # Get top 5 words for each cluster
            centroid = kmeans.cluster_centers_[i]
            top_indices = centroid.argsort()[-5:][::-1]
            top_words = [feature_names[idx] for idx in top_indices]
            cluster_labels[i] = f"Cluster {i}: " + ", ".join(top_words)
        
        # Calculate silhouette score
        from sklearn.metrics import silhouette_score
        if len(set(clusters)) > 1:
            silhouette_avg = silhouette_score(X, clusters)
        else:
            silhouette_avg = 0
        
        return clusters, cluster_labels, kmeans, X, vectorizer, silhouette_avg
    
    except Exception as e:
        st.error(f"Clustering error: {str(e)}")
        return None, None, None, None, None, 0

def visualize_clusters(X, clusters, articles):
    """Create 3D visualization of clusters"""
    try:
        # Check if we have enough data
        if X is None or len(articles) < 3:
            return None
        
        # Check if we have clusters
        if len(set(clusters)) < 2:
            st.info("Need at least 2 clusters for visualization.")
            return None
        
        # Reduce to 3 dimensions
        pca = PCA(n_components=min(3, X.shape[1], len(articles)-1), random_state=42)
        
        # Handle sparse or dense matrix
        if hasattr(X, 'toarray'):
            X_dense = X.toarray()
        else:
            X_dense = X
        
        # Ensure we have enough features for PCA
        if X_dense.shape[1] < 3:
            # Pad with zeros if needed
            X_dense = np.pad(X_dense, ((0, 0), (0, 3 - X_dense.shape[1])), mode='constant')
        
        X_pca = pca.fit_transform(X_dense)
        
        # Create DataFrame for plotting
        plot_df = pd.DataFrame({
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1],
            'PC3': X_pca[:, 2] if X_pca.shape[1] > 2 else np.zeros(len(X_pca)),
            'Cluster': clusters,
            'Title': [art['title'][:30] + '...' if len(art['title']) > 30 else art['title'] for art in articles],
            'Category': [art.get('category', 'Unknown') for art in articles],
            'Source': [art.get('source', 'Unknown') for art in articles],
            'Sentiment': [art.get('sentiment', 'neutral') for art in articles]
        })
        
        # Create 3D scatter plot
        fig = px.scatter_3d(
            plot_df, x='PC1', y='PC2', z='PC3',
            color='Cluster',
            hover_data=['Title', 'Category', 'Source', 'Sentiment'],
            title='News Clusters Visualization (PCA Reduced to 3D)',
            color_continuous_scale=px.colors.sequential.Viridis,
            symbol='Sentiment'
        )
        
        fig.update_traces(marker=dict(size=8, opacity=0.8))
        fig.update_layout(
            scene=dict(
                xaxis_title='Principal Component 1',
                yaxis_title='Principal Component 2',
                zaxis_title='Principal Component 3'
            ),
            height=600
        )
        
        return fig
    
    except Exception as e:
        st.error(f"Visualization error: {str(e)}")
        return None

def generate_wordcloud(articles, cluster_id=None):
    """Generate word cloud for articles"""
    try:
        if cluster_id is not None:
            # Get articles for specific cluster
            cluster_articles = []
            for cl_id, cl_data in st.session_state.clusters.items():
                if cl_id == cluster_id:
                    cluster_articles = cl_data['articles']
                    break
            text = ' '.join([art['title'] + ' ' + art['content'] for art in cluster_articles])
        else:
            # All articles
            text = ' '.join([art['title'] + ' ' + art['content'] for art in st.session_state.articles])
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=100,
            contour_width=3,
            contour_color='steelblue'
        ).generate(text)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        
        return fig
    
    except Exception as e:
        st.error(f"Word cloud error: {str(e)}")
        return None

# Main app layout
def main():
    # Header
    st.markdown('<h1 class="main-header">📰 AI News Clustering Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/190/190411.png", width=80)
        st.title("Navigation")
        
        # Navigation buttons with unique keys
        nav_cols = st.columns(2)
        with nav_cols[0]:
            if st.button("🏠 Dashboard", key="nav_dashboard", use_container_width=True):
                st.session_state.current_page = 'dashboard'
                st.rerun()
        
        with nav_cols[1]:
            if st.button("📊 Clusters", key="nav_clusters", use_container_width=True):
                st.session_state.current_page = 'clusters'
                st.rerun()
        
        nav_cols2 = st.columns(2)
        with nav_cols2[0]:
            if st.button("📈 Analytics", key="nav_analytics", use_container_width=True):
                st.session_state.current_page = 'analytics'
                st.rerun()
        
        with nav_cols2[1]:
            if st.button("⚙️ Settings", key="nav_settings", use_container_width=True):
                st.session_state.current_page = 'settings'
                st.rerun()
        
        st.divider()
        
        st.title("📁 File Browser")
        
        # File upload section
        uploaded_files = st.file_uploader(
            "Browse and Upload News Files",
            type=['csv', 'json', 'txt'],
            accept_multiple_files=True,
            key="file_uploader_main",
            help="Upload CSV, JSON, or TXT files containing news articles"
        )
        
        # Process uploaded files
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in st.session_state.uploaded_files:
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        if process_uploaded_file(uploaded_file, uploaded_file.name):
                            st.session_state.uploaded_files.append(uploaded_file.name)
                            st.success(f"✅ {uploaded_file.name} loaded successfully!")
        
        # Display uploaded files
        if st.session_state.uploaded_files:
            st.subheader("Uploaded Files")
            for file_name in st.session_state.uploaded_files:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"📄 **{file_name}**")
                    if file_name in st.session_state.file_data:
                        meta = st.session_state.file_data[file_name]['metadata']
                        st.caption(f"{meta.get('row_count', 0)} articles | {meta.get('file_size', 'N/A')}")
                
                with col2:
                    if st.button("📊 Use", key=f"use_{file_name}"):
                        st.session_state.articles = st.session_state.file_data[file_name]['articles']
                        st.session_state.active_dataset = file_name
                        st.session_state.clustered = False
                        st.success(f"Using dataset: {file_name}")
                        st.rerun()
        
        st.divider()
        
        st.title("Clustering Controls")
        
        # Dataset selector
        dataset_options = ["Sample Data"] + st.session_state.uploaded_files
        selected_dataset = st.selectbox(
            "Select Dataset",
            options=dataset_options,
            index=0,
            key="dataset_select"
        )
        
        if selected_dataset == "Sample Data":
            if st.button("📥 Load Sample Data", key="load_sample_btn"):
                st.session_state.articles = fetch_sample_news()
                st.session_state.active_dataset = "Sample Data"
                st.session_state.clustered = False
                st.success("Sample data loaded!")
                st.rerun()
        elif selected_dataset in st.session_state.file_data:
            if st.button(f"📥 Load {selected_dataset}", key=f"load_{selected_dataset}"):
                st.session_state.articles = st.session_state.file_data[selected_dataset]['articles']
                st.session_state.active_dataset = selected_dataset
                st.session_state.clustered = False
                st.success(f"Loaded dataset: {selected_dataset}")
                st.rerun()
        
        n_clusters = st.slider("Number of Clusters", 2, 10, 4, key="n_clusters_slider")
        
        if st.button("🔍 Perform Clustering", key="cluster_btn", type="primary"):
            if not st.session_state.articles:
                st.warning("Please load data first!")
            else:
                with st.spinner("Clustering in progress..."):
                    clusters, cluster_labels, model, X, vectorizer, silhouette = perform_clustering(
                        st.session_state.articles, n_clusters
                    )
                    
                    if clusters is not None:
                        # Organize articles by cluster
                        st.session_state.clusters = {}
                        for i, article in enumerate(st.session_state.articles):
                            cluster_id = clusters[i]
                            if cluster_id not in st.session_state.clusters:
                                st.session_state.clusters[cluster_id] = {
                                    'label': cluster_labels[cluster_id],
                                    'articles': []
                                }
                            st.session_state.clusters[cluster_id]['articles'].append(article)
                        
                        st.session_state.clustered = True
                        st.session_state.cluster_model = model
                        st.session_state.cluster_X = X
                        st.session_state.vectorizer = vectorizer
                        st.session_state.silhouette_score = silhouette
                        st.success(f"✅ Clustered {len(st.session_state.articles)} articles into {n_clusters} clusters!")
                        st.rerun()
        
        st.divider()
        
        if st.button("🧹 Clear All Data", key="clear_all_btn"):
            st.session_state.articles = []
            st.session_state.clustered = False
            st.session_state.clusters = {}
            st.session_state.selected_article = None
            st.session_state.uploaded_files = []
            st.session_state.file_data = {}
            st.session_state.active_dataset = None
            st.rerun()
        
        st.divider()
        
        # Stats
        if st.session_state.articles:
            st.metric("📊 Total Articles", len(st.session_state.articles))
            if st.session_state.clustered:
                st.metric("🔷 Clusters", len(st.session_state.clusters))
                if hasattr(st.session_state, 'silhouette_score'):
                    st.metric("⭐ Silhouette Score", f"{st.session_state.silhouette_score:.3f}")
    
    # Main content based on current page
    if st.session_state.current_page == 'dashboard':
        show_dashboard()
    elif st.session_state.current_page == 'clusters':
        show_clusters()
    elif st.session_state.current_page == 'analytics':
        show_analytics()
    elif st.session_state.current_page == 'settings':
        show_settings()

def show_dashboard():
    """Display dashboard page"""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("🌐 News Clustering Overview")
        
        # Data source info
        if st.session_state.active_dataset:
            st.markdown(f'<div class="highlight-box">', unsafe_allow_html=True)
            st.markdown(f"**Active Dataset:** {st.session_state.active_dataset}")
            if st.session_state.active_dataset in st.session_state.file_data:
                meta = st.session_state.file_data[st.session_state.active_dataset]['metadata']
                st.markdown(f"**File Info:** {meta.get('row_count', 0)} articles | {meta.get('file_size', 'N/A')} | Uploaded: {meta.get('upload_time', 'N/A')}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        if not st.session_state.articles:
            # Show upload section prominently
            st.markdown('<div class="upload-section">', unsafe_allow_html=True)
            st.markdown("### 📁 Get Started with News Clustering")
            st.markdown("""
            1. **Upload your news data** using the file browser in the sidebar
            2. **Load sample data** to see a demo
            3. **Configure clustering settings** and click "Perform Clustering"
            
            Supported formats:
            - 📊 CSV files (with title, content columns)
            - 📄 JSON files
            - 📝 Text files
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Quick upload option
            st.subheader("Quick Upload")
            quick_upload = st.file_uploader(
                "Drag and drop your news file here",
                type=['csv', 'json', 'txt'],
                key="quick_upload"
            )
            if quick_upload:
                with st.spinner(f"Processing {quick_upload.name}..."):
                    if process_uploaded_file(quick_upload, quick_upload.name):
                        st.session_state.uploaded_files.append(quick_upload.name)
                        st.session_state.articles = st.session_state.file_data[quick_upload.name]['articles']
                        st.session_state.active_dataset = quick_upload.name
                        st.success(f"✅ {quick_upload.name} loaded successfully!")
                        st.rerun()
        
        elif not st.session_state.clustered:
            st.info("📊 Data loaded! Click 'Perform Clustering' in the sidebar to analyze.")
            
            # Show data preview
            with st.expander("📋 Data Preview", expanded=True):
                preview_df = pd.DataFrame(st.session_state.articles)
                st.dataframe(preview_df[['title', 'source', 'category', 'date']].head(10), use_container_width=True)
            
            # Quick stats
            col1_stat, col2_stat, col3_stat = st.columns(3)
            with col1_stat:
                st.metric("Articles", len(st.session_state.articles))
            with col2_stat:
                sources = len(set([art['source'] for art in st.session_state.articles]))
                st.metric("Sources", sources)
            with col3_stat:
                categories = len(set([art['category'] for art in st.session_state.articles]))
                st.metric("Categories", categories)
        
        else:
            # Metrics row
            st.subheader("📈 Performance Metrics")
            metric_cols = st.columns(4)
            with metric_cols[0]:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Total Articles", len(st.session_state.articles))
                st.markdown('</div>', unsafe_allow_html=True)
            
            with metric_cols[1]:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Clusters", len(st.session_state.clusters))
                st.markdown('</div>', unsafe_allow_html=True)
            
            with metric_cols[2]:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                avg_articles = np.mean([len(c['articles']) for c in st.session_state.clusters.values()])
                st.metric("Avg/Cluster", f"{avg_articles:.1f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with metric_cols[3]:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                if hasattr(st.session_state, 'silhouette_score'):
                    st.metric("Silhouette Score", f"{st.session_state.silhouette_score:.3f}")
                else:
                    st.metric("Quality", "Calculating...")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Visualization
            st.subheader("🔍 Cluster Visualization")
            
            # Visualization options
            viz_tabs = st.tabs(["3D Scatter Plot", "2D Projection", "Word Cloud"])
            
            with viz_tabs[0]:
                fig = visualize_clusters(
                    st.session_state.cluster_X,
                    [cl_id for cl_id, cl_data in st.session_state.clusters.items() 
                    for _ in cl_data['articles']],
                    st.session_state.articles
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            with viz_tabs[1]:
                # 2D visualization
                try:
                    pca_2d = PCA(n_components=2, random_state=42)
                    X_dense = st.session_state.cluster_X.toarray() if hasattr(st.session_state.cluster_X, 'toarray') else st.session_state.cluster_X
                    X_pca_2d = pca_2d.fit_transform(X_dense)
                    
                    plot_df_2d = pd.DataFrame({
                        'PC1': X_pca_2d[:, 0],
                        'PC2': X_pca_2d[:, 1],
                        'Cluster': [cl_id for cl_id, cl_data in st.session_state.clusters.items() 
                                   for _ in cl_data['articles']],
                        'Title': [art['title'][:30] + '...' for art in st.session_state.articles]
                    })
                    
                    fig_2d = px.scatter(
                        plot_df_2d, x='PC1', y='PC2',
                        color='Cluster',
                        hover_data=['Title'],
                        title='2D Cluster Visualization',
                        color_continuous_scale=px.colors.sequential.Viridis
                    )
                    st.plotly_chart(fig_2d, use_container_width=True)
                except Exception as e:
                    st.error(f"2D visualization error: {str(e)}")
            
            with viz_tabs[2]:
                # Word cloud for all articles
                fig_wc = generate_wordcloud(st.session_state.articles)
                if fig_wc:
                    st.pyplot(fig_wc)
                else:
                    st.info("Word cloud generation requires more text data.")
    
    with col2:
        st.subheader("📌 Quick Actions")
        
        # File management
        with st.expander("📁 File Management", expanded=True):
            if st.button("📤 Upload New File", key="upload_new_btn"):
                st.session_state.current_page = 'dashboard'
                st.rerun()
            
            if st.session_state.uploaded_files:
                if st.button("🔄 Refresh All Files", key="refresh_files_btn"):
                    st.rerun()
            
            if st.button("💾 Export Clusters", key="export_btn_sidebar"):
                # Export functionality
                export_data = []
                for cluster_id, cluster_data in st.session_state.clusters.items():
                    for article in cluster_data['articles']:
                        export_data.append({
                            'cluster_id': cluster_id,
                            'cluster_label': cluster_data['label'],
                            'title': article['title'],
                            'content': article['content'][:500],
                            'source': article['source'],
                            'date': article['date'],
                            'category': article['category'],
                            'sentiment': article.get('sentiment', 'neutral')
                        })
                
                if export_data:
                    df_export = pd.DataFrame(export_data)
                    csv = df_export.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name="news_clusters_export.csv",
                        mime="text/csv",
                        key="download_sidebar"
                    )
        
        st.divider()
        
        # Recent activity
        st.subheader("🔄 Recent Activity")
        
        if st.session_state.selected_article:
            with st.expander("📖 Selected Article", expanded=True):
                article = st.session_state.selected_article
                st.markdown(f"**{article['title']}**")
                st.caption(f"📰 {article['source']} | 🏷️ {article['category']}")
                st.caption(f"📅 {article['date']} | ⏱️ {article.get('read_time', 'N/A')}")
                st.write(article['content'][:150] + "...")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Read Full", key="read_full_btn"):
                        st.session_state.current_page = 'clusters'
                        st.rerun()
                with col2:
                    if st.button("Close", key="close_article_sidebar"):
                        st.session_state.selected_article = None
                        st.rerun()
        
        st.divider()
        
        # Cluster distribution pie chart
        if st.session_state.clusters:
            st.subheader("📊 Cluster Distribution")
            cluster_sizes = [len(cl['articles']) for cl in st.session_state.clusters.values()]
            cluster_names = [f"Cluster {i}" for i in range(len(cluster_sizes))]
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=cluster_names, 
                values=cluster_sizes,
                hole=0.3,
                marker_colors=px.colors.sequential.Viridis
            )])
            fig_pie.update_layout(
                height=300, 
                margin=dict(t=0, b=0, l=0, r=0),
                showlegend=True
            )
            st.plotly_chart(fig_pie, use_container_width=True)

def show_clusters():
    """Display clusters page"""
    st.subheader("📊 News Clusters Analysis")
    
    if not st.session_state.clustered:
        st.warning("No clusters available. Please perform clustering first.")
        return
    
    # Cluster overview
    st.markdown(f"**Dataset:** {st.session_state.active_dataset} | **Total Clusters:** {len(st.session_state.clusters)}")
    
    # Create tabs for each cluster
    cluster_tabs = st.tabs([f"🔷 {cl['label']}" for cl_id, cl in sorted(st.session_state.clusters.items())])
    
    for tab_idx, (cluster_id, cluster_data) in enumerate(sorted(st.session_state.clusters.items())):
        with cluster_tabs[tab_idx]:
            col_header1, col_header2, col_header3 = st.columns([3, 1, 1])
            with col_header1:
                st.markdown(f'<div class="cluster-card">', unsafe_allow_html=True)
                st.markdown(f"### {cluster_data['label']}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_header2:
                st.metric("Articles", len(cluster_data['articles']))
            
            with col_header3:
                # Word cloud for this cluster
                if st.button("📊 Word Cloud", key=f"wc_btn_{cluster_id}"):
                    fig_wc_cluster = generate_wordcloud(st.session_state.articles, cluster_id)
                    if fig_wc_cluster:
                        st.pyplot(fig_wc_cluster)
            
            # Display articles in this cluster
            st.subheader(f"📰 Articles in Cluster {cluster_id}")
            
            for art_idx, article in enumerate(cluster_data['articles']):
                # Create a unique key for each article button
                unique_key = f"cluster_{cluster_id}_article_{art_idx}"
                
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f'<div class="article-card">', unsafe_allow_html=True)
                    
                    # Article header
                    header_cols = st.columns([3, 1])
                    with header_cols[0]:
                        st.markdown(f"**{article['title']}**")
                    with header_cols[1]:
                        sentiment = article.get('sentiment', 'neutral')
                        sentiment_color = {
                            'positive': '🟢',
                            'negative': '🔴',
                            'neutral': '⚪'
                        }.get(sentiment, '⚪')
                        st.markdown(f"{sentiment_color} {sentiment.capitalize()}")
                    
                    # Article metadata
                    meta_cols = st.columns(3)
                    with meta_cols[0]:
                        st.caption(f"📰 {article['source']}")
                    with meta_cols[1]:
                        st.caption(f"🏷️ {article['category']}")
                    with meta_cols[2]:
                        st.caption(f"📅 {article['date']}")
                    
                    # Article preview
                    st.markdown(f"{article['content'][:200]}...")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    # Use unique key for the button
                    if st.button("📖 Read", key=unique_key):
                        st.session_state.selected_article = article
                        st.rerun()
            
            st.divider()
            
            # Cluster insights
            with st.expander("📈 Cluster Insights & Statistics"):
                col_insight1, col_insight2 = st.columns(2)
                
                with col_insight1:
                    st.write("**Category Distribution:**")
                    categories = [art['category'] for art in cluster_data['articles']]
                    if categories:
                        cat_counts = Counter(categories)
                        for cat, count in cat_counts.most_common():
                            st.write(f"- `{cat}`: {count} articles ({count/len(categories)*100:.1f}%)")
                    
                    st.write("**Source Analysis:**")
                    sources = [art['source'] for art in cluster_data['articles']]
                    if sources:
                        source_counts = Counter(sources)
                        for source, count in source_counts.most_common(3):
                            st.write(f"- `{source}`: {count} articles")
                
                with col_insight2:
                    st.write("**Sentiment Analysis:**")
                    sentiments = [art.get('sentiment', 'neutral') for art in cluster_data['articles']]
                    if sentiments:
                        sentiment_counts = Counter(sentiments)
                        for sentiment, count in sentiment_counts.items():
                            percentage = count/len(sentiments)*100
                            sentiment_emoji = {
                                'positive': '🟢',
                                'negative': '🔴',
                                'neutral': '⚪'
                            }.get(sentiment, '⚪')
                            st.write(f"- {sentiment_emoji} {sentiment.capitalize()}: {count} ({percentage:.1f}%)")
                    
                    st.write("**Temporal Analysis:**")
                    dates = [art['date'] for art in cluster_data['articles']]
                    if dates:
                        try:
                            date_counts = Counter(dates)
                            st.write(f"- Unique dates: {len(date_counts)}")
                            most_recent = max(dates)
                            st.write(f"- Most recent: {most_recent}")
                        except:
                            st.write("- Date analysis not available")

def show_analytics():
    """Display analytics page"""
    st.subheader("📈 Advanced Analytics")
    
    if not st.session_state.clustered:
        st.warning("No data available. Please perform clustering first.")
        return
    
    # Create tabs for different analytics
    analytics_tabs = st.tabs(["Temporal Analysis", "Source Analysis", "Sentiment Analysis", "Advanced Metrics"])
    
    with analytics_tabs[0]:
        st.subheader("📅 Temporal Analysis")
        
        # Convert dates to datetime
        dates = []
        for article in st.session_state.articles:
            try:
                date_str = article['date']
                # Try different date formats
                for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y.%m.%d']:
                    try:
                        dates.append(datetime.strptime(date_str, fmt))
                        break
                    except:
                        continue
                else:
                    dates.append(datetime.now())  # Default if parsing fails
            except:
                dates.append(datetime.now())
        
        if dates:
            date_df = pd.DataFrame({
                'date': dates,
                'cluster': [cl_id for cl_id, cl_data in st.session_state.clusters.items() 
                           for art in cl_data['articles']]
            })
            
            # Group by date and cluster
            date_cluster_counts = date_df.groupby(['date', 'cluster']).size().reset_index(name='count')
            
            fig_time = px.line(
                date_cluster_counts,
                x='date',
                y='count',
                color='cluster',
                title='Article Distribution Over Time by Cluster',
                labels={'date': 'Date', 'count': 'Number of Articles', 'cluster': 'Cluster'}
            )
            st.plotly_chart(fig_time, use_container_width=True)
    
    with analytics_tabs[1]:
        st.subheader("📰 Source Analysis")
        
        sources = [art['source'] for art in st.session_state.articles]
        source_counts = Counter(sources)
        
        # Bar chart
        source_df = pd.DataFrame({
            'Source': list(source_counts.keys()),
            'Count': list(source_counts.values())
        }).sort_values('Count', ascending=False).head(15)
        
        fig_source = px.bar(
            source_df,
            x='Source',
            y='Count',
            title='Top 15 News Sources',
            color='Count',
            color_continuous_scale=px.colors.sequential.Viridis
        )
        st.plotly_chart(fig_source, use_container_width=True)
        
        # Source by cluster
        st.subheader("Source Distribution by Cluster")
        source_cluster_data = []
        for cluster_id, cluster_data in st.session_state.clusters.items():
            cluster_sources = [art['source'] for art in cluster_data['articles']]
            source_counts_cluster = Counter(cluster_sources)
            for source, count in source_counts_cluster.most_common(5):
                source_cluster_data.append({
                    'Cluster': f'Cluster {cluster_id}',
                    'Source': source,
                    'Count': count
                })
        
        if source_cluster_data:
            source_cluster_df = pd.DataFrame(source_cluster_data)
            fig_source_cluster = px.bar(
                source_cluster_df,
                x='Cluster',
                y='Count',
                color='Source',
                title='Top Sources per Cluster',
                barmode='stack'
            )
            st.plotly_chart(fig_source_cluster, use_container_width=True)
    
    with analytics_tabs[2]:
        st.subheader("😊 Sentiment Analysis")
        
        sentiments = [art.get('sentiment', 'neutral') for art in st.session_state.articles]
        sentiment_counts = Counter(sentiments)
        
        # Pie chart
        fig_sentiment = px.pie(
            values=list(sentiment_counts.values()),
            names=list(sentiment_counts.keys()),
            title='Overall Sentiment Distribution',
            color_discrete_sequence=px.colors.sequential.Viridis
        )
        st.plotly_chart(fig_sentiment, use_container_width=True)
        
        # Sentiment by cluster
        st.subheader("Sentiment by Cluster")
        sentiment_cluster_data = []
        for cluster_id, cluster_data in st.session_state.clusters.items():
            cluster_sentiments = [art.get('sentiment', 'neutral') for art in cluster_data['articles']]
            sentiment_counts_cluster = Counter(cluster_sentiments)
            for sentiment, count in sentiment_counts_cluster.items():
                sentiment_cluster_data.append({
                    'Cluster': f'Cluster {cluster_id}',
                    'Sentiment': sentiment,
                    'Count': count
                })
        
        if sentiment_cluster_data:
            sentiment_cluster_df = pd.DataFrame(sentiment_cluster_data)
            fig_sentiment_cluster = px.bar(
                sentiment_cluster_df,
                x='Cluster',
                y='Count',
                color='Sentiment',
                title='Sentiment Distribution by Cluster',
                barmode='stack',
                color_discrete_map={
                    'positive': '#00CC96',
                    'negative': '#EF553B',
                    'neutral': '#636EFA'
                }
            )
            st.plotly_chart(fig_sentiment_cluster, use_container_width=True)
    
    with analytics_tabs[3]:
        st.subheader("📊 Advanced Clustering Metrics")
        
        if hasattr(st.session_state, 'cluster_model') and hasattr(st.session_state, 'cluster_X'):
            from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
            
            try:
                labels = st.session_state.cluster_model.labels_
                
                col_metric1, col_metric2, col_metric3 = st.columns(3)
                
                with col_metric1:
                    silhouette = st.session_state.silhouette_score
                    st.metric("Silhouette Score", f"{silhouette:.3f}")
                    st.caption("Higher is better (range: -1 to 1)")
                
                with col_metric2:
                    try:
                        calinski = calinski_harabasz_score(st.session_state.cluster_X.toarray(), labels)
                        st.metric("Calinski-Harabasz", f"{calinski:.0f}")
                        st.caption("Higher is better")
                    except:
                        st.metric("Calinski-Harabasz", "N/A")
                
                with col_metric3:
                    try:
                        davies = davies_bouldin_score(st.session_state.cluster_X.toarray(), labels)
                        st.metric("Davies-Bouldin", f"{davies:.3f}")
                        st.caption("Lower is better")
                    except:
                        st.metric("Davies-Bouldin", "N/A")
                
                # Cluster separation visualization
                st.subheader("Cluster Separation")
                
                # Calculate inter-cluster distances
                from sklearn.metrics.pairwise import euclidean_distances
                
                centroids = st.session_state.cluster_model.cluster_centers_
                centroid_distances = euclidean_distances(centroids)
                
                # Create heatmap of centroid distances
                fig_heatmap = px.imshow(
                    centroid_distances,
                    labels=dict(x="Cluster", y="Cluster", color="Distance"),
                    x=[f"Cluster {i}" for i in range(len(centroids))],
                    y=[f"Cluster {i}" for i in range(len(centroids))],
                    title="Inter-Cluster Distance Matrix",
                    color_continuous_scale=px.colors.sequential.Viridis
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error calculating metrics: {str(e)}")

def show_settings():
    """Display settings page"""
    st.subheader("⚙️ Application Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="file-card">', unsafe_allow_html=True)
        st.write("### 📁 File Management Settings")
        
        # Auto-process uploaded files
        auto_process = st.checkbox(
            "Auto-process uploaded files",
            value=True,
            key="auto_process_check"
        )
        
        # Max file size (MB)
        max_file_size = st.slider(
            "Maximum file size (MB)",
            min_value=1,
            max_value=100,
            value=10,
            key="max_file_size"
        )
        
        # Default file format
        default_format = st.selectbox(
            "Preferred file format",
            ["CSV", "JSON", "TXT"],
            key="default_format"
        )
        
        # Clear uploaded files after session
        clear_on_exit = st.checkbox(
            "Clear uploaded files on exit",
            value=False,
            key="clear_on_exit"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="file-card">', unsafe_allow_html=True)
        st.write("### 🔧 Clustering Settings")
        
        # Default number of clusters
        default_clusters = st.slider(
            "Default number of clusters",
            min_value=2,
            max_value=10,
            value=4,
            key="default_clusters"
        )
        
        # Vectorizer settings
        max_features = st.slider(
            "Max features for vectorization",
            min_value=100,
            max_value=5000,
            value=1000,
            step=100,
            key="max_features"
        )
        
        # Clustering algorithm
        algorithm = st.selectbox(
            "Clustering algorithm",
            ["K-Means", "Agglomerative", "DBSCAN", "OPTICS"],
            key="algorithm"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="file-card">', unsafe_allow_html=True)
        st.write("### 🎨 Display Settings")
        
        # Theme selection
        theme = st.selectbox(
            "Color theme",
            ["Light", "Dark", "Auto"],
            key="theme"
        )
        
        # Results per page
        results_per_page = st.slider(
            "Articles per page",
            min_value=5,
            max_value=50,
            value=10,
            key="results_per_page"
        )
        
        # Auto-refresh interval
        refresh_interval = st.selectbox(
            "Auto-refresh interval",
            ["Off", "5 minutes", "15 minutes", "30 minutes", "1 hour"],
            key="refresh_interval"
        )
        
        # Show preview images
        show_images = st.checkbox(
            "Show preview images",
            value=True,
            key="show_images"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="file-card">', unsafe_allow_html=True)
        st.write("### 💾 Export Settings")
        
        # Default export format
        export_format = st.selectbox(
            "Default export format",
            ["CSV", "JSON", "Excel", "PDF"],
            key="export_format"
        )
        
        # Include metadata in export
        include_metadata = st.checkbox(
            "Include metadata in exports",
            value=True,
            key="include_metadata"
        )
        
        # Auto-generate report
        auto_report = st.checkbox(
            "Auto-generate analysis report",
            value=False,
            key="auto_report"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Save and reset buttons
    col_save1, col_save2, col_save3 = st.columns(3)
    
    with col_save1:
        if st.button("💾 Save Settings", key="save_settings", type="primary"):
            st.success("Settings saved successfully!")
    
    with col_save2:
        if st.button("🔄 Reset to Defaults", key="reset_settings"):
            st.warning("This will reset all settings to default values.")
    
    with col_save3:
        if st.button("📋 Export Settings", key="export_settings"):
            settings_data = {
                "file_management": {
                    "auto_process": auto_process,
                    "max_file_size": max_file_size,
                    "default_format": default_format
                },
                "clustering": {
                    "default_clusters": default_clusters,
                    "max_features": max_features,
                    "algorithm": algorithm
                },
                "display": {
                    "theme": theme,
                    "results_per_page": results_per_page,
                    "refresh_interval": refresh_interval
                }
            }
            
            settings_json = json.dumps(settings_data, indent=2)
            st.download_button(
                label="Download Settings",
                data=settings_json,
                file_name="clustering_settings.json",
                mime="application/json",
                key="download_settings"
            )
    
    st.divider()
    
    # System information
    st.subheader("🖥️ System Information")
    
    sys_col1, sys_col2, sys_col3 = st.columns(3)
    
    with sys_col1:
        st.metric("Python Version", "3.9+")
        st.metric("Streamlit Version", "1.28.0+")
    
    with sys_col2:
        st.metric("Scikit-learn Version", "1.3.0+")
        st.metric("Pandas Version", "2.0.0+")
    
    with sys_col3:
        st.metric("Uploaded Files", len(st.session_state.uploaded_files))
        st.metric("Active Dataset", st.session_state.active_dataset or "None")
    
    # Memory usage (simulated)
    st.progress(65, text="System Resource Usage: 65%")

# Run the app
if __name__ == "__main__":
    main()