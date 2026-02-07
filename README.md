📰 AI News Clustering Dashboard

An intelligent web application for clustering and analyzing news articles using machine learning algorithms. Automatically categorize, visualize, and gain insights from your news data.

✨ Features
🤖 AI-Powered Clustering
K-Means Clustering: Automatically group similar news articles

TF-IDF Vectorization: Advanced text processing for accurate similarity detection

Automatic Cluster Labeling: Each cluster is labeled with top keywords

Silhouette Analysis: Measure clustering quality with advanced metrics

📊 Interactive Visualizations
3D Cluster Visualization: Explore clusters in three-dimensional space using PCA

Word Clouds: Visual representation of key topics

Sentiment Analysis: Track positive, negative, and neutral sentiment

Temporal Analysis: View article trends over time

Source Distribution: Analyze news sources

📁 Multi-Format Support
CSV Upload: Upload news data with automatic column detection

JSON Support: Import structured JSON news data

Text Files: Process plain text files

Sample Data: Get started quickly with built-in sample articles

🔍 Advanced Analytics
Cluster Quality Metrics: Silhouette score, Calinski-Harabasz, Davies-Bouldin

Source Analysis: Identify dominant news sources per cluster

Sentiment Tracking: Monitor emotional tone

Temporal Patterns: Discover trends over time

💾 Export Capabilities
CSV Export: Download clustered data for further analysis

Report Generation: Create comprehensive analysis reports

Configuration Export: Save and reuse clustering settings

🚀 Quick Start
Prerequisites
Python 3.9 or higher

pip package manager

Installation
bash
# Clone the repository
git clone https://github.com/yourusername/ai-news-clustering.git
cd ai-news-clustering

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
