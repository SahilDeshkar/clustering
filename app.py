import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import google.generativeai as genai
import json
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="AI-Enhanced Customer Segmentation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.example.com/help',
        'Report a bug': 'https://www.example.com/bug',
        'About': 'Customer Segmentation Platform v1.0'
    }
)

# Custom CSS for enhanced aesthetics
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4F46E5, #7C3AED);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        padding-top: 1.5rem;
        text-align: center;
    }
    
    .sub-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #4F46E5;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e0e0ff;
    }
    
    /* Card styling */
    .card {
        background-color: white;
        border-radius: 0.8rem;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
        border-left: 4px solid #4F46E5;
    }
    
    .card-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #4F46E5;
        margin-bottom: 1rem;
    }
    
    /* Info boxes */
    .info-box {
        background-color: #f0f7ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4F46E5;
        margin-bottom: 1rem;
    }
    
    .success-box {
        background-color: #f0fff4;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #10B981;
        margin-bottom: 1rem;
    }
    
    .warning-box {
        background-color: #fffbeb;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #F59E0B;
        margin-bottom: 1rem;
    }
    
    /* Metrics styling */
    .metric-container {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background-color: white;
        border-radius: 0.5rem;
        padding: 1rem;
        flex: 1;
        min-width: 150px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #4F46E5;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #6B7280;
        margin-top: 0.3rem;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #4F46E5;
        color: white;
        border-radius: 0.5rem;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    
    .stButton > button:hover {
        background-color: #4338CA;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8fafc;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f1f5f9;
        border-radius: 0.5rem 0.5rem 0 0;
        padding: 0.5rem 1rem;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: white !important;
        border-top: 3px solid #4F46E5 !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #e5e7eb;
        color: #6B7280;
        font-size: 0.8rem;
    }
    
    /* Tooltip */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Animation for loading */
    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
    
    .loading-animation {
        animation: pulse 1.5s infinite;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #c7d2fe;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #4F46E5;
    }
    
    /* Table styling */
    .dataframe {
        border-collapse: collapse;
        width: 100%;
        border-radius: 0.5rem;
        overflow: hidden;
    }
    
    .dataframe th {
        background-color: #f1f5f9;
        padding: 0.75rem 1rem;
        text-align: left;
        font-weight: 600;
        color: #4F46E5;
    }
    
    .dataframe td {
        padding: 0.75rem 1rem;
        border-top: 1px solid #f1f5f9;
    }
    
    .dataframe tr:hover {
        background-color: #f8fafc;
    }
    </style>
    """, unsafe_allow_html=True)

# Project logo and title
def render_header():
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown('<div class="main-header">AI-Enhanced Customer Segmentation</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <span style="background-color: #EEF2FF; padding: 0.4rem 0.8rem; border-radius: 2rem; font-weight: 500; color: #4F46E5; margin-right: 0.5rem;">
                <span style="font-size: 1.2rem;">🔍</span> Discover Customer Patterns
            </span>
            <span style="background-color: #EEF2FF; padding: 0.4rem 0.8rem; border-radius: 2rem; font-weight: 500; color: #4F46E5; margin-right: 0.5rem;">
                <span style="font-size: 1.2rem;">🎯</span> Target Marketing Strategies
            </span>
            <span style="background-color: #EEF2FF; padding: 0.4rem 0.8rem; border-radius: 2rem; font-weight: 500; color: #4F46E5;">
                <span style="font-size: 1.2rem;">💡</span> AI-Powered Insights
            </span>
        </div>
        """, unsafe_allow_html=True)

# Initialize Gemini API
def init_gemini():
    try:
        api_key = os.getenv("GEMINI_API_KEY")
    except:
        api_key = st.sidebar.text_input("Enter Gemini API Key:", type="password")
        if not api_key:
            st.sidebar.warning("Please enter a Gemini API key to enable AI features")
            return None
    
    genai.configure(api_key=api_key)
    return genai

# Function to generate insights using Gemini
def generate_ai_insights(df, cluster_df, features, centers):
    genai_instance = init_gemini()
    
    if not genai_instance:
        return "AI insights unavailable. Please provide a valid Gemini API key."
    
    # Prepare cluster statistics
    cluster_stats = {}
    for i in range(len(centers)):
        cluster_data = cluster_df[cluster_df['Cluster'] == i+1]
        cluster_stats[f"Cluster_{i+1}"] = {
            "size": len(cluster_data),
            "percentage": f"{(len(cluster_data) / len(df) * 100):.1f}%",
            f"{features[0]}_mean": f"{cluster_data[features[0]].mean():.2f}",
            f"{features[1]}_mean": f"{cluster_data[features[1]].mean():.2f}",
            f"{features[0]}_min": f"{cluster_data[features[0]].min():.2f}",
            f"{features[0]}_max": f"{cluster_data[features[0]].max():.2f}",
            f"{features[1]}_min": f"{cluster_data[features[1]].min():.2f}",
            f"{features[1]}_max": f"{cluster_data[features[1]].max():.2f}",
            "centroid": f"{features[0]}={centers[i][0]:.2f}, {features[1]}={centers[i][1]:.2f}"
        }
        
        # Add other demographic info if available
        if 'Age' in df.columns:
            cluster_stats[f"Cluster_{i+1}"]["age_mean"] = f"{cluster_data['Age'].mean():.1f}"
        
        if 'Gender' in df.columns:
            gender_counts = cluster_data['Gender'].value_counts()
            for gender, count in gender_counts.items():
                percentage = (count / len(cluster_data)) * 100
                cluster_stats[f"Cluster_{i+1}"][f"{gender}_percentage"] = f"{percentage:.1f}%"
    
    # Prepare prompt for Gemini
    prompt = f"""
    You are a marketing and customer segmentation expert. Based on the following customer segments data, provide:
    
    1. A concise summary of each cluster (give them meaningful names based on characteristics)
    2. Marketing strategies tailored for each segment
    3. Product recommendations for each segment
    4. Key business insights from this segmentation
    
    Features used for clustering: {features[0]} and {features[1]}
    
    Cluster statistics:
    {json.dumps(cluster_stats, indent=2)}
    
    Format your response in markdown with clear headings and bullet points.
    """
    
    try:
        model = genai_instance.GenerativeModel('gemini-1.0-pro')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating AI insights: {str(e)}"

# Function to display dataset info with enhanced visuals
def display_data_info(df):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">Dataset Overview</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.dataframe(df.head(5), use_container_width=True)
    
    with col2:
        st.markdown('<div style="background-color: #f8fafc; padding: 1rem; border-radius: 0.5rem;">', unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 1.1rem;'><strong>Dataset Shape:</strong> {df.shape[0]} rows × {df.shape[1]} columns</p>", unsafe_allow_html=True)
        
        # Missing values summary
        missing_values = df.isnull().sum().sum()
        if missing_values > 0:
            st.markdown(f"<p style='color: #F59E0B;'><strong>Missing Values:</strong> {missing_values} ({missing_values/(df.shape[0]*df.shape[1]):.2%})</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='color: #10B981;'><strong>Missing Values:</strong> None</p>", unsafe_allow_html=True)
        
        # Data types summary
        dtype_counts = df.dtypes.value_counts().to_dict()
        dtype_html = "<p><strong>Data Types:</strong></p><ul style='margin-top: 0.5rem;'>"
        for dtype, count in dtype_counts.items():
            dtype_html += f"<li>{dtype}: {count} columns</li>"
        dtype_html += "</ul>"
        st.markdown(dtype_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Statistical summary with enhanced visuals
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">Statistical Summary</div>', unsafe_allow_html=True)
    
    # Get numerical columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(num_cols) > 0:
        # Create a styled statistical summary
        stats_df = df[num_cols].describe().T
        stats_df = stats_df.round(2)
        stats_df['range'] = stats_df['max'] - stats_df['min']
        stats_df['cv'] = (stats_df['std'] / stats_df['mean'] * 100).round(2)  # Coefficient of variation
        
        # Reorder columns
        stats_df = stats_df[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'range', 'cv']]
        
        # Rename columns for better readability
        stats_df.columns = ['Count', 'Mean', 'Std Dev', 'Min', '25%', 'Median', '75%', 'Max', 'Range', 'CV (%)']
        
        st.dataframe(stats_df, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Data visualizations
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">Data Visualizations</div>', unsafe_allow_html=True)
    
    if len(num_cols) > 0:
        viz_tabs = st.tabs(["📊 Distributions", "🔄 Correlations", "📈 Feature Exploration"])
        
        with viz_tabs[0]:
            # Distribution plots using Plotly
            selected_cols = st.multiselect("Select features to visualize:", num_cols, default=num_cols[:min(3, len(num_cols))])
            
            if selected_cols:
                fig = make_subplots(rows=len(selected_cols), cols=1, subplot_titles=[f"Distribution of {col}" for col in selected_cols])
                
                for i, col in enumerate(selected_cols):
                    # Add histogram
                    fig.add_trace(
                        go.Histogram(
                            x=df[col],
                            name=col,
                            marker_color='#4F46E5',
                            opacity=0.7,
                            nbinsx=30
                        ),
                        row=i+1, col=1
                    )
                    
                    # Add KDE (approximated with a line)
                    kde_x, kde_y = sns.kdeplot(data=df[col], bw_adjust=0.5).get_lines()[0].get_data()
                    # Scale KDE to match histogram height
                    hist_max = np.histogram(df[col], bins=30)[0].max()
                    kde_y_scaled = kde_y * (hist_max / kde_y.max())
                    
                    fig.add_trace(
                        go.Scatter(
                            x=kde_x, 
                            y=kde_y_scaled,
                            mode='lines',
                            name=f"{col} KDE",
                            line=dict(color='#10B981', width=2)
                        ),
                        row=i+1, col=1
                    )
                
                fig.update_layout(
                    height=300 * len(selected_cols),
                    showlegend=False,
                    template="plotly_white",
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with viz_tabs[1]:
            if len(num_cols) >= 2:
                # Correlation heatmap
                corr = df[num_cols].corr().round(2)
                
                fig = px.imshow(
                    corr,
                    text_auto=True,
                    color_continuous_scale='Blues',
                    zmin=-1, zmax=1,
                    aspect="auto"
                )
                
                fig.update_layout(
                    title="Feature Correlation Matrix",
                    height=500,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Highlight strong correlations
                strong_corr = []
                for i in range(len(corr.columns)):
                    for j in range(i+1, len(corr.columns)):
                        if abs(corr.iloc[i, j]) > 0.5:  # Threshold for strong correlation
                            strong_corr.append({
                                'Feature 1': corr.columns[i],
                                'Feature 2': corr.columns[j],
                                'Correlation': corr.iloc[i, j]
                            })
                
                if strong_corr:
                    st.markdown("#### Strong Correlations Detected")
                    strong_corr_df = pd.DataFrame(strong_corr)
                    st.dataframe(strong_corr_df, use_container_width=True)
                else:
                    st.info("No strong correlations (>0.5) detected between features.")
        
        with viz_tabs[2]:
            if len(num_cols) >= 2:
                col1, col2 = st.columns(2)
                
                with col1:
                    x_feature = st.selectbox("Select X feature:", num_cols, index=0)
                
                with col2:
                    y_feature = st.selectbox("Select Y feature:", num_cols, index=min(1, len(num_cols)-1))
                
                # Create scatter plot with additional features
                color_by = None
                if 'Gender' in df.columns:
                    color_by = 'Gender'
                elif 'Age' in df.columns and df['Age'].nunique() <= 10:
                    color_by = 'Age'
                
                if color_by:
                    fig = px.scatter(
                        df, x=x_feature, y=y_feature, 
                        color=color_by,
                        opacity=0.7,
                        marginal_x="histogram", 
                        marginal_y="histogram",
                        template="plotly_white"
                    )
                else:
                    fig = px.scatter(
                        df, x=x_feature, y=y_feature,
                        opacity=0.7,
                        marginal_x="histogram", 
                        marginal_y="histogram",
                        color_discrete_sequence=['#4F46E5'],
                        template="plotly_white"
                    )
                
                fig.update_layout(
                    title=f"{x_feature} vs {y_feature}",
                    height=600,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Function to perform data preprocessing
def preprocess_data(df, features):
    # Extract the features
    X = df[features].values
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, scaler

# Function to perform k-means clustering with enhanced visuals
def perform_kmeans(data, features, max_clusters=10, use_scaled=True):
    # Preprocess and scale the data
    X_scaled, scaler = preprocess_data(data, features)
    X = X_scaled if use_scaled else data[features].values
    
    # Calculate WCSS for different number of clusters
    wcss = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    # Plot the Elbow Method graph with Plotly
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">Finding the Optimal Number of Clusters</div>', unsafe_allow_html=True)
    
    fig = go.Figure()
    
    # Add WCSS line
    fig.add_trace(
        go.Scatter(
            x=list(range(1, max_clusters + 1)),
            y=wcss,
            mode='lines+markers',
            name='WCSS',
            marker=dict(size=10, color='#4F46E5'),
            line=dict(width=3, color='#4F46E5')
        )
    )
    
    # Add annotations for the first and last points
    fig.add_annotation(
        x=1,
        y=wcss[0],
        text="High WCSS",
        showarrow=True,
        arrowhead=1,
        ax=40,
        ay=-40
    )
    
    fig.add_annotation(
        x=max_clusters,
        y=wcss[-1],
        text="Low WCSS",
        showarrow=True,
        arrowhead=1,
        ax=-40,
        ay=40
    )
    
    # Calculate the "elbow point" using the maximum curvature
    deltas = np.diff(wcss)
    delta_deltas = np.diff(deltas)
    elbow_point = np.argmax(delta_deltas) + 2  # +2 because of two diff operations
    
    # Add a marker for the suggested elbow point
    fig.add_trace(
        go.Scatter(
            x=[elbow_point],
            y=[wcss[elbow_point-1]],
            mode='markers',
            marker=dict(size=15, color='#F59E0B', line=dict(width=2, color='black')),
            name='Suggested Elbow Point'
        )
    )
    
    fig.add_annotation(
        x=elbow_point,
        y=wcss[elbow_point-1],
        text="Suggested Optimal",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-60
    )
    
    fig.update_layout(
        title="The Elbow Method for Optimal K",
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Within-Cluster Sum of Squares (WCSS)",
        template="plotly_white",
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div class="info-box">
        <p><strong>How to interpret:</strong> The "elbow point" in the graph above indicates the optimal number of clusters. 
        After this point, adding more clusters provides diminishing returns in terms of explained variance.</p>
        <p>The suggested optimal number of clusters is highlighted in orange.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Let user select the number of clusters based on the elbow graph
    optimal_clusters = st.slider(
        "Select number of clusters:", 
        min_value=2, 
        max_value=max_clusters, 
        value=elbow_point
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Perform clustering with the selected number of clusters
    kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42, n_init=10)
    y_kmeans = kmeans.fit_predict(X)
    
    # Get the centroids
    centers_scaled = kmeans.cluster_centers_
    
    # Convert the centroids back to original scale if scaled data was used
    if use_scaled:
        centers = scaler.inverse_transform(centers_scaled)
    else:
        centers = centers_scaled
    
    # Plot the clusters with Plotly
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<div class="card-header">Customer Segments ({optimal_clusters} Clusters)</div>', unsafe_allow_html=True)
    
    # Define colors for the clusters
    colors = px.colors.qualitative.Bold
    
    # Create a DataFrame for plotting
    plot_df = pd.DataFrame({
        features[0]: data[features[0]],
        features[1]: data[features[1]],
        'Cluster': [f'Cluster {i+1}' for i in y_kmeans]
    })
    
    # Add additional information if available
    if 'Age' in data.columns:
        plot_df['Age'] = data['Age']
    
    if 'Gender' in data.columns:
        plot_df['Gender'] = data['Gender']
    
    # Create hover text
    hover_data = {features[0]: True, features[1]: True, 'Cluster': True}
    if 'Age' in plot_df.columns:
        hover_data['Age'] = True
    if 'Gender' in plot_df.columns:
        hover_data['Gender'] = True
    
    # Create scatter plot
    fig = px.scatter(
        plot_df, 
        x=features[0], 
        y=features[1],
        color='Cluster',
        color_discrete_sequence=colors[:optimal_clusters],
        hover_data=hover_data,
        opacity=0.7,
        size_max=10
    )
    
    # Add centroids
    for i in range(optimal_clusters):
        fig.add_trace(
            go.Scatter(
                x=[centers[i, 0]],
                y=[centers[i, 1]],
                mode='markers',
                marker=dict(
                    color='black',
                    size=15,
                    symbol='star',
                    line=dict(width=2, color=colors[i % len(colors)])
                ),
                name=f'Centroid {i+1}'
            )
        )
    
    # Add density contours for each cluster
    for i in range(optimal_clusters):
        cluster_data = plot_df[plot_df['Cluster'] == f'Cluster {i+1}']
        if len(cluster_data) > 10:  # Only add contours if enough points
            fig.add_trace(
                go.Histogram2dContour(
                    x=cluster_data[features[0]],
                    y=cluster_data[features[1]],
                    colorscale=[[0, 'rgba(0,0,0,0)'], [1, colors[i % len(colors)]]],
                    showscale=False,
                    opacity=0.3,
                    name=f'Density {i+1}',
                    hoverinfo='none'
                )
            )
    
    fig.update_layout(
        title=f"Customer Segmentation based on {features[0]} and {features[1]}",
        xaxis_title=features[0],
        yaxis_title=features[1],
        template="plotly_white",
        height=600,
        legend=dict(
            title="Segments",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 3D visualization if age is available
    if 'Age' in data.columns:
        st.markdown('<div class="sub-header">3D Visualization with Age</div>', unsafe_allow_html=True)
        
        plot_df_3d = plot_df.copy()
        
        fig_3d = px.scatter_3d(
            plot_df_3d,
            x=features[0],
            y=features[1],
            z='Age',
            color='Cluster',
            color_discrete_sequence=colors[:optimal_clusters],
            opacity=0.7,
            size_max=10,
            hover_data=hover_data
        )
        
        # Add 3D centroids if we have age data for them
        if 'Age' in data.columns:
            centroid_ages = []
            for i in range(optimal_clusters):
                cluster_data = data[y_kmeans == i]
                centroid_ages.append(cluster_data['Age'].mean())
            
            for i in range(optimal_clusters):
                fig_3d.add_trace(
                    go.Scatter3d(
                        x=[centers[i, 0]],
                        y=[centers[i, 1]],
                        z=[centroid_ages[i]],
                        mode='markers',
                        marker=dict(
                            color='black',
                            size=8,
                            symbol='diamond',
                            line=dict(width=2, color=colors[i % len(colors)])
                        ),
                        name=f'Centroid {i+1}'
                    )
                )
        
        fig_3d.update_layout(
            title="3D Customer Segmentation with Age",
            template="plotly_white",
            height=700,
            scene=dict(
                xaxis_title=features[0],
                yaxis_title=features[1],
                zaxis_title='Age'
            )
        )
        
        st.plotly_chart(fig_3d, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

    return X, y_kmeans, centers

def create_sample_data():
    # Create sample data similar to Mall_Customers.csv
    np.random.seed(42)  # For reproducibility
    n_samples = 200
    
    # Customer IDs
    customer_ids = range(1, n_samples + 1)
    
    # Gender (60% Female, 40% Male)
    genders = np.random.choice(['Female', 'Male'], size=n_samples, p=[0.6, 0.4])
    
    # Age (18-70)
    ages = np.random.normal(38, 12, n_samples).astype(int)
    ages = np.clip(ages, 18, 70)  # Ensure ages are between 18 and 70
    
    # Annual Income - create 5 income groups
    means = [30, 55, 70, 85, 120]
    stds = [5, 7, 8, 10, 15]
    weights = [0.3, 0.3, 0.2, 0.15, 0.05]  # More people in lower and middle income ranges
    
    income_group = np.random.choice(5, size=n_samples, p=weights)
    annual_income = np.array([np.random.normal(means[g], stds[g]) for g in income_group])
    annual_income = np.clip(annual_income, 15, 150).astype(int)  # Ensure income is between 15k and 150k
    
    # Spending Score - create a pattern with 5 spending groups that somewhat correlate with income
    # but with interesting patterns (like high income low spenders and low income high spenders)
    ss_means = [20, 40, 50, 75, 90]  # Spending score means
    ss_stds = [10, 12, 15, 12, 8]  # Standard deviations
    
    # Create a base spending score from income groups with some variation
    base_spending = np.array([np.random.normal(ss_means[g], ss_stds[g]) for g in income_group])
    
    # Add some random variation to create interesting patterns
    # Some high income, low spenders
    high_income_mask = (annual_income > 80)
    low_spend_mask = np.random.choice([True, False], size=n_samples, p=[0.3, 0.7])
    base_spending[high_income_mask & low_spend_mask] = np.random.normal(30, 10, size=np.sum(high_income_mask & low_spend_mask))
    
    # Some low income, high spenders
    low_income_mask = (annual_income < 40)
    high_spend_mask = np.random.choice([True, False], size=n_samples, p=[0.25, 0.75])
    base_spending[low_income_mask & high_spend_mask] = np.random.normal(80, 10, size=np.sum(low_income_mask & high_spend_mask))
    
    # Clip spending score between 1 and 100
    spending_score = np.clip(base_spending, 1, 100).astype(int)
    
    # Create the DataFrame
    data = {
        'CustomerID': customer_ids,
        'Gender': genders,
        'Age': ages,
        'Annual Income (k$)': annual_income,
        'Spending Score (1-100)': spending_score
    }
    return pd.DataFrame(data)

# Function to display cluster insights with enhanced visuals
def display_cluster_insights(df, result_df, centers, features):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">Cluster Analysis</div>', unsafe_allow_html=True)
    
    # Get unique clusters
    unique_clusters = sorted(result_df['Cluster'].unique())
    
    # Create tabs for different views
    insight_tabs = st.tabs(["📊 Overview", "🔍 Cluster Details", "📈 Comparison"])
    
    with insight_tabs[0]:
        # Create cluster summary table with enhanced styling
        cluster_summary = []
        for i in range(len(centers)):
            cluster_i = i + 1
            cluster_data = result_df[result_df['Cluster'] == cluster_i]
            
            cluster_info = {
                'Cluster': f"Cluster {cluster_i}",
                'Size': len(cluster_data),
                'Percentage': f"{(len(cluster_data) / len(df) * 100):.1f}%",
                f"{features[0]} (Mean)": f"{cluster_data[features[0]].mean():.2f}",
                f"{features[1]} (Mean)": f"{cluster_data[features[1]].mean():.2f}",
                'Centroid': f"({centers[i][0]:.2f}, {centers[i][1]:.2f})"
            }
            
            if 'Age' in df.columns:
                cluster_info['Age (Mean)'] = f"{cluster_data['Age'].mean():.1f}"
            
            if 'Gender' in df.columns:
                gender_counts = cluster_data['Gender'].value_counts().to_dict()
                for gender, count in gender_counts.items():
                    percentage = (count / len(cluster_data)) * 100
                    cluster_info[f'{gender} %'] = f"{percentage:.1f}%"
            
            cluster_summary.append(cluster_info)
        
        cluster_summary_df = pd.DataFrame(cluster_summary)
        st.dataframe(cluster_summary_df, use_container_width=True)
        
        # Visualize cluster sizes
        fig = px.pie(
            cluster_summary_df, 
            values='Size', 
            names='Cluster',
            color='Cluster',
            color_discrete_sequence=px.colors.qualitative.Bold,
            hole=0.4
        )
        
        fig.update_layout(
            title="Cluster Size Distribution",
            template="plotly_white",
            height=500
        )
        
        fig.update_traces(
            textposition='inside', 
            textinfo='percent+label',
            hoverinfo='label+percent+value'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature distribution across clusters
        feature_comparison = []
        for feature in features:
            for i in range(len(centers)):
                cluster_i = i + 1
                cluster_data = result_df[result_df['Cluster'] == cluster_i]
                
                feature_comparison.append({
                    'Feature': feature,
                    'Cluster': f"Cluster {cluster_i}",
                    'Mean Value': cluster_data[feature].mean(),
                    'Min Value': cluster_data[feature].min(),
                    'Max Value': cluster_data[feature].max()
                })
        
        feature_comparison_df = pd.DataFrame(feature_comparison)
        
        fig = px.bar(
            feature_comparison_df,
            x='Cluster',
            y='Mean Value',
            color='Feature',
            barmode='group',
            error_y=[row['Max Value'] - row['Mean Value'] for _, row in feature_comparison_df.iterrows()],
            error_y_minus=[row['Mean Value'] - row['Min Value'] for _, row in feature_comparison_df.iterrows()],
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        
        fig.update_layout(
            title="Feature Distribution Across Clusters",
            template="plotly_white",
            height=500,
            xaxis_title="",
            yaxis_title="Value"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with insight_tabs[1]:
        # Select which cluster to display
        selected_cluster = st.selectbox(
            "Select a cluster to view details:",
            ["All Clusters"] + [f"Cluster {c}" for c in unique_clusters]
        )
        
        if selected_cluster == "All Clusters":
            st.dataframe(result_df, use_container_width=True)
        else:
            cluster_num = int(selected_cluster.split()[1])
            cluster_data = result_df[result_df['Cluster'] == cluster_num]
            
            # Display cluster data
            st.dataframe(cluster_data, use_container_width=True)
            
            # Cluster metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-value">{len(cluster_data)}</div>
                        <div class="metric-label">Customers</div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            
            with col2:
                percentage = (len(cluster_data) / len(df)) * 100
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-value">{percentage:.1f}%</div>
                        <div class="metric-label">of Total</div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            
            with col3:
                if 'Age' in df.columns:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-value">{cluster_data['Age'].mean():.1f}</div>
                            <div class="metric-label">Avg. Age</div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-value">N/A</div>
                            <div class="metric-label">Avg. Age</div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
            
            with col4:
                if 'Gender' in df.columns:
                    most_common_gender = cluster_data['Gender'].value_counts().idxmax()
                    gender_percentage = (cluster_data['Gender'].value_counts().max() / len(cluster_data)) * 100
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-value">{most_common_gender}</div>
                            <div class="metric-label">{gender_percentage:.1f}% of Cluster</div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-value">N/A</div>
                            <div class="metric-label">Gender</div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
            
            # Feature distributions within the cluster
            st.markdown("#### Feature Distributions")
            
            # Create distribution plots for the selected cluster
            feature_figs = []
            for feature in features:
                fig = go.Figure()
                
                # Add histogram for the selected cluster
                fig.add_trace(
                    go.Histogram(
                        x=cluster_data[feature],
                        name=f"{selected_cluster}",
                        marker_color='#4F46E5',
                        opacity=0.7,
                        nbinsx=20
                    )
                )
                
                # Add histogram for all data (transparent)
                fig.add_trace(
                    go.Histogram(
                        x=df[feature],
                        name="All Data",
                        marker_color='gray',
                        opacity=0.3,
                        nbinsx=20
                    )
                )
                
                fig.update_layout(
                    title=f"Distribution of {feature}",
                    xaxis_title=feature,
                    yaxis_title="Count",
                    template="plotly_white",
                    height=300,
                    barmode='overlay',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                feature_figs.append(fig)
            
            # Display the plots in columns
            cols = st.columns(len(feature_figs))
            for i, fig in enumerate(feature_figs):
                with cols[i]:
                    st.plotly_chart(fig, use_container_width=True)
    
    with insight_tabs[2]:
        # Create radar chart for comparing clusters
        if len(features) >= 2:
            # Normalize features for radar chart
            radar_data = []
            
            # Get min and max values for normalization
            min_values = {}
            max_values = {}
            for feature in features:
                min_values[feature] = df[feature].min()
                max_values[feature] = df[feature].max()
            
            if 'Age' in df.columns:
                min_values['Age'] = df['Age'].min()
                max_values['Age'] = df['Age'].max()
            
            # Create radar data
            for i in range(len(centers)):
                cluster_i = i + 1
                cluster_data = result_df[result_df['Cluster'] == cluster_i]
                
                radar_point = {'Cluster': f"Cluster {cluster_i}"}
                
                for feature in features:
                    # Normalize to 0-1 scale
                    normalized_value = (cluster_data[feature].mean() - min_values[feature]) / (max_values[feature] - min_values[feature])
                    radar_point[feature] = normalized_value
                
                if 'Age' in df.columns:
                    normalized_age = (cluster_data['Age'].mean() - min_values['Age']) / (max_values['Age'] - min_values['Age'])
                    radar_point['Age'] = normalized_age
                
                radar_data.append(radar_point)
            
            # Create radar chart
            radar_df = pd.DataFrame(radar_data)
            
            # Get feature columns for radar
            radar_features = features.copy()
            if 'Age' in df.columns:
                radar_features.append('Age')
            
            fig = go.Figure()
            
            for _, row in radar_df.iterrows():
                cluster_name = row['Cluster']
                cluster_index = int(cluster_name.split()[1]) - 1
                
                fig.add_trace(
                    go.Scatterpolar(
                        r=[row[f] for f in radar_features],
                        theta=radar_features,
                        fill='toself',
                        name=cluster_name,
                        line_color=px.colors.qualitative.Bold[cluster_index % len(px.colors.qualitative.Bold)]
                    )
                )
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                title="Cluster Comparison (Normalized Values)",
                template="plotly_white",
                height=600,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add explanation
            st.markdown("""
            <div class="info-box">
                <p><strong>How to interpret:</strong> The radar chart shows normalized values (0-1 scale) for each feature across clusters. 
                This allows for easy comparison of cluster characteristics regardless of the original feature scales.</p>
                <p>Clusters with similar shapes have similar patterns, while the size of the shape indicates the overall magnitude of the values.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Feature comparison bar chart
            st.markdown("#### Feature Comparison Across Clusters")
            
            # Select feature to compare
            feature_to_compare = st.selectbox(
                "Select feature to compare:",
                features + (['Age'] if 'Age' in df.columns else [])
            )
            
            # Create comparison data
            comparison_data = []
            for i in range(len(centers)):
                cluster_i = i + 1
                cluster_data = result_df[result_df['Cluster'] == cluster_i]
                
                comparison_data.append({
                    'Cluster': f"Cluster {cluster_i}",
                    'Mean': cluster_data[feature_to_compare].mean(),
                    'Min': cluster_data[feature_to_compare].min(),
                    'Max': cluster_data[feature_to_compare].max(),
                    'Std Dev': cluster_data[feature_to_compare].std()
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Create bar chart with error bars
            fig = go.Figure()
            
            # Add bars for mean values
            fig.add_trace(
                go.Bar(
                    x=comparison_df['Cluster'],
                    y=comparison_df['Mean'],
                    error_y=dict(
                        type='data',
                        array=comparison_df['Std Dev'],
                        visible=True
                    ),
                    marker_color=px.colors.qualitative.Bold[:len(comparison_df)],
                    name='Mean Value'
                )
            )
            
            # Add markers for min and max
            fig.add_trace(
                go.Scatter(
                    x=comparison_df['Cluster'],
                    y=comparison_df['Min'],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-down',
                        size=10,
                        color='rgba(0,0,0,0.7)'
                    ),
                    name='Minimum'
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=comparison_df['Cluster'],
                    y=comparison_df['Max'],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up',
                        size=10,
                        color='rgba(0,0,0,0.7)'
                    ),
                    name='Maximum'
                )
            )
            
            fig.update_layout(
                title=f"Comparison of {feature_to_compare} Across Clusters",
                xaxis_title="",
                yaxis_title=feature_to_compare,
                template="plotly_white",
                height=500,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Download option for results
    csv = result_df.to_csv(index=False)
    st.download_button(
        label="Download clustered data as CSV",
        data=csv,
        file_name="customer_segments.csv",
        mime="text/csv",
    )
    
    st.markdown('</div>', unsafe_allow_html=True)

# Main application
def main():
    # Render header with logo and title
    render_header()
    
    # Sidebar with options
    st.sidebar.markdown('<div style="text-align: center; padding: 1rem 0;"><h2 style="color: #4F46E5; margin: 0;">⚙️ Configuration</h2></div>', unsafe_allow_html=True)
    
    # Data input section
    st.sidebar.markdown('<p style="font-size: 1.2rem; font-weight: 600; color: #4F46E5; margin-bottom: 0.5rem;">Data Source</p>', unsafe_allow_html=True)
    data_source = st.sidebar.radio(
        "Select data source:",
        ["Upload CSV", "Use sample data"]
    )
    
    # Load data
    df = None
    if data_source == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader("Upload customer data CSV", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.sidebar.markdown('<div class="success-box">✅ File uploaded successfully!</div>', unsafe_allow_html=True)
    else:
        if st.sidebar.button("Generate Sample Data", key="generate_sample"):
            with st.spinner("Generating sample data..."):
                df = create_sample_data()
                st.sidebar.markdown('<div class="success-box">✅ Sample data generated!</div>', unsafe_allow_html=True)
    
    # If data is loaded
    if df is not None:
        # Sidebar options for clustering
        st.sidebar.markdown('<p style="font-size: 1.2rem; font-weight: 600; color: #4F46E5; margin: 1rem 0 0.5rem 0;">Clustering Settings</p>', unsafe_allow_html=True)
        
        # Get numerical columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Let user select features
        default_features = []
        if 'Annual Income (k$)' in numeric_cols and 'Spending Score (1-100)' in numeric_cols:
            default_features = ['Annual Income (k$)', 'Spending Score (1-100)']
        elif len(numeric_cols) >= 2:
            default_features = numeric_cols[:2]
        
        selected_features = st.sidebar.multiselect(
            "Select features for clustering:", 
            options=numeric_cols,
            default=default_features
        )
        
        # Advanced settings
        st.sidebar.markdown('<p style="font-size: 1rem; font-weight: 500; color: #4F46E5; margin: 1rem 0 0.5rem 0;">Advanced Settings</p>', unsafe_allow_html=True)
        
        use_scaled_data = st.sidebar.checkbox("Scale data before clustering", value=True)
        
        max_clusters = st.sidebar.slider(
            "Maximum number of clusters:", 
            min_value=2, 
            max_value=15, 
            value=10
        )
        
        # Tabs for different sections
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Exploration", "🔍 Clustering Analysis", "🤖 AI Insights", "📋 Documentation"])
        
        with tab1:
            display_data_info(df)
        
        with tab2:
            # Perform clustering if two features selected
            if len(selected_features) == 2:
                X, y_kmeans, centers = perform_kmeans(df, selected_features, max_clusters, use_scaled_data)
                
                # Show cluster results
                if X is not None and y_kmeans is not None:
                    # Add cluster labels to the dataframe
                    result_df = df.copy()
                    result_df['Cluster'] = y_kmeans + 1  # Adding 1 to make it 1-based instead of 0-based
                    
                    # Display cluster insights
                    display_cluster_insights(df, result_df, centers, selected_features)
            else:
                st.markdown("""
                <div class="warning-box">
                    <p>⚠️ Please select exactly 2 features to perform clustering.</p>
                    <p>K-means clustering works best with 2 features for visualization purposes. You can select features from the sidebar.</p>
                </div>
                """, unsafe_allow_html=True)
        
        with tab3:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-header">AI-Generated Marketing Insights</div>', unsafe_allow_html=True)
            
            if len(selected_features) == 2:
                if 'y_kmeans' in locals() and 'centers' in locals():
                    # Add cluster labels to the dataframe if not already done
                    if 'result_df' not in locals():
                        result_df = df.copy()
                        result_df['Cluster'] = y_kmeans + 1
                    
                    if st.button("Generate AI Insights", key="generate_insights"):
                        with st.spinner("AI is analyzing your customer segments..."):
                            insights = generate_ai_insights(df, result_df, selected_features, centers)
                            st.markdown(insights)
                else:
                    st.info("Please run the clustering analysis first.")
            else:
                st.info("Please select exactly 2 features and run clustering to generate AI insights.")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab4:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-header">Project Documentation</div>', unsafe_allow_html=True)
            
            st.markdown("""
            ### Customer Segmentation with K-means Clustering
            
            This platform uses K-means clustering to segment customers based on their characteristics and behaviors. The segmentation helps businesses understand different customer groups and develop targeted marketing strategies.
            
            #### Methodology
            
            1. **Data Preparation**: Upload your customer data or use our sample dataset
            2. **Feature Selection**: Choose two features for clustering (e.g., Annual Income and Spending Score)
            3. **K-means Clustering**: The algorithm groups similar customers together
            4. **Visualization**: Interactive plots help you understand the segments
            5. **AI Analysis**: Get AI-generated insights for each segment
            
            #### How to Use This Platform
            
            1. **Upload Data**: Start by uploading your customer data CSV or generating sample data
            2. **Configure Settings**: Select features and adjust clustering parameters
            3. **Explore Results**: Analyze the clusters and their characteristics
            4. **Generate Insights**: Use AI to get marketing recommendations
            5. **Download Results**: Export the segmented data for further analysis
            
            #### Interpreting Results
            
            - **Elbow Method**: Helps determine the optimal number of clusters
            - **Cluster Plot**: Shows how customers are grouped in feature space
            - **Cluster Statistics**: Provides detailed metrics for each segment
            - **Comparison Tools**: Compare characteristics across segments
            
            #### Business Applications
            
            - **Targeted Marketing**: Develop personalized campaigns for each segment
            - **Product Development**: Create offerings tailored to specific customer groups
            - **Pricing Strategy**: Optimize pricing based on segment characteristics
            - **Customer Retention**: Identify at-risk segments and develop retention strategies
            - **Resource Allocation**: Focus resources on the most valuable segments
            
            #### Technical Details
            
            The platform uses the following technologies:
            
            - **Python**: Core programming language
            - **Streamlit**: Web application framework
            - **Scikit-learn**: For K-means clustering algorithm
            - **Plotly**: For interactive visualizations
            - **Google's Gemini AI**: For generating marketing insights
            """)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>AI-Enhanced Customer Segmentation Platform | Version 1.0</p>
        <p>Created with ❤️ using Python, Streamlit, and Plotly</p>
        <p>Last updated: {}</p>
    </div>
    """.format(datetime.now().strftime("%B %d, %Y")), unsafe_allow_html=True)

if __name__ == "__main__":
    main()