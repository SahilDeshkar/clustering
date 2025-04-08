# 🧠 Customer Segmentation with K-Means Clustering

This platform leverages **K-Means Clustering** to segment customers based on their characteristics and behaviors. These insights enable businesses to understand different customer groups and build data-driven, targeted marketing strategies.

---

## 🚀 Features

- **Data Upload**: Import your own customer data (CSV) or use sample data
- **Feature Selection**: Choose features like Annual Income and Spending Score
- **Interactive Clustering**: Segment customers using K-Means clustering
- **Data Visualization**: Explore clusters with interactive Plotly visualizations
- **AI-Generated Insights**: Get actionable marketing suggestions using Gemini AI
- **Download Results**: Export segmented data for further use

---

## 🔍 Methodology

1. **Data Preparation**  
   Upload your own dataset or use a built-in sample dataset.

2. **Feature Selection**  
   Select two features (e.g., `Annual Income`, `Spending Score`) for clustering.

3. **K-Means Clustering**  
   Segment your customers into groups based on feature similarity.

4. **Visualization**  
   Interactive charts help you visually understand your customer segments.

5. **AI Analysis**  
   Get automated marketing insights for each segment using Google's Gemini AI.

---

## 🧑‍💻 How to Use

1. **Upload Data**  
   Upload a CSV file with customer data or generate a sample dataset.

2. **Configure Settings**  
   Select two features for clustering and adjust the number of clusters.

3. **Explore Results**  
   Analyze clusters using visual tools and summary statistics.

4. **Generate Insights**  
   Click to generate AI-powered marketing recommendations.

5. **Download Results**  
   Export your segmented dataset as a CSV.

---

## 📊 Interpreting Results

- **Elbow Method**: Suggests the optimal number of clusters.
- **Cluster Plot**: Shows customer groupings in 2D feature space.
- **Cluster Statistics**: Summary metrics for each segment.
- **Comparison Tools**: Compare characteristics between segments.

---

## 💼 Business Applications

- 🎯 **Targeted Marketing**: Create personalized campaigns for each segment.
- 🧪 **Product Development**: Design offerings tailored to group needs.
- 💸 **Pricing Strategy**: Adjust pricing strategies by segment.
- 🤝 **Customer Retention**: Identify and retain at-risk customers.
- 📈 **Resource Allocation**: Focus on high-value segments.

---

## ⚙️ Tech Stack

- **Python** – Core programming language  
- **Streamlit** – Web app framework  
- **Scikit-learn** – Machine learning and clustering  
- **Plotly** – Interactive data visualizations  
- **Google's Gemini AI** – AI-generated business insights  

---

## 📦 Installation

To run locally:

```bash
pip install -r requirements.txt
streamlit run app.py
