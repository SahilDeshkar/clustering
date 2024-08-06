import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import io

# Set the title of the app
st.title("K-means Clustering of Mall Customers")

# Upload the dataset
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load the data
    customer_data = pd.read_csv(uploaded_file)

    # Display the first few rows of the dataframe
    st.write("Customer Data", customer_data.head())

    # Display the shape of the dataframe
    st.write("Shape of the dataset:", customer_data.shape)

    # Display dataset info
    buffer = io.StringIO()
    customer_data.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    # Display dataset description
    st.write("Description of the dataset", customer_data.describe())

    # Select columns for clustering
    x = customer_data.iloc[:, [3, 4]].values

    # Plot the Elbow Method
    st.write("### Elbow Method for Optimal Number of Clusters")
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(x)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(10, 5))
    sns.set()
    plt.plot(range(1, 11), wcss)
    plt.title('The Elbow Point Graph')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    st.pyplot()

    # Choose the number of clusters
    st.write("### Choose the Number of Clusters")
    n_clusters = st.slider("Number of clusters", 1, 10, 5)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=0)
    Y = kmeans.fit_predict(x)

    # Plot the clusters
    st.write("### Customer Segments")
    plt.figure(figsize=(8, 8))
    colors = ['green', 'yellow', 'blue', 'violet', 'black', 'orange', 'pink', 'brown', 'cyan', 'magenta']
    for i in range(n_clusters):
        plt.scatter(x[Y == i, 0], x[Y == i, 1], s=50, c=colors[i % len(colors)], label=f'Cluster {i + 1}')
    clusters = kmeans.cluster_centers_
    plt.scatter(clusters[:, 0], clusters[:, 1], s=100, c='red', label='Centroids')
    plt.title('Customer Segments')
    plt.xlabel('Annual Income')
    plt.ylabel('Spending Score')
    plt.legend()
    st.pyplot()

    # Add a section for cluster information
    st.write("### Cluster Information")
    cluster_data = customer_data.copy()
    cluster_data['Cluster'] = Y

    for i in range(n_clusters):
        st.write(f"**Cluster {i + 1}**")
        st.write(cluster_data[cluster_data['Cluster'] == i].describe())
