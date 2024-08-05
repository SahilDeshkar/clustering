# -*- coding: utf-8 -*-
"""K-means_clustering.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1T06ZM0xY_-KE7HzmUlVvFTTOox0NN3X0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

customer_data=pd.read_csv('//content//Mall_Customers.csv')

customer_data.head()

customer_data.shape

customer_data.info()

customer_data.describe()

x=customer_data.iloc[:,[3,4]].values

print(x)

wcss=[]

for i in range(1,11):
  kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
  kmeans.fit(x)

  wcss.append(kmeans.inertia_)

sns.set()
plt.plot(range(1,11),wcss)
plt.title(' The elbow point graph ')
plt.xlabel ('number of cluster')
plt.ylabel('wcss')
plt.show()

kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)
  Y=kmeans.fit_predict(x)
  print(Y)

plt.figure(figsize=(8,8))
plt.scatter(x[Y==0,0],x[Y==0,1], s=50 , c='green', label='cluster 1')
plt.scatter(x[Y==1,0],x[Y==1,1], s=50 , c='yelloW', label='cluster 2')
plt.scatter(x[Y==2,0],x[Y==2,1], s=50 , c='blue', label='cluster 3')
plt.scatter(x[Y==3,0],x[Y==3,1], s=50 , c='violet', label='cluster 4')
plt.scatter(x[Y==4,0],x[Y==4,1], s=50 , c='black', label='cluster 5')
clusters=kmeans.cluster_centers_
plt.scatter(clusters[:,0], clusters[:,1],s=100,c='red', label='centroid')

plt.title('customer segments')
plt.xlabel('Annual income')
plt.ylabel('spending score')
plt.show()


