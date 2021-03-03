from sklearn.decomposition import PCA 
import pandas as pd
import pdb 
import numpy as np 
import plotly.express as px
from sklearn.cluster import KMeans 
from sklearn import metrics 
from scipy.spatial.distance import cdist 
import matplotlib.pyplot as plt
df = pd.read_csv('output.csv')
df_pca = df.drop('Unnamed: 0', axis = 1)
labels = df['Unnamed: 0'].str.contains('non_autistic') 
"""nums = range(1,10)
distances = []
for n in nums: 
    kmeanModel = KMeans(n_clusters=n)
    kmeanModel.fit(df_pca)
    distances.append(kmeanModel.inertia_)
plt.plot(nums, distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()"""

kmeanModel = KMeans(n_clusters=4, random_state=42)
kmeanModel.fit(df_pca)
predict = kmeanModel.predict(df_pca)
predict = pd.Series(predict, index = df_pca.index)

df['color'] = df['Unnamed: 0'].str.contains('non_autistic')
df['color'].loc[df['color'] == True] = 'Non Autistic'
df['color'].loc[df['color'] == False] = 'Autistic'
pca = PCA(n_components=3) 
out = pca.fit_transform(df_pca)
out = pd.concat([pd.DataFrame(out),df['color']], axis = 1)
out['cluster'] = predict.astype('category')
out["Principal Component 1"] = out[0]
out["Principal Component 2"] = out[1]
out["Principal Component 3"] = out[2]
fig = px.scatter_3d(out, x="Principal Component 1", y="Principal Component 2", z="Principal Component 3",color=out['cluster'], symbol = df['color'],hover_name=df['Unnamed: 0'])

fig.show()







