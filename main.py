import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("star_with_gravity.csv")
df.head()

X = df.iloc[:,[3,4]].values

wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',random_state=42)
    kmeans.fit(X)
    wcss.append((kmeans.inertia_))
plt.plot(range(1,11),wcss)
plt.title("elbow method")
plt.xlabel('Number of clusters')
plt.show()