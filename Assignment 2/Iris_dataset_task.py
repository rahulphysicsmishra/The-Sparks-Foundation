import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import OrdinalEncoder
# Loading iris dataset and separating input and output
iris = pd.read_csv('iris.csv')
X = np.c_[iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]]
y = np.c_[iris['Species']]

# Converting species to numerical value by using OrdinalEncoder
ord_encoder_species = OrdinalEncoder()
ord_encoded_species = ord_encoder_species.fit_transform(y)
# print(ord_encoded_species[:10])

# Creating the data
X1 = np.array(list(zip(X, ord_encoded_species)), dtype=object).reshape(len(X), 2)
print(X1)

# Visualising the data
import seaborn as sns
iris = sns.load_dataset('iris')
sns.set_style('whitegrid')
sns.FacetGrid(iris, hue='species', height=6).map(plt.scatter,
                                                  'sepal_length',
                                                  'petal_length').add_legend()

plt.show()

# Building the clustering model and calculating inertias

inertias = []
mapping = {}
K = range(1, 9)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(X)
    inertias.append(kmeanModel.inertia_)
    mapping[k] = kmeanModel.inertia_

# Checking inertia values for each value of k
for key, val in mapping.items():
    print(f'{key}: {val}')

# here are the values for each inertias for each k
# 1: 680.8243999999996
# 2: 152.36870647733915
# 3: 78.94084142614601
# 4: 57.34540931571815
# 5: 46.535582051282034
# 6: 39.251830892636775
# 7: 34.62085338680927
# 8: 29.90685675596547

# After K = 3, there's no significant drop in inertias.
# So, K=3 fits best.

# Plotting elbow graph for each value of k and inertias to show why K=3 fits best.
plt.plot(K, inertias, 'bx-')
plt.show()

