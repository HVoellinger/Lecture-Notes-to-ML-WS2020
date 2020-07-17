#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sklearn.metrics as sm
import math
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


iris = datasets.load_iris()
print(iris.data)


# In[3]:


print(iris.target_names)


# In[4]:


print(iris.target)


# In[5]:


x = pd.DataFrame(iris.data, columns=['Sepal Length', 'Sepal Width', 
                                     'Petal Length', 'Petal Width'])
y = pd.DataFrame(iris.target, columns=['Target'])


# In[6]:


x.head()


# In[7]:


y.head()


# In[8]:


iris_k_mean_model = KMeans(n_clusters=3)
iris_k_mean_model.fit(x)


# In[9]:


print(iris_k_mean_model.labels_)


# In[11]:


print(iris_k_mean_model.cluster_centers_)


# In[12]:


predictedY = np.choose(iris_k_mean_model.labels_, [0, 1, 2]).astype(np.int64)


# In[13]:


sm.accuracy_score(predictedY, y['Target'])


# ## Interpretation of Confusion Matrix
# Correctly identifed all 0 classes as 0’s
# correctly classified 48 class 1’s but miss-classified 2 class 1’s as class 2
# correctly classified 36 class 2’s but miss-classified 14 class 2’s as class 1

# In[14]:


sm.confusion_matrix(predictedY, y['Target'])


# In[15]:


x.shape


# In[16]:


dist_points_from_cluster_center = []
K = range(1,10)
for no_of_clusters in K:
  k_model = KMeans(n_clusters=no_of_clusters)
  k_model.fit(x)
  dist_points_from_cluster_center.append(k_model.inertia_)


# In[17]:


dist_points_from_cluster_center


# In[18]:


plt.plot(K, dist_points_from_cluster_center)


# In[19]:


plt.plot(K, dist_points_from_cluster_center)
plt.plot([K[0], K[8]], [dist_points_from_cluster_center[0], 
                        dist_points_from_cluster_center[8]], 'ro-')
plt.show()


# In[20]:


x = [K[0], K[8]]
y = [dist_points_from_cluster_center[0], dist_points_from_cluster_center[8]]

# Calculate the coefficients. This line answers the initial question. 
coefficients = np.polyfit(x, y, 1)

# Print the findings
print('a =', coefficients[0])
print ('b =', coefficients[1])

# Let's compute the values of the line...
polynomial = np.poly1d(coefficients)
x_axis = np.linspace(0,9,100)
y_axis = polynomial(x_axis)

# ...and plot the points and the line
plt.plot(x_axis, y_axis)
plt.grid('on')
plt.show()


# In[21]:


# Function to find distance
# https://www.geeksforgeeks.org/perpendicular-distance-
# between-a-point-and-a-line-in-2-d/
def calc_distance(x1, y1, a, b, c):
  d = abs((a * x1 + b * y1 + c)) / (math.sqrt(a * a + b * b))
  return d


# In[22]:


# (y1 – y2)x + (x2 – x1)y + (x1y2 – x2y1) = 0
# https://bobobobo.wordpress.com/2008/01/07/solving-linear-equations-ax-by-c-0/
a = dist_points_from_cluster_center[0] - dist_points_from_cluster_center[8]
b = K[8] - K[0]
c1 = K[0] * dist_points_from_cluster_center[8]
c2 = K[8] * dist_points_from_cluster_center[0]
c = c1 - c2


# In[23]:


dist_points_from_cluster_center


# In[24]:


distance_of_points_from_line = []
for k in range(9):
  distance_of_points_from_line.append(
      calc_distance(K[k], dist_points_from_cluster_center[k], a, b, c))


# In[25]:


distance_of_points_from_line


# In[26]:


plt.plot(K, distance_of_points_from_line)

