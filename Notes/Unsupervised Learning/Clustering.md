## Introduction

Clustering is an unsupervised machine learning technique. It is used to place the data elements into related groups without any prior knowledge of the group definitions

### Types of Segmentation

3 types of segmentation are used for customer segmentation:

* **Behavioural segmentation** : Segmentation is based on the actual patterns displayed by the consumer
* **Attitudinal segmentation** : Segmentation is based on the beliefs or the intents of people, which may not translate into similar action
* **Demographic segmentation** : Segmentation is based on the person’s profile and uses information such as age, gender, residence locality, income, etc.

### Euclidean Distance

The method in which any clustering algorithm goes about doing that is through the method of finding something called a “ **distance measure** ”. The distance measure that is used in K-means clustering is called the **Euclidean Distance measure**

As mentioned in the video above, the Euclidean Distance between the 2 points is measured as follows: If there are 2 points X and Y having n dimensions

**X=(X1,X2,X3,...Xn)**

**Y=(Y1,Y2,Y3,....Yn)**

Then the Euclidean Distance D is given as

**D = sqrt( (X1−Y1)^2 + (X2−Y2)^2 + ... (Xn−Yn)^2  )**

### Steps of KNN Algorithm

1. Start by choosing k initial centoids randomly
2. Assign all closest point to each centoid to form a cluster
3. Find an optimized centoid for all the points in a cluster
4. Keep repeating 2 and 3 till centoids don't change



The equation for the assignment step is as follows:

**Zi = argmin||Xi−μk||^2**


The equation for optimisation is as follows:

**μk = 1/nk ∑i:zi = k^Xi**


### K-Means++ algorithm

In K-Means++ algorithm,

1. We choose one center as one of the data points at random.
2. For each data point **Xi**, We compute the distance between **Xi** and the nearest **center** that had already been chosen.
3. Now, we choose the next cluster center using the weighted probability distribution where a point X is chosen with probability proportional to d(X)^2 .
4. Repeat Steps 2 and 3 until **K** **centers** have been chosen.



### Major practical considerations involved in K-Means clustering are:

* The number of clusters that you want to divide your data points into, i.e. the value of K has to be pre-determined.
* The choice of the initial cluster centres can have an impact on the final cluster formation.
* The clustering process is very sensitive to the presence of outliers in the data.
* Since the distance metric used in the clustering process is the Euclidean distance, you need to bring all your attributes on the same scale. This can be achieved through standardisation.
* The K-Means algorithm does not work with categorical data.
* The process may not converge in the given number of iterations. You should always check for convergence.


### Choosing right number of clusters

Silhouette analysis or silhouette coefficient is a measure of how similar a data point is to its own cluster (cohesion) compared to other clusters (separation).

So to compute silhouette metric, we need to compute two measures i.e. **a**(**i**) and **b**(**i**) where,

* **a**(**i**) is the average distance from own cluster(Cohesion). - As small as possible
* **b**(**i**) is the average distance from the nearest neighbour cluster(Separation). - As large as possible

We can also choose the right number of clusters using Elbow Curve or SSD (Sum of squared distances)- Find an example in Jupiter notebook

- Basically it is the distance between points and centoids
- Idea is as you keep adding more centoids, the distance between the points and it's respective centoid keeps decreasing and so will it's SSD
- But after certain number of centoids you will not see any significant value in adding centoids as the SSD will not decrease significantly

## K-Means in Python

### Data Preparation Steps

1. Importing the data
2. Removing missing values
3. Aggregating on customer ID to get the required RFM value (R- Recency, F- Frequency, M-Monetory)
4. Outlier treatment

## Heirarchical Clustering Algorithm

The output of the hierarchical clustering algorithm is quite different from the K-mean algorithm as well. It results in an inverted tree-shaped structure, called the dendrogram.

In Heirarchical Clustering Algoritm, you don’t have to specify the number of clusters beforehand.

Given a set of N items to be clustered, the steps in hierarchical clustering are:

1. Calculate the NxN distance (similarity) matrix, which calculates the distance of each data point from the other
2. Each item is first assigned to its own cluster, i.e. N clusters are formed
3. The clusters which are closest to each other are merged to form a single cluster
4. The same step of computing the distance and merging the closest clusters is repeated till all the points become part of a single cluster

Thus, what you have at the end is the dendrogram, which shows you which data points group together in which cluster at what distance.

Hierarchical clustering can proceed in 2 ways — **agglomerative** and divisive. 

* If you start with n distinct clusters and iteratively reach to a point where you have only 1 cluster in the end, it is called **agglomerative clustering**.
* On the other hand, if you start with 1 big cluster and subsequently keep on partitioning this cluster to reach n clusters, each containing 1 element, it is called divisive clustering.

### Linkage

Pairwise distances between the data points as the representative of the distance between 2 clusters. This measure of the distance is called single **linkage**.

The different types of linkages are:

* **Single Linkage**: Here, the distance between 2 clusters is defined as the shortest distance between points in the two clusters
* **Complete Linkage:** Here, the distance between 2 clusters is defined as the maximum distance between any 2 points in the clusters
* **Average Linkage**: Here, the distance between 2 clusters is defined as the average distance between every point of one cluster to every other point of the other cluster


### Disadvantages

Major disadvantage of Heirarchical clustering is that, since you compute the distance of each point from every other point, it is time-consuming and needs a lot of processing power.


## Industry application of K-Mean and Heirarchical clustering

 [Upgrad industry insights](https://learn.upgrad.com/course/5800/segment/54463/324128/981340/4902373)

Can you use the dendrogram to make meaningful clusters? 

Yes. It is a great tool. You can look at what stage an element is joining a cluster and hence see how similar or dissimilar it is to the rest of the cluster. If it joins at the higher height, it is quite different from the rest of the group. You can also see which elements are joining which cluster at what stage and can thus use business understanding to cut the dendrogram more accurately.
