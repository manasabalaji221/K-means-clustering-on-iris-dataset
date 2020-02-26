# K-means-clustering-on-iris-dataset
This program has the following elements:

k_means(dataset, k_value):
This function performs k-means clustering of the dataset passed. The number of 
clusters is also passed to the function.

find_accuracy(output, list_data, C):
This function is used to find the accuracy of the k-means algorithm.
Steps followed in the code:
1. The dataset is taken in as a csv.
2. The dataset is passed to kmeans function and the Within Cluster Sum of 
Squares (WCSS) are computed for the result. 
3. The graph is plotted and the elbow is found. This is used to find which is the 
optimal number of clusters to be used for this particular dataset. 
4. Now, the number found using the above graph is 3.
5. We call the k_means function to find the 3 clusters.
6. The functions returns the output data along with the centroids. These are 
plotted on a graph.
7. Now, these values are compared with the actual values of the type of flower in 
the dataset to find the accuracy.
