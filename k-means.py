import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
import random as rd


def k_means(dataset, k_value):
    print("Performing K-Means on the data...")
    X = dataset.iloc[:, [1, 2, 3, 4]].values
    m = X.shape[0]  # number of training examples
    n = X.shape[1]  # number of features. Here n=4
    n_iter = 100
    # print(X)
    K = k_value  # number of clusters
    centroids = np.array([]).reshape(n, 0)
    for i in range(K):
        rand = rd.randint(0, m - 1)
        centroids = np.c_[centroids, X[rand]]
    Output = {}
    for i in range(n_iter):
        # step 2.a
        euclidianDistance = np.array([]).reshape(m, 0)
        for k in range(K):
            tempDist = np.sum((X - centroids[:, k]) ** 4, axis=1)
            euclidianDistance = np.c_[euclidianDistance, tempDist]
        C = np.argmin(euclidianDistance, axis=1) + 1
        # step 2.b
        Y = {}
        for k in range(K):
            Y[k + 1] = np.array([]).reshape(4, 0)
        for i in range(m):
            Y[C[i]] = np.c_[Y[C[i]], X[i]]

        for k in range(K):
            Y[k + 1] = Y[k + 1].T

        for k in range(K):
            centroids[:, k] = np.mean(Y[k + 1], axis=0)
        Output = Y
    # Output = Y
    print("clusters formed...")
    # print(Output)
    # for c in centroids:
    #     print(c)
    color = ['red', 'blue', 'green']
    labels = ['cluster1', 'cluster2', 'cluster3']
    for k in range(K):
        plt.scatter(Output[k + 1][:, 0], Output[k + 1][:, 1], c=color[k], label=labels[k])
    plt.scatter(centroids[0, :], centroids[1, :], s=300, c='yellow', label='Centroids')
    plt.title('Clusters found on K-means \n Close this window to find the accuracy')

    plt.xlabel('x-axis: SepalLengthCm')
    plt.ylabel('y-axis: SepalWidthCm')
    plt.legend()
    plt.show()
    # print(C)
    return Output, centroids, C


def find_accuracy(output, list_data, C):
    count = 0
    for i in range(0, 149):
        if (list_data[i][5] == C[i]):
            count = count + 1

    # print("in accuracy")
    # for x in list_data:
    #     print(x[5])
    # print(C)
    print("The accuracy is : ")
    print((count/150)*100)



if __name__ == '__main__':

    #importing the Iris dataset with pandas
    dataset = pd.read_csv('Iris.csv')
    x = dataset.iloc[:, [1, 2, 3, 4]].values
    # Finding the optimum number of clusters for k-means classification
    list_data = dataset.values.tolist()
    # print(x)
    for l in list_data:
        # x.insert(0, 1)
        # print(x)
        # print(x[4])
        if l[5] == 'Iris-setosa':
            l[5] = 1
        elif l[5] == 'Iris-versicolor':
            l[5] = 2
        elif l[5] == 'Iris-virginica':
            l[5] = 3

    wcss = []
    print("Applying K-means using the number of clusters between 1 to 10...")

    for i in range(1, 10):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(x)
        wcss.append(kmeans.inertia_)

    # Plotting the results onto a line graph, allowing us to observe 'The elbow'
    plt.plot(range(1, 10), wcss)
    plt.title('The elbow method \n Close this window to continue on K-means')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')  # within cluster sum of squares
    plt.show()
    print("Using the elbow method we can now pick the optimum amount of clusters for classification, which is 3... ")
    print("Applying K-means algorithm using the number of clusters as 3...")
    Output, centroids, C = k_means(dataset, 3)
    accuracy = find_accuracy(Output, list_data, C)

