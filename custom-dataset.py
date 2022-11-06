import csv
import time

import hdbscan
import kmedoids
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import cluster
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.neighbors import NearestNeighbors


def calculate_score(data, labels):
    if len(set(labels)) <= 1:
        return -1
    return silhouette_score(data, labels)

def k_means(data, n_clusters):
    model = cluster.KMeans(n_clusters = n_clusters, init = 'k-means++')
    model.fit(data)
    return model.labels_

def k_medoids_manhattan(data, n_clusters):
    distmatrix = manhattan_distances(data)
    fp = kmedoids.fasterpam(distmatrix, n_clusters)
    return fp.labels

def find_clusters(data, cluster_method_func):
    initial_time = time.time()
    best_labels = (0, [0 for _ in data])
    best_score = None
    best_n_clusters = None

    for n_clusters in range(5,round(len(data)**0.5)):
        labels = cluster_method_func(data, n_clusters)
        score = calculate_score(data, labels)
        if not best_score or score > best_score:
            best_score = score
            best_n_clusters = n_clusters
            best_labels = labels

    execution_time = time.time() - initial_time
    return best_labels, best_score, best_n_clusters, execution_time



def find_agglomerative(data, distance_threshold, linkage, n_clusters):
    model = cluster.AgglomerativeClustering(
        linkage = linkage,
        distance_threshold = distance_threshold,
        n_clusters = n_clusters
    )
    model = model.fit(data)
    return model.labels_, model.n_clusters_

def find_clusters_agglomerative(data, distance_thresholds, linkages, n_clusters):
    initial_time = time.time()
    best_labels = (0, [0 for _ in data])
    best_score = None
    best_distance = None
    best_linkage = None
    best_n_clusters = None

    for d in distance_thresholds:
        for l in linkages:
            for n in n_clusters:
                labels, n_clust = find_agglomerative(data, d, l, n)
                score = calculate_score(data, labels)
                if not best_score or score > best_score:
                    best_score = score
                    best_distance = d
                    best_linkage = l
                    best_n_clusters = n_clust
                    best_labels = labels
    execution_time = time.time() - initial_time
    return best_labels, best_score, best_distance, best_linkage, best_n_clusters, execution_time



def find_dbscan(data, eps, min_samples):
    model = cluster.DBSCAN(
        eps=eps,
        min_samples=min_samples,
    )
    model = model.fit(data)
    n_clusters = len(np.unique(model.labels_))
    return model.labels_, n_clusters

def calculate_dbscan(data, eps, min_samples):
    initial_time = time.time()
    best_labels = (0, [0 for _ in data])
    best_score = None
    best_eps = None
    best_min_samples = None
    best_n_clusters = None

    for e in eps:
        for m in min_samples:
            labels, n_clust = find_dbscan(data, e, m)
            score = calculate_score(data, labels)
            if not best_score or score > best_score:
                best_score = score
                best_eps = e
                best_min_samples = m
                best_n_clusters = n_clust
                best_labels = labels

    execution_time = time.time() - initial_time
    return best_labels, best_score, best_eps, best_min_samples, best_n_clusters, execution_time

def find_hdbscan(data):
    model = hdbscan.HDBSCAN(gen_min_span_tree=True)
    model = model.fit(data)
    # model.minimum_spanning_tree_.plot(edge_cmap='viridis',
    #                                   edge_alpha=0.6,
    #                                   node_size=20,
    #                                   edge_linewidth=2)
    # plt.show()
    n_clusters = len(np.unique(model.labels_))
    return model.labels_, n_clusters

def calculate_hdbscan(data):
    initial_time = time.time()
    best_labels = (0, [0 for _ in data])
    best_score = None
    best_n_clusters = None

    labels, n_clust = find_hdbscan(data)
    score = calculate_score(data, labels)
    if not best_score or score > best_score:
        best_score = score
        best_n_clusters = n_clust
        best_labels = labels

    execution_time = time.time() - initial_time
    return best_labels, best_score, best_n_clusters, execution_time

def get_nn_max_distance(data):
    # Distances k plus proches voisins
    # Donnees dans X
    k = 5
    neigh = NearestNeighbors(n_neighbors = k)
    neigh.fit(data)
    distances,indices = neigh.kneighbors(data)
    # retirer le point " origine "
    newDistances = np.asarray([np.average(distances[i][1:]) for i in range (0, distances.shape[0])])
    return max(newDistances)

if __name__ == '__main__':
    FILES = [
        # "./custom_dataset/x1.txt",
        # "./custom_dataset/x2.txt",
        # "./custom_dataset/x3.txt",
        # "./custom_dataset/x4.txt",
        # "./custom_dataset/y1.txt",
        # "./custom_dataset/zz1.txt",
        "./custom_dataset/zz2.txt",
    ]
    with open("./results/custom2.csv", 'w') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(("file", "method", "score", "n_clusters", "execution time", "params", "labels"))
        for file in FILES:
            print(f"running for file {file}")
            raw_data = pd.read_csv(file, sep=' ', skipinitialspace=True)
            data = raw_data.to_numpy()
            f0 = [f[0] for f in data]
            f1 = [f[1] for f in data]

            # k-means
            print("k-means")
            labels, score, n_cluster, execution_time = find_clusters(data, k_means)
            csv_out.writerow((file.split("/")[-1], "k_means", score, n_cluster, execution_time, '-', labels.tolist()))

            # k-medoids
            print("k-medoids")
            labels, score, n_cluster, execution_time = find_clusters(data, k_medoids_manhattan)
            csv_out.writerow((file.split("/")[-1], "k_medoids_manhattan", score, n_cluster, execution_time, '-', labels.tolist()))

            # agglomerative
            print("agglomerative")
            corners_dist = ((max(f0)-min(f0))**2 + (max(f1)-min(f1))**2)**0.5
            varying_distances = [corners_dist*(2**x) for x in range(-8,0)]
            varying_linkages = ['ward', 'complete', 'average', 'single']
            labels, score, best_distance, best_linkage, n_cluster, execution_time = find_clusters_agglomerative(data, varying_distances, varying_linkages, [None])
            csv_out.writerow((file.split("/")[-1], "agglomerative", score, n_cluster, execution_time, f'best_distance: {best_distance}; best_linkage: {best_linkage}', labels.tolist()))

            # dbscan
            print("dbscan")
            max_dist = get_nn_max_distance(data)
            eps_samples = 10
            varying_eps = [max_dist*i/eps_samples for i in range(1,2*eps_samples)]
            varying_min_samples = [2**i for i in range(2, 5)]
            labels, score, best_eps, best_min_samples, n_cluster, execution_time = calculate_dbscan(data, varying_eps, varying_min_samples)
            csv_out.writerow((file.split("/")[-1], "dbscan", score, n_cluster, execution_time, f'best_eps: {best_eps}; best_min_samples: {best_min_samples}', labels.tolist()))

            # hdbscan
            print("hdbscan")
            labels, score, n_cluster, execution_time = calculate_hdbscan(data)
            csv_out.writerow((file.split("/")[-1], "hdbscan", score, n_cluster, execution_time, '-', labels.tolist()))

            # plt.scatter(f0, f1, c=labels, s=3)
            # plt.show()
