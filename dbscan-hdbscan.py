from scipy.io import arff
from sklearn import cluster
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import time
import csv
import numpy as np
import hdbscan
from sklearn.neighbors import NearestNeighbors

def calculate_score(data, labels):
    if len(set(labels)) <= 1:
        return 0
    return silhouette_score(data, labels)

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

def find_hdbscan(data, min_samples):
    model = hdbscan.HDBSCAN(
        min_samples=min_samples
    )
    model = model.fit(data)
    n_clusters = len(np.unique(model.labels_))
    return model.labels_, n_clusters, model.cluster_selection_epsilon

def calculate_hdbscan(data, eps, min_samples):
    initial_time = time.time()
    best_labels = (0, [0 for _ in data])
    best_score = None
    best_eps = None
    best_min_samples = None
    best_n_clusters = None

    for m in min_samples:
        labels, n_clust, eps = find_hdbscan(data, m)
        score = calculate_score(data, labels)
        print(n_clust, score)
        if not best_score or score > best_score:
            best_score = score
            best_eps = eps
            best_min_samples = m
            best_n_clusters = n_clust
            best_labels = labels

    execution_time = time.time() - initial_time
    return best_labels, best_score, best_eps, best_min_samples, best_n_clusters, execution_time

def get_nn_max_distance(data):
    # Distances k plus proches voisins
    # Donnees dans X
    k = 4
    neigh = NearestNeighbors(n_neighbors = k)
    neigh.fit(data)
    distances,indices = neigh.kneighbors(data)
    # retirer le point " origine "
    newDistances = np.asarray([np.average(distances[i][1:]) for i in range (0, distances.shape[0])])
    return max(newDistances)

if __name__ == '__main__':
    FILES = [
        "./artificial/smile1.arff",
    ]
    with open("./results/dbscan-hdbscan.csv", 'w') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(("file", "method", "score", "best eps", "best min samples", "n_clusters", "execution time", "varying eps", "varying min samples"))
        for file in FILES:
            print(f"running for file {file}")
            raw_data = arff.loadarff(open(file, 'r'))
            data = [[x[0], x[1]] for x in raw_data[0]]
            f0 = [f[0] for f in data]
            f1 = [f[1] for f in data]
            real_labels = [x[2] for x in raw_data[0]]
            max_dist = get_nn_max_distance(data)
            eps_samples = 10
            varying_eps = [max_dist*i/eps_samples for i in range(1,2*eps_samples)]
            min_samples_samples = 10
            varying_min_samples = [2**i for i in range(0, 5)]
            labels, score, best_eps, best_min_samples, n_cluster, execution_time = calculate_dbscan(data, varying_eps, varying_min_samples)
            csv_out.writerow((file.split("/")[-1], "dbscan varying eps and min samples", score, best_eps, best_min_samples, n_cluster, execution_time, varying_eps, varying_min_samples))
            labels, score, best_eps, best_min_samples, n_cluster, execution_time = calculate_hdbscan(data, varying_eps, varying_min_samples)
            csv_out.writerow((file.split("/")[-1], "hdbscan varying min samples", score, best_eps, best_min_samples, n_cluster, execution_time, '-', varying_min_samples))
            plt.scatter(f0, f1, c=labels, s=3)
            plt.show()
