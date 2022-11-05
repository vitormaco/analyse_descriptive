from scipy.io import arff
from sklearn import cluster
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import rand_score, mutual_info_score
import kmedoids
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import time
import csv
import numpy as np
import hdbscan
from sklearn.neighbors import NearestNeighbors

def find_cluster(
    data,
    cluster_method,
    score_method,
    mode,
    params,
):
    initial_time = time.time()
    best_labels = (0, [0 for _ in data])
    best_score = None
    best_n_clusters = None
    best_params = None

    if mode == "range":
        for n_clusters in range(2,round(len(data)**0.5)):
            labels = cluster_method(data, n_clusters)
            score = score_method(data, labels)
            if not best_score or score > best_score:
                best_score = score
                best_n_clusters = n_clusters
                best_labels = labels
        execution_time = time.time() - initial_time
        return best_labels, best_score, best_n_clusters, execution_time, best_params

    if mode == "discrete":
        for p in params:
            labels, n_clusters = cluster_method(data, p)
            score = score_method(data, labels)
            if not best_score or score > best_score:
                best_score = score
                best_n_clusters = n_clusters
                best_labels = labels
                best_params = p
        execution_time = time.time() - initial_time
        return best_labels, best_score, best_n_clusters, execution_time, best_params

def find_kmeans(data, n_clusters):
    model = cluster.KMeans(n_clusters = n_clusters, init = 'k-means++')
    model.fit(data)
    return model.labels_

def find_kmedoids(data, n_clusters):
    # distmatrix = euclidean_distances(data)
    distmatrix = manhattan_distances(data)
    fp = kmedoids.fasterpam(distmatrix, n_clusters)
    return fp.labels

def find_agglomerative(data, params):
    model = cluster.AgglomerativeClustering(
        distance_threshold = params[1],
        linkage = params[0],
        n_clusters = None
    )
    model = model.fit(data)
    return model.labels_, model.n_clusters_

def find_dbscan(data, params):
    model = cluster.DBSCAN(
        eps=params[0],
        min_samples=params[1]
    )
    model = model.fit(data)
    n_clusters = len(np.unique(labels))
    return model.labels_, n_clusters

def find_hdbscan(data, params):
    model = hdbscan.HDBSCAN(
        min_cluster_size=params[0],
        min_samples=params[1]
    )
    model = model.fit(data)
    n_clusters = len(np.unique(labels))
    return model.labels_, n_clusters

CLUSTER_METHODS = {
    "kmeans" : {
        "method": find_kmeans,
        "mode": "range",
        "params" : []
    },
    "kmetoids": {
        "method": find_kmedoids,
        "mode": "range",
        "params" : []
    },
    "agglomerative": {
        "method": find_agglomerative,
        "mode": "discrete",
        "params" : [[linkage, distance] for linkage in ["single", "ward", "complete", "average"] for distance in [5, 10, 20]]
    },
    "dbscan": {
        "method": find_dbscan,
        "mode": "discrete",
        "params" : [[eps, min_samples] for eps in [2, 5, 10] for min_samples in [2, 4, 6]]
    },
    "hdbscan": {
        "method": find_hdbscan,
        "mode": "discrete",
        "params" : [[eps, min_samples] for eps in [2, 5, 10] for min_samples in [2, 4, 6]]
    },
}

SCORE_METHODS = [
    ("SILHOUETTE", silhouette_score),
    ("DAVIES", lambda x,y : -davies_bouldin_score(x,y)),
    ("CALINSKI", calinski_harabasz_score),
]

FILES = [
    "./artificial/2d-4c.arff",
    "./artificial/2d-10c.arff",
]

def find_nearest_neighbors(data):
    # Distances k plus proches voisins
    # Donnees dans X
    k = 5
    neigh = NearestNeighbors(n_neighbors = k)
    neigh.fit(data)
    distances,indices = neigh.kneighbors(data)
    # retirer le point " origine "
    newDistances = np.asarray([np.average(distances[i][1:]) for i in range (0, distances.shape[0])])
    trie = np.sort(newDistances)
    plt.title(" Plus proches voisins (5) ")
    plt.plot(trie)
    plt.show()

with open("./results.csv", 'w') as out:
    csv_out = csv.writer(out)
    csv_out.writerow(("method", "score_method", "score", "ideal n clusters", "execution time", "best params"))
    for f in FILES:
        print("running for file " + f)
        databrut = arff.loadarff(open(f, 'r'))
        data = [[x[0], x[1]] for x in databrut[0]]
        f0 = [f[0] for f in data]
        f1 = [f[1] for f in data]
        labels = [x[2] for x in databrut[0]]
        # find_nearest_neighbors(data)
        for cluster_method_name, cluster_method in CLUSTER_METHODS.items():
            for score_method_name, score_method in SCORE_METHODS:
                print("running " + str(cluster_method_name) + " with score " + str(score_method_name) + " with params " + str(cluster_method["params"]))
                labels, score, n_cluster, execution_time, best_params = find_cluster(data, cluster_method["method"], score_method, cluster_method["mode"], cluster_method["params"])
                csv_out.writerow((cluster_method_name, score_method_name, score, n_cluster, execution_time, best_params))

                plt.scatter(f0, f1, c=labels, s=3)
                plt.show()
