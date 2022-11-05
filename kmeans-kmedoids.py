from scipy.io import arff
from sklearn import cluster
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
from sklearn.metrics import silhouette_score
from sklearn.metrics import rand_score, mutual_info_score
import kmedoids
import matplotlib.pyplot as plt
import time
import csv

def calculate_score(data, labels):
    return silhouette_score(data, labels)

def k_means(data, n_clusters):
    model = cluster.KMeans(n_clusters = n_clusters, init = 'k-means++')
    model.fit(data)
    return model.labels_

def k_medoids_manhattan(data, n_clusters):
    distmatrix = manhattan_distances(data)
    fp = kmedoids.fasterpam(distmatrix, n_clusters)
    return fp.labels

def k_medoids_euclidean(data, n_clusters):
    distmatrix = euclidean_distances(data)
    fp = kmedoids.fasterpam(distmatrix, n_clusters)
    return fp.labels

def find_clusters(data, cluster_method_func):
    initial_time = time.time()
    best_labels = (0, [0 for _ in data])
    best_score = None
    best_n_clusters = None

    for n_clusters in range(2,round(len(data)**0.5)):
        labels = cluster_method_func(data, n_clusters)
        score = calculate_score(data, labels)
        if not best_score or score > best_score:
            best_score = score
            best_n_clusters = n_clusters
            best_labels = labels

    execution_time = time.time() - initial_time
    return best_labels, best_score, best_n_clusters, execution_time


if __name__ == '__main__':
    FILES = [
        "./artificial/2d-3c-no123.arff",
        "./artificial/2d-4c.arff",
        "./artificial/2d-10c.arff",
        "./artificial/2d-20c-no0.arff",
    ]
    with open("./results/kmeans-kmedoids.csv", 'w') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(("file", "method", "score", "n_clusters", "execution time", "rand score", "mutual info score"))
        for file in FILES:
            print(f"running for file {file}")
            raw_data = arff.loadarff(open(file, 'r'))
            data = [[x[0], x[1]] for x in raw_data[0]]
            f0 = [f[0] for f in data]
            f1 = [f[1] for f in data]
            real_labels = [x[2] for x in raw_data[0]]
            labels, score, n_cluster, execution_time = find_clusters(data, k_means)
            rscore = rand_score(real_labels, labels)
            miscore = mutual_info_score(real_labels, labels)
            csv_out.writerow((file.split("/")[-1], "k_means", score, n_cluster, execution_time, rscore, miscore))
            labels, score, n_cluster, execution_time = find_clusters(data, k_medoids_manhattan)
            rscore = rand_score(real_labels, labels)
            miscore = mutual_info_score(real_labels, labels)
            csv_out.writerow((file.split("/")[-1], "k_medoids_manhattan", score, n_cluster, execution_time, rscore, miscore))
            labels, score, n_cluster, execution_time = find_clusters(data, k_medoids_euclidean)
            rscore = rand_score(real_labels, labels)
            miscore = mutual_info_score(real_labels, labels)
            csv_out.writerow((file.split("/")[-1], "k_medoids_euclidean", score, n_cluster, execution_time, rscore, miscore))
