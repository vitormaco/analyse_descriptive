from scipy.io import arff
from sklearn import cluster
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import time
import csv

def calculate_score(data, labels):
    if len(set(labels)) <= 1:
        return 0
    return silhouette_score(data, labels)

def find_agglomerative(data, distance_threshold, linkage, n_clusters):
    model = cluster.AgglomerativeClustering(
        linkage = linkage,
        distance_threshold = distance_threshold,
        n_clusters = n_clusters
    )
    model = model.fit(data)
    return model.labels_, model.n_clusters_

def find_best_n_clusters(data, distance_thresholds, linkages, n_clusters):
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

if __name__ == '__main__':
    FILES = [
        "./artificial/2d-3c-no123.arff",
        "./artificial/2d-4c.arff",
        "./artificial/2d-10c.arff",
        "./artificial/2d-20c-no0.arff",
    ]
    with open("./results/agglomerative.csv", 'w') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(("file", "method", "score", "best distance", "best linkage", "n_clusters", "execution time", "varying parameter"))
        for file in FILES:
            print(f"running for file {file}")
            raw_data = arff.loadarff(open(file, 'r'))
            data = [[x[0], x[1]] for x in raw_data[0]]
            f0 = [f[0] for f in data]
            f1 = [f[1] for f in data]
            real_labels = [x[2] for x in raw_data[0]]
            corners_dist = ((max(f0)-min(f0))**2 + (max(f1)-min(f1))**2)**0.5
            varying_distances = [corners_dist*(2**x) for x in range(-8,0)]
            varying_linkages = ['ward', 'complete', 'average', 'single']
            varying_n_clusters = list(range(2,round(len(data)**0.5)))
            labels, score, best_distance, best_linkage, n_cluster, execution_time = find_best_n_clusters(data, varying_distances, ['single'], [None])
            csv_out.writerow((file.split("/")[-1], "agglomerative varying distance", score, best_distance, best_linkage, n_cluster, execution_time, varying_distances))
            labels, score, best_distance, best_linkage, n_cluster, execution_time = find_best_n_clusters(data, [best_distance], varying_linkages, [None])
            csv_out.writerow((file.split("/")[-1], "agglomerative varying linkage", score, best_distance, best_linkage, n_cluster, execution_time, varying_linkages))
            labels, score, best_distance, best_linkage, n_cluster, execution_time = find_best_n_clusters(data, [None], [best_linkage], varying_n_clusters)
            csv_out.writerow((file.split("/")[-1], "agglomerative varying n clusters", score, best_distance, best_linkage, n_cluster, execution_time, varying_n_clusters))
