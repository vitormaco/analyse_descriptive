from scipy.io import arff
from sklearn import cluster
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import rand_score, mutual_info_score
import kmedoids
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import time

path = './artificial/'
# databrut = arff.loadarff(open(path+"2d-4c.arff", 'r'))
# databrut = arff.loadarff(open(path+"2d-10c.arff", 'r'))
databrut = arff.loadarff(open(path+"xor.arff", 'r'))
data = [[x[0], x[1]] for x in databrut[0]]
f0 = [f[0] for f in data]
f1 = [f[1] for f in data]
labels = [x[2] for x in databrut[0]]

def calculate_score(data, labels):
    score = silhouette_score(data, labels)
    # score = -davies_bouldin_score(data, labels)
    # score = calinski_harabasz_score(data, labels)
    return score

def cluster_kmeans(data):
    best_model = (0, [0 for _ in data])
    best_score = None
    best_k = None
    for k in range(2,round(len(data)**0.5)):
        model = cluster.KMeans(n_clusters = k, init = 'k-means++')
        model.fit(data)
        score = calculate_score(data, model.labels_)
        if not best_score or score > best_score:
            best_score = score
            best_k = k
            best_model = (model.n_iter_, model.labels_)
    print("optimal number of clusters: " + str(best_k))
    return best_model

def cluster_kmedoids(data):
    best_model = (0, [0 for _ in data])
    best_score = None
    best_k = None
    for k in range(2,round(len(data)**0.5)):
        distmatrix = manhattan_distances(data)
        fp = kmedoids.fasterpam(distmatrix, k)
        score = calculate_score(data, fp.labels)
        if not best_score or score > best_score:
            best_score = score
            best_k = k
            best_model = (fp.n_iter, fp.labels)
    print("optimal number of clusters: " + str(best_k))
    return best_model

tps1 = time.time()
iteractions, labels1 = cluster_kmeans(data)
iteractions, labels2 = cluster_kmedoids(data)
print("rand_score kmeans: " + str(rand_score(labels, labels1)))
print("rand_score kmedoids: " + str(rand_score(labels, labels2)))
print("mutual_info_score kmeans: " + str(mutual_info_score(labels, labels1)))
print("mutual_info_score kmedoids: " + str(mutual_info_score(labels, labels2)))
tps2 = time.time()

print("execution time: " + str((tps2-tps1)*1000))

plt.scatter(f0, f1, c=labels, s=8)
plt.show()
