from scipy.io import arff
from sklearn import cluster, metrics
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances
import kmedoids
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import time

path = './artificial/'
databrut = arff.loadarff(open(path+"xclara.arff", 'r'))
data = [[x[0], x[1]] for x in databrut[0]]
f0 = [f[0] for f in data]
f1 = [f[1] for f in data]

def cluster_kmeans(data):
    k = 3
    model = cluster.KMeans(n_clusters = k, init = 'k-means++')
    model.fit(data)
    return (model.n_iter_, model.labels_)

def cluster_kmedoids(data):
    k = 3
    distmatrix = euclidean_distances(data)
    fp = kmedoids.fasterpam(distmatrix, k)
    return (fp.n_iter, fp.labels)

tps1 = time.time()
# iteractions, labels = [[],[0 for _ in data]]
# iteractions, labels = cluster_kmeans(data)
# iteractions, labels = cluster_kmedoids(data)
tps2 = time.time()

plt.scatter(f0, f1, c=labels, s=8)
plt.show()
