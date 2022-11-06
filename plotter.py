from scipy.io import arff
from sklearn import cluster
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import time
import csv

if __name__ == '__main__':
    file = "./artificial/spherical_6_2.arff"
    print(f"running for file {file}")
    raw_data = arff.loadarff(open(file, 'r'))
    data = [[x[0], x[1]] for x in raw_data[0]]
    f0 = [f[0] for f in data]
    f1 = [f[1] for f in data]
    real_labels = [x[2] for x in raw_data[0]]
    plt.scatter(f0, f1, c=real_labels, s=3)
    plt.show()
