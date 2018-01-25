"""
The idea here is to project everything into a lower dimension, in order to:
- Have a better visualization
- Create a new manifold where I can make an adversarial attack
So far, PCA and KernelPCA (tried different kernels) have failed to give good results.

Next, tSNE (as a post-processing)
"""

from sklearn.datasets import fetch_mldata
from sklearn import neighbors, linear_model
from sklearn.neural_network import MLPClassifier
import pygmo as pg
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from sklearn import datasets, neighbors, linear_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA, KernelPCA

# PARAMETERS
dataset_used = 'small' # small or large
algorithm = 'knn'

"""
Load and process the dataset
"""
digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target

"""
Apply PCA
"""
# pca = KernelPCA(n_components=2, kernel="rbf", gamma=1/1500)
# pca.fit(X_digits)
# X_digits = pca.transform(X_digits)
# print ("X_digits.shape: ", X_digits.shape)
# print("pca.explained_variance_ratio_: ", pca.explained_variance_ratio_)

# scaler = MinMaxScaler(feature_range=(0, 1))
X_digits_shape = X_digits.shape
scaler = StandardScaler()
X_digits = scaler.fit_transform(X_digits.reshape(-1, 1)).reshape(X_digits_shape)

n_samples = len(X_digits)
X_train = X_digits[:int(.9 * n_samples)]
y_train = y_digits[:int(.9 * n_samples)]
X_test = X_digits[int(.9 * n_samples):]
y_test = y_digits[int(.9 * n_samples):]


counter_x = Counter(X_train.flatten().tolist())
counter_y = Counter(y_train.flatten().tolist())
min_limit = np.min(X_train)
max_limit = np.max(X_train)
print ("np.min(X_train) = {}, np.max(X_train) = {}".format(min_limit, max_limit))
print ("Unique values in X: {}".format(len(counter_x.keys())))
print ("Unique values in Y: {}".format(len(counter_y.keys())))
quantization_bins = list(counter_x.keys())
quantization_bins.sort()
quantization_bins = np.array(quantization_bins)
print ("Train data: {}, Train labels: {}\nTest data: {}, Test Labels: {}".format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))

"""
Load and fit the classifier
"""
clf = neighbors.KNeighborsClassifier(n_jobs=-1)
clf.fit(X_train, y_train)

print('{} score train: {}'.format(algorithm, clf.score(X_train, y_train)))
print('{} score test: {}'.format(algorithm, clf.score(X_test, y_test)))

class adverserial_example:
    def __init__(self, dim=28*28, target=1, quantization=True, min_limit=0, max_limit=1):
        self.dim = dim
        self.target = target
        self.quantization = quantization
        self.min_limit = min_limit
        self.max_limit = max_limit

    def fitness(self, x, print_extra=False):
        global clf
        if self.quantization:
            global quantization_bins
            x_ind = np.digitize(x, quantization_bins, right=False)
            new_x = quantization_bins[x_ind]
            predict_prob = clf.predict_proba(new_x.reshape((1, -1)))
        else:
            predict_prob = clf.predict_proba(x.reshape((1, -1)))
        loss = - predict_prob[0][self.target]
        if print_extra:
            print (predict_prob[0])
        return [loss]

    def get_bounds(self):
        return ([self.min_limit]*self.dim, [self.max_limit]*self.dim)

if __name__ == "__main__":
    quantization = True
    for iteration in range(1):
        f, axarr = plt.subplots(2, 5, figsize=(13,5))
        axarr = axarr.reshape((-1))
        for target in range(10):
            prob = pg.problem(adverserial_example(dim=X_train.shape[1], target=target, quantization=quantization, min_limit=min_limit, max_limit=max_limit))
            algo = pg.algorithm(pg.sea(gen = 1))
            # algo = pg.algorithm(pg.sga(gen = 200))
            pop = pg.population(prob,2)
            pop = algo.evolve(pop)

            axarr[target].imshow(pop.champion_x.reshape((int(np.sqrt(X_train.shape[1])), int(np.sqrt(X_train.shape[1])))))
            axarr[target].set_title("Target: {}, Prob: {}".format(target, str(-np.round(pop.champion_f[0], 2))))
            axarr[target].set_xticks([])
            axarr[target].set_yticks([])
        plt.savefig("{} - Data: {} - Quant: {} - Iteration: {}".format(algorithm.upper(), dataset_used, quantization, iteration))
        plt.close()
