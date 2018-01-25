"""
This is a quick example coding an adverserial attack on logstic regression / k-NN / Small MLP
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
# PARAMETERS
dataset_used = 'small' # small or large
# dataset_used = 'large' # small or large
algorithm = 'knn'

if dataset_used == 'small':
    digits = datasets.load_digits()
    X_digits = digits.data

    # scaler = MinMaxScaler(feature_range=(0, 1))
    X_digits_shape = X_digits.shape
    scaler = StandardScaler()
    X_digits = scaler.fit_transform(X_digits.reshape(-1, 1)).reshape(X_digits_shape)
    # print ("np.min(X_digits) = {}, np.max(X_digits) = {}".format(np.min(X_digits), np.max(X_digits)))

    y_digits = digits.target
    n_samples = len(X_digits)
    X_train = X_digits[:int(.9 * n_samples)]
    y_train = y_digits[:int(.9 * n_samples)]
    X_test = X_digits[int(.9 * n_samples):]
    y_test = y_digits[int(.9 * n_samples):]


elif dataset_used == 'large':
    mnist = fetch_mldata('MNIST original', data_home="./")
    X_train, y_train = mnist.data[:60000] / 255., mnist.target[:60000]
    X_test, y_test = mnist.data[60000:] / 255., mnist.target[60000:]

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
# plt.figure()
# plt.imshow(mnist.data[20000].reshape((28, 28)))
# plt.show()
# plt.close()
print ("Train data: {}, Train labels: {}\nTest data: {}, Test Labels: {}".format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))
# exit()
if algorithm == 'randomforest':
    # clf = RandomForestClassifier(n_estimators=20, n_jobs=-1)
    clf = RandomForestClassifier(n_estimators=30, n_jobs=-1)

elif algorithm == 'knn':
    clf = neighbors.KNeighborsClassifier(n_jobs=-1)

elif algorithm == 'lr':
    clf = linear_model.LogisticRegression()

elif algorithm == 'dt':
    clf = DecisionTreeClassifier()

elif algorithm == 'mlp':
    clf = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
    solver='sgd', verbose=10, tol=1e-4, random_state=1, learning_rate_init=.1)

clf.fit(X_train, y_train)

print('{} score train: {}'.format(algorithm, clf.score(X_train, y_train)))
print('{} score test: {}'.format(algorithm, clf.score(X_test, y_test)))

if algorithm == 'knn':
    dist_test, indices_test = clf.kneighbors(X=X_test, n_neighbors=5, return_distance=True)

# print ("dist_test: ", np.mean(dist_test))
# print (clf.predict_proba(X_test[0].reshape(1, -1))[0])
# class permuation_ga:
#     def __init__(self):
#         pass
#     def

exit()

class adverserial_example:
    def __init__(self, dim=28*28, target=1, quantization=True, min_limit=0, max_limit=1):
        self.dim = dim
        self.target = target
        self.quantization = quantization
        self.min_limit = min_limit
        self.max_limit = max_limit

    def fitness(self, x, print_extra=False):
        global clf, algorithm
        if self.quantization:
            global quantization_bins
            x_ind = np.digitize(x, quantization_bins, right=False)
            new_x = quantization_bins[x_ind]
            predict_prob = clf.predict_proba(new_x.reshape((1, -1)))
        else:
            predict_prob = clf.predict_proba(x.reshape((1, -1)))
        loss = - predict_prob[0][self.target]
        # New, for knn only
        # dist_adv, indices_adv = clf.kneighbors(X=[x], n_neighbors=5, return_distance=True)

        if print_extra:
            print ("Loss: {}, Prediction Vector: {}".format(loss, predict_prob[0]))
            # print ("Loss: {}, Distance: {}".format(loss, np.mean(dist_adv)))
        # loss = loss + (0.1 * np.mean(dist_adv))
        return [loss]
        # return [np.mean(dist_adv)]

    def get_bounds(self):
        return ([self.min_limit]*self.dim, [self.max_limit]*self.dim)

    def evaluation(self, x):
        """
        Works only for KNN method
        """
        global algorithm, clf
        assert (algorithm == 'knn')
        dist_adv, indices_adv = clf.kneighbors(X=[x], n_neighbors=5, return_distance=True)
        return dist_adv, indices_adv

if __name__ == "__main__":
    quantization = True
    for iteration in range(1):
        f, axarr = plt.subplots(2, 5, figsize=(13,5))
        axarr = axarr.reshape((-1))
        for target in range(10):
            print ("#####################################################################################")
            print ("TARGET: ", target)
            print ("#####################################################################################")
            prob_instance = adverserial_example(dim=X_train.shape[1], target=target, quantization=quantization, min_limit=min_limit, max_limit=max_limit)
            prob = pg.problem(prob_instance)
            algo = pg.algorithm(pg.sea(gen = 5000))
            # algo = pg.algorithm(pg.sga(gen = 1000))
            pop = pg.population(prob,30)
            pop = algo.evolve(pop)
            print ("Best: {}".format(pop.champion_f[0]))
            prob_instance.fitness(pop.champion_x, print_extra=True)
            # print (prob_instance.evaluation(pop.champion_x))

            axarr[target].imshow(pop.champion_x.reshape((int(np.sqrt(X_train.shape[1])), int(np.sqrt(X_train.shape[1])))))
            axarr[target].set_title("Target: {}, Prob: {}".format(target, str(-np.round(pop.champion_f[0], 2))))
            # axarr[target].set_title("Target: {}".format(target))
            axarr[target].set_xticks([])
            axarr[target].set_yticks([])
        plt.savefig("{} - Data: {} - Quant: {} - Iteration: {}".format(algorithm.upper(), dataset_used, quantization, iteration))
        plt.close()
        # plt.show()
    # f.show()
