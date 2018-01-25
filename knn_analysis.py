import numpy as np
import matplotlib.pyplot as plt

dataset_test = []
dataset_adv = []
with open("knn_test.txt", 'r') as fileHandle:
    lines = fileHandle.read().splitlines()
    for line in lines:
        line_split = line.split(" ")
        if len(line_split) == 5:
            dataset_test.append(list(map(float, line_split)))

    dataset_test = np.array(dataset_test)
    print (dataset_test.shape)

with open("knn_adv.txt", 'r') as fileHandle:
    lines = fileHandle.read().splitlines()
    for line in lines:
        line_split = line.split(" ")
        if len(line_split) == 5:
            dataset_adv.append(list(map(float, line_split)))

    dataset_adv = np.array(dataset_adv)
    print (dataset_adv.shape)

dataset_test_mean = np.mean(dataset_test, axis=1)
dataset_adv_mean = np.mean(dataset_adv, axis=1)
plt.figure()
plt.ylabel("Distance")
plt.boxplot([dataset_test_mean, dataset_adv_mean])
plt.xticks([1, 2], ["Test examples", "Adversarial examples"])
plt.show()
plt.close()
