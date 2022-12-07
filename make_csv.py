import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


dataset = fetch_california_housing()
feat_train, feat_test, target_train, target_test = \
    train_test_split(dataset.data, dataset.target, test_size=0.1)
target_train = target_train[..., np.newaxis]
target_test = target_test[..., np.newaxis]

train = np.concatenate((feat_train, target_train), axis=-1)
np.savetxt('train.csv', train, fmt='%.5f', delimiter=',')
test = np.concatenate((feat_test, target_test), axis=-1)
np.savetxt('test.csv', test, fmt='%.5f', delimiter=',')