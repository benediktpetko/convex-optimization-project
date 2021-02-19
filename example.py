import numpy as np
import ot
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score


X, y = load_digits(return_X_y=True)
#Uncomment below to load MNIST and comment out the above line
#X, y = fetch_openml('mnist_784', return_X_y=True, as_frame=False)

# normalize pixel intensities
X = normalize(X, axis=1, norm='l1')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# pixel coordinate array
p = np.mgrid[0:8:1, 0:8:1].reshape(2,-1).T
#Uncomment below to use bigger cost matrix for MNIST and comment out the above line
#p = np.mgrid[0:64:1, 0:64:1].reshape(2,-1).T

# cost matrix with Euclidean base metric
M = ot.dist(p,  metric='euclidean')
