import numpy as np
import ot
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier
from ot.utils import clean_zeros
from utils import compute_sinkhorn_divergence, compute_emd


X, y = load_digits(return_X_y=True)

# normalize pixel intensities
X = normalize(X, axis=1, norm='l1')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# pixel coordinate array
p = np.mgrid[0:8:1, 0:8:1].reshape(2,-1).T

# cost matrix with Euclidean base metric
M = ot.dist(p,  metric='euclidean')


# example: Sinkhorn divergence of the first two images with lambda = 1e-3
print(compute_sinkhorn_divergence(X[0,:], X[1,:], M, reg=1e-2))

# example: compute EMD of the first two images
print(compute_emd(X[0,:], X[1,:], M))
