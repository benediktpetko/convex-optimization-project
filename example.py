import numpy as np
import ot
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier
from ot.utils import clean_zeros

from utils import compute_sinkhorn_divergence, compute_emd

X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

#X = X.reshape(70000, 28, 28) 
#sum of intensities should be normalized to 1
X = normalize(X, axis=1, norm='l1')

X_train, X_test, y_train, y_test

# pixel coordinate array
p = np.mgrid[0:28:1, 0:28:1].reshape(2,-1).T

# cost matrix with Euclidean base metric
M = ot.dist(p,  metric='euclidean')

