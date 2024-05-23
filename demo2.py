from knn import knn
from tools import acuracy
import pandas as pd
import scipy.io as scipy

mat = scipy.loadmat('datasets\\grupoDados1.mat')
 
X_train = mat['grupoTrain']
X_test = mat['grupoTest']
y_train = mat['trainRots']
y_test = mat['testRots']

print(X_train)