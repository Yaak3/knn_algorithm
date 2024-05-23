'''
    Bruno Gabriel de Sousa, Alexandre Zeni,
    Lorhan Felipe Melo, Leonardo Oliani Fernandes,
    Joshua Patrick Loesch Alves
'''

from knn import knn
from tools import accuracy
import pandas as pd
import scipy.io as scipy

mat = scipy.loadmat('datasets\\grupoDados1.mat')
 
X_train = mat['grupoTrain']
X_test = mat['grupoTest']
y_train = mat['trainRots']
y_test = mat['testRots']

predicted_classes = knn(X_train, y_train, X_test, 10)

print(accuracy(predicted_classes, y_test))
