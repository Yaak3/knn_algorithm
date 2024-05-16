import scipy.io as scipy
from tools import euclidian_distance

mat = scipy.loadmat('grupoDados1.mat')
 
X_train = mat['grupoTrain']
X_test = mat['grupoTest']
y_train = mat['trainRots']
y_test = mat['testRots']

def knn(X_tain, y_train, X_test, k):
  len_data = len(X_train)
  result_train = {}

  for row_test in X_test:
    for index, row_train in enumerate(X_train):
      if(index not in result_train.keys()):
        result_train[index] = []

      distance = euclidian_distance(row_train, row_test)
      result_train[index].append(distance)

  return result_train