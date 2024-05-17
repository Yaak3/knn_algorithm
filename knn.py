import scipy.io as scipy
from tools import euclidian_distance

mat = scipy.loadmat('datasets\\grupoDados1.mat')
 
X_train = mat['grupoTrain']
X_test = mat['grupoTest']
y_train = mat['trainRots']
y_test = mat['testRots']

def knn(X_train, y_train, X_test, k):
  distances = {}

  for row_test in X_test:
    for index, row_train in enumerate(X_train):
      if(index not in distances.keys()):
        distances[index] = []

      distance = euclidian_distance(row_train, row_test)
      distances[index].append(distance)

  values_feature_knn = []
  ordered_distances = []
  index_test_labels = []
  value = 0

  for row in range(len(X_test)):
    for col in distances.values():
      values_feature_knn.append(col[row])

    ordered_distances = values_feature_knn
    
    ordered_distances.sort()
    value = ordered_distances[k]
    index_value = values_feature_knn.index(value)

    index_test_labels.append(index_value)

    values_feature_knn = []

  return index_test_labels

print(knn(X_train, y_train, X_test, 3))