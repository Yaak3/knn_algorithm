from tools import euclidian_distance
from statistics import mode
import numpy as np

def knn(X_train, y_train, X_test, k):

  distances = {}
  predicted_class = []

  for row_test in X_test:
    for index, row_train in enumerate(X_train):
      if(index not in distances.keys()):
        distances[index] = []

      distance = euclidian_distance(row_train, row_test)
      distances[index].append(distance)

  values_feature_knn = []
  ordered_distances = []
  values_test_labels = []

  for row in range(len(X_test)):
    for col in distances.values():
      values_feature_knn.append(col[row])

    ordered_distances = values_feature_knn.copy()
    
    ordered_distances.sort()
    ordered_distances = ordered_distances[0:k]

    for distance in ordered_distances:
      index_value = values_feature_knn.index(distance)
      values_test_labels.append(y_train[index_value][0])

    predicted_class.append([mode(values_test_labels)])

    values_feature_knn.clear()
    values_test_labels.clear()
    ordered_distances.clear()

  return np.array(predicted_class)
