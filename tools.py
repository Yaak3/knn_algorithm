import math

def euclidian_distance(X_train, X_test):
    distance = 0

    for index, _ in enumerate(X_train):
      distance += (X_train[index] - X_test[index])**2

    return math.sqrt(distance)