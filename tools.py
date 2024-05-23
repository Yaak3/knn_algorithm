import math
import matplotlib.pyplot as plt

def euclidian_distance(X_train, X_test):
    distance = 0

    for index, _ in enumerate(X_train):
      distance += (X_train[index] - X_test[index])**2

    return math.sqrt(distance)

def accuracy(predicted_values, y_test):
   correct_predictions = predicted_values == y_test
   sum_correct_predicionts = sum(correct_predictions)

   return sum_correct_predicionts / len(y_test)

#TODO desenvolver normalização e padronização dos dados nos metodos abaixo
def normalization(df):
   pass

def padronization(df):
   pass

#TODO desenvolver o plot de scatter plot
def scatter_plot(x_train, y_train, x_test, predicted_values):
   fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 5))
