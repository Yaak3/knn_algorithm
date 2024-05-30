import math
import matplotlib.pyplot as plt
import statistics
import itertools

symbols = ['^', '+', '.', '-', '*', '/', '~']
colors = ['red', 'green', 'blue', 'yellow', 'orange', 'pink', 'cyano']


def transform_list(to_transform):
   return list(itertools.chain(*to_transform))

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
   return apply_transformation(df, 'normalization')

def padronization(df):
   return apply_transformation(df, 'padronization')

def apply_transformation(df, type):
   number_of_features = len(df[0])
   number_of_values = len(df)
   feature_transformed = []

   df_transformed = create_empty_dataframe(number_of_values)
   
   for num_feature in range(number_of_features):
      feature = get_feature_df(df, num_feature)
      
      if(type == 'normalization'):
         for value in feature:
            feature_transformed.append(
               (value - min(feature)) / (max(feature) - min(feature))
            )

         df_transformed = set_feature_df(feature_transformed, df_transformed)
      else:
         for value in feature:
            feature_transformed.append(
               (value - statistics.mean(feature)) / statistics.stdev(feature)
               )
      
         df_transformed = set_feature_df(feature_transformed, df_transformed)

      feature_transformed.clear()

   return df_transformed

def create_empty_dataframe(number_of_values):
   empty_df_transformed = []
   
   for row_num in range(number_of_values):
      empty_df_transformed.append([])

   return empty_df_transformed


def set_feature_df(feature_transformed, df_transformed):
   df_temp = df_transformed.copy()
   for index, _ in enumerate(df_temp):
      df_temp[index].append(feature_transformed[index])

   return df_temp

def get_multiple_features_df(data_frame, num_features):
   new_df = []
   temp_row = []

   for row in data_frame:
      for index, value in enumerate(row):
         if(index in num_features):
            temp_row.append(value)

      new_df.append(temp_row.copy())
      temp_row.clear()

   return new_df

def get_feature_df(data_frame, num_feature):
   feature = []

   for row in data_frame:
      feature.append(row[num_feature])   

   return feature

def get_data_with_lables(X, labels, label):
   x_graph_values = []
   y_graph_values = []
   for index, label_y in enumerate(labels):
      if(label_y == label):
         x_graph_values.append(X[0][index])
         y_graph_values.append(X[1][index])
   
   return x_graph_values, y_graph_values

#TODO desenvolver o plot de scatter plot
def scatter_plot(X, labels):
   fig, ax = plt.subplots(figsize=(18, 5))
   labels_transformed = transform_list(labels)
   label_values = set(labels_transformed)

   for index, label in enumerate(label_values):
      X_values, y_values = get_data_with_lables(X, labels, label)

      ax.scatter(X_values, y_values, c=colors[index] , marker=symbols[index])
   
   plt.show()
   