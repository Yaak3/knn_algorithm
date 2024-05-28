import math
import matplotlib.pyplot as plt
import statistics

def euclidian_distance(X_train, X_test):
    distance = 0

    for index, _ in enumerate(X_train):
      distance += (X_train[index] - X_test[index])**2

    return math.sqrt(distance)

def acuracy(predicted_values, y_test):
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

def get_feature_df(data_frame, num_feature):
   feature = []

   for row in data_frame:
      feature.append(row[num_feature])   

   return feature


#TODO desenvolver o plot de scatter plot
def scatter_plot(x_train, y_train, x_test, predicted_values):
   fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 5))
   