'''
    Bruno Gabriel de Sousa, Alexandre Zeni,
    Lorhan Felipe Melo, Leonardo Oliani Fernandes,
    Joshua Patrick Loesch Alves
'''

from knn import knn
from tools import acuracy, scatter_plot, get_feature_df, remove_features_df
import scipy.io as scipy

mat = scipy.loadmat('datasets\\grupoDados1.mat')
 
X_train = mat['grupoTrain']
X_test = mat['grupoTest']
y_train = mat['trainRots']
y_test = mat['testRots']

predicted_classes = knn(X_train, y_train, X_test, 10)
print(f'Acuracia inicial {acuracy(predicted_classes, y_test)}')

X_train_sepal = remove_features_df(X_train, [2,3])
X_test_sepal = remove_features_df(X_test, [2,3])

X_graph = [get_feature_df(X_train_sepal, 0), get_feature_df(X_train_sepal, 1)]
scatter_plot(X_graph, y_test)

for x in range(1, 10):
    predicted_classes = knn(X_train_sepal, y_train, X_test_sepal, x)
    print(f'Acuracia apenas com features sepal com k = {x} {acuracy(predicted_classes, y_test)}')

X_train_petal = remove_features_df(X_train, [0,1])
X_test_petal = remove_features_df(X_test, [0,1])

X_graph = [get_feature_df(X_train_petal, 0), get_feature_df(X_train_petal, 1)]
scatter_plot(X_graph, y_test)

for x in range(1, 10):
    predicted_classes = knn(X_train_petal, y_train, X_test_petal, x)
    print(f'Acuracia apenas com features petal com k = {x} {acuracy(predicted_classes, y_test)}')
