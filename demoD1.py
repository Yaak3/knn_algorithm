'''
    Bruno Gabriel de Sousa, Alexandre Zeni,
    Lorhan Felipe Melo, Leonardo Oliani Fernandes,
    Joshua Patrick Loesch Alves
'''

from knn import knn
from tools import accuracy, scatter_plot, get_feature_df, get_multiple_features_df
import scipy.io as scipy

mat = scipy.loadmat('datasets\\grupoDados1.mat')
 
X_train = mat['grupoTrain']
X_test = mat['grupoTest']
y_train = mat['trainRots']
y_test = mat['testRots']

predicted_classes = knn(X_train, y_train, X_test, 10)
print(f'Acuracia inicial {accuracy(predicted_classes, y_test)}')

X_train_sepal = get_multiple_features_df(X_train, [0,1])
X_test_sepal = get_multiple_features_df(X_test, [0,1])

X_graph = [get_feature_df(X_train_sepal, 0), get_feature_df(X_train_sepal, 1)]
scatter_plot(X_graph, y_test)

for x in range(1, 11):
    predicted_classes = knn(X_train_sepal, y_train, X_test_sepal, x)
    print(f'Acuracia apenas com features sepala com k = {x} {accuracy(predicted_classes, y_test)}')

X_train_petal = get_multiple_features_df(X_train, [2,3])
X_test_petal = get_multiple_features_df(X_test, [2,3])

X_graph = [get_feature_df(X_train_petal, 0), get_feature_df(X_train_petal, 1)]
scatter_plot(X_graph, y_test)

for x in range(1, 11):
    predicted_classes = knn(X_train_petal, y_train, X_test_petal, x)
    print(f'Acuracia apenas com features petala com k = {x} {accuracy(predicted_classes, y_test)}')

'''
Resultados:

Realizando o treinamento com todas as variáveis e com k= 10, chegamos a 94% de acurácia conforme esperado.

Realizando a divisão das features, encontramos a seguinte situação:

Apenas com os dados da sepala, a classificação não apresentou uma acurácia esperada, sendo bem abaixo dos 94% anteriores. 
Isso se dá pois os valores relacionados ao cm das sepalas não é suficiente para descrever qual tipo de flor iris.

Alterando o cenário para apenas os dados da petala, o cenário já muda. A acurácia fica maior, e as duas varáveis sozinhas
conseguem descrever bem melhor, chegando a 98% de acurácia com k igual a 9.

Foram testados k's de 1 a 9 no cenário pós pré processamento.
'''