'''
    Bruno Gabriel de Sousa, Alexandre Zeni,
    Lorhan Felipe Melo, Leonardo Oliani Fernandes,
    Joshua Patrick Loesch Alves
'''

from knn import knn
from tools import accuracy, normalization
import scipy.io as scipy

mat = scipy.loadmat('datasets\\grupoDados2.mat')
 
X_train = mat['grupoTrain']
X_test = mat['grupoTest']
y_train = mat['trainRots']
y_test = mat['testRots']

for x in range(1, 10):
    predicted = knn(X_train, y_train, X_test, x)
    print(f'A acuracia passando todas as variáveis e k = {x} é {accuracy(predicted, y_test)}')

print('Aplicando pre-processing')

X_train = normalization(X_train)
X_test = normalization(X_test)

for x in range(1, 10):
    predicted = knn(X_train, y_train, X_test, x)
    print(f'A acuracia passando todas as variáveis normalizadas e k = {x} é {accuracy(predicted, y_test)}')

'''
Resultados:

Realizando a previsão sem pré processamento dos dados, o máximo atingido foi 78% com k igual a 8.

Para o pré processamento, foi escolhido a normalização devido a escala das variáveis serem muito diferntes
uma das outras. Após a normalização, atingimos 98% de acurácia com vários k's.

Foram testados k's de 1 a 9 em todos os cenários.


'''