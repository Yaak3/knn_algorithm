'''
    Bruno Gabriel de Sousa, Alexandre Zeni,
    Lorhan Felipe Melo, Leonardo Oliani Fernandes,
    Joshua Patrick Loesch Alves
'''
from knn import knn
from tools import accuracy, padronization
import scipy.io as scipy

mat = scipy.loadmat('datasets\\grupoDados2.mat')
 
X_train = mat['grupoTrain']
X_test = mat['grupoTest']
y_train = mat['trainRots']
y_test = mat['testRots']

predicted = knn(X_train, y_train, X_test, 1)
print(f'A acuracia passando todas as variáveis e k = 1 é {accuracy(predicted, y_test)}')

X_train = padronization(X_train)
X_test = padronization(X_test)

for x in range(1, 10):
    predicted = knn(X_train, y_train, X_test, x)
    print(f'A acuracia passando todas as variáveis normalizadas e k = {x} é {accuracy(predicted, y_test)}')

'''
Resultados:

A classificação utilizando k = 1 com os dados sem tratamento levou a uma acurácia de 68%. 

Para tratamento dos dados, foi realizado um tratamento de padronização pois as features não possuiam
uma escala muito diferente uma da outra.

Após o tratamento, chegamos a 96% com k igual a 1.

Com outros k's, foi possível chegar até 98%.

'''