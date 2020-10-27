import numpy as np
import matplotlib.pyplot as plt

ta = 0.01
numIteracoes = 20  # loops
q = 13  # numero de entradas
m = 2  # numeros de neuronios na camada de entrada
n = 4  # numero de neuronios na camada oculta
l = 1  # numero de neuronios na camada de saida
W1 = np.random.random((n, m + 1))
W2 = np.random.random((l, n + 1))
# armazenamento de erros
# Erro de saidas
E = np.zeros(q)
# Erro total medio
Etm = np.zeros(numIteracoes)
bias = 1


def main():
    stop = 0
    while stop == 0:
        treinamento()
        stop = int(input('A rede está bem treinada? 1-SIM | 0-NÃO\n'))
    entradas()


def treinamento():
    global ta, numIteracoes, q, m, n, l, W1, W2, E, Etm, bias
    peso = np.array([130, 137, 120, 139, 127, 150, 131, 142, 129, 140, 134, 136, 128])
    ph = np.array([3, 3.9, 2.9, 4.1, 3.4, 4.0, 2.8, 4.2, 3.1, 3.8, 2.7, 4.6, 2.2])
    s = np.array([-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1])
    Error_Test = np.zeros(q)
    # -1 = maça
    # 1 = laranja
    # entrada do Percepton
    X = np.stack((peso, ph))  # concatena os paramentros em uma matriz
    for i in range(0, numIteracoes):
        for j in range(q):
            # bias no vetor de entrada
            Xb = np.hstack((bias, X[:, j]))

            # saida camada oculta
            # tangentehiperbolica(peso * xb)
            Y = np.tanh(W1.dot(Xb))

            # incluindo bias
            Yb = np.insert(Y, 0, bias)

            Z = np.tanh(W2.dot(Yb))

            # erro saidas
            e = s[j] - Z
            Error_Test[j] = s[j] - Z

            # erro total
            E[j] = (e.transpose().dot(e)) / 2

            # iterações e erro total
            # print('i = 'str(i) + ' E = ' + str(E))

            # backpropagation
            # Cálculo do gradiente na camada de saída
            delta2 = np.diag(e).dot((1 - Z * Z))
            vdelta2 = (W2.transpose()).dot(delta2)
            delta1 = np.diag(1 - Yb * Yb).dot(vdelta2)

            # Atualização dos pesos
            W1 = W1 + ta * (np.outer(delta1[1:], Xb))
            W2 = W2 + ta * (np.outer(delta2, Yb))

        Etm[i] = E.mean()

    # print("Erro total medio = " + str(Etm))

    print(Error_Test)
    print(np.round(Error_Test) - s)
    plt.plot(Etm)
    plt.show()


def entradas():
    global ta, numIteracoes, q, m, n, l, W1, W2, E, Etm, bias
    peso = np.array([122, 126, 143, 147])
    ph = np.array([2.3, 2.4, 4.4, 4.5])
    s = np.array([-1, -1, 1, 1])
    X = np.stack((peso, ph))  # concatena os paramentros em uma matriz

    Error_Test = np.zeros(4)
    for j in range(4):
        # bias no vetor de entrada
        Xb = np.hstack((bias, X[:, j]))

        # saida camada oculta
        # tangentehiperbolica(peso * xb)
        Y = np.tanh(W1.dot(Xb))

        # incluindo bias
        Yb = np.insert(Y, 0, bias)

        Z = np.tanh(W2.dot(Yb))

        # erro saidas
        Error_Test[j] = s[j] - Z
    print('\nEntradas:')
    for i in range(0, len(peso)):
        print('Peso: {p} | PH: {ph}\n'.format(p=peso[i], ph=ph[i]))
    print('Saída esperada: Maça, Maça, Laranja, Laranja')
    if -1 in np.round(Error_Test) - s:
        print('A saída está errada!')
    else:
        print('A saída está correta!')

    print(Error_Test)
    print(np.round(Error_Test) - s)


main()
