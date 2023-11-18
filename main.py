import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import pinv
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB

# Parâmetros para as distribuições Gaussianas
mean_C1 = [0, 0]
cov_C1 = [[1, 0], [0, 1]]
mean_C2 = [4, 4]
cov_C2 = [[1, 0], [0, 1]]

# Número de amostras por classe para treino e teste
n_samples = 500

# Gerar amostras de treino e teste para C1 e C2
C1_train = np.random.multivariate_normal(mean_C1, cov_C1, n_samples)
C2_train = np.random.multivariate_normal(mean_C2, cov_C2, n_samples)
C1_test = np.random.multivariate_normal(mean_C1, cov_C1, n_samples)
C2_test = np.random.multivariate_normal(mean_C2, cov_C2, n_samples)

# Salvar os dados de treino e teste em um arquivo npz
np.savez('dados_trab_2.npz', C1_train=C1_train, C2_train=C2_train, C1_test=C1_test, C2_test=C2_test)

# Carregar os dados do arquivo npz
npzfile = np.load('dados_trab_2.npz')
C1_train = npzfile['C1_train']
C2_train = npzfile['C2_train']
C1_test = npzfile['C1_test']
C2_test = npzfile['C2_test']

## 1. Preparar os Dados: Adicionando uma coluna de uns (para bias)
C1_train_bias = np.hstack((C1_train, np.ones((C1_train.shape[0], 1))))
C2_train_bias = np.hstack((C2_train, np.ones((C2_train.shape[0], 1))))

# 2. Etiqueta das Classes
labels_C1 = -np.ones((C1_train_bias.shape[0], 1))  # Classe C1 com etiqueta -1
labels_C2 = np.ones((C2_train_bias.shape[0], 1))   # Classe C2 com etiqueta +1

# 3. Combinar Dados de Treino
X_train = np.vstack((C1_train_bias, C2_train_bias))
y_train = np.vstack((labels_C1, labels_C2))

# 4. Calcular a Pseudo-Inversa e encontrar pesos
pseudo_inv = np.linalg.pinv(X_train)
weights = pseudo_inv.dot(y_train)

# 5. Projeção do Classificador e Avaliação
# Define a função para plotar o hiperplano
def plot_hyperplane(X, weights, label='Decision Boundary'):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = -(weights[2] + weights[0]*x_min) / weights[1], -(weights[2] + weights[0]*x_max) / weights[1]
    plt.plot([x_min, x_max], [y_min, y_max], label=label)

# Plotar dados de teste e a reta de separação
plt.scatter(C1_test[:, 0], C1_test[:, 1], color='red', marker='o', label='C1 Test')
plt.scatter(C2_test[:, 0], C2_test[:, 1], color='blue', marker='x', label='C2 Test')
plot_hyperplane(np.vstack((C1_test, C2_test)), weights)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid()
plt.show()