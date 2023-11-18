# Importando as bibliotecas necessárias
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import pinv

# Função para calcular a fronteira de decisão do classificador MSE
def decision_boundary_mse(weights, x):
    return -(weights[2] + weights[0] * x) / weights[1]

# Função para calcular a fronteira de decisão do classificador de Bayes
def decision_boundary_bayes(mean_diff, x):
    return (mean_diff / 2) - x * (mean_diff / 8)

# Parâmetros para as distribuições Gaussianas
mean_C1 = [0, 0]
cov_C1 = [[1, 0], [0, 1]]
mean_C2 = [4, 4]
cov_C2 = [[1, 0], [0, 1]]

# 7) Análise com 500 eventos de cada classe
n_samples = 500
C1_train_500 = np.random.multivariate_normal(mean_C1, cov_C1, n_samples)
C2_train_500 = np.random.multivariate_normal(mean_C2, cov_C2, n_samples)
C1_test_500 = np.random.multivariate_normal(mean_C1, cov_C1, n_samples)
C2_test_500 = np.random.multivariate_normal(mean_C2, cov_C2, n_samples)

# Combinar os dados de treino e rótulos para treinamento com 500 eventos
all_data_train_500 = np.vstack((C1_train_500, C2_train_500))
all_labels_train_500 = np.vstack((-np.ones((n_samples, 1)), np.ones((n_samples, 1))))
all_data_bias_train_500 = np.hstack((all_data_train_500, np.ones((2 * n_samples, 1))))

# Projetar o classificador MSE usando pseudo-inversa com 500 eventos
W_pseudo_500 = pinv(all_data_bias_train_500).dot(all_labels_train_500)

# 8) Análise com 50 eventos de cada classe
n_samples_train = 50
C1_train_50 = np.random.multivariate_normal(mean_C1, cov_C1, n_samples_train)
C2_train_50 = np.random.multivariate_normal(mean_C2, cov_C2, n_samples_train)
# Geração de conjuntos de teste com 50 eventos para cada classe
C1_test_50 = np.random.multivariate_normal(mean_C1, cov_C1, n_samples_train)
C2_test_50 = np.random.multivariate_normal(mean_C2, cov_C2, n_samples_train)

# Combinar os dados de treino e rótulos para treinamento com 50 eventos
all_data_train_50 = np.vstack((C1_train_50, C2_train_50))
all_labels_train_50 = np.vstack((-np.ones((n_samples_train, 1)), np.ones((n_samples_train, 1))))
all_data_bias_train_50 = np.hstack((all_data_train_50, np.ones((2 * n_samples_train, 1))))

# Projetar o classificador MSE usando pseudo-inversa com 50 eventos
W_pseudo_50 = pinv(all_data_bias_train_50).dot(all_labels_train_50)

# Gerar valores para o eixo x para plotar as fronteiras de decisão
x_values = np.linspace(-3, 7, 300)

# Fronteiras de decisão para os classificadores com 500 e 50 eventos
y_values_mse_500 = decision_boundary_mse(W_pseudo_500, x_values)
y_values_mse_50 = decision_boundary_mse(W_pseudo_50, x_values)

# A fronteira de Bayes permanece a mesma, pois não depende do tamanho do conjunto de treinamento
y_values_bayes = 4 - x_values

# Plot para análise com 500 eventos
plt.figure()
plt.scatter(C1_test_500[:, 0], C1_test_500[:, 1], color='red', label='Class C1 (500 events)')
plt.scatter(C2_test_500[:, 0], C2_test_500[:, 1], color='blue', label='Class C2 (500 events)')
plt.plot(x_values, y_values_mse_500, color='green', label='MSE Decision Boundary (500 events)')
plt.plot(x_values, y_values_bayes, '--', color='purple', label='Bayes Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('7) MSE and Bayes Classifiers with 500 Training Events')
plt.legend()
plt.grid(True)

# Plot para análise com 50 eventos
plt.figure()
plt.scatter(C1_test_50[:, 0], C1_test_50[:, 1], color='red', label='Class C1 (50 events)')
plt.scatter(C2_test_50[:, 0], C2_test_50[:, 1], color='blue', label='Class C2 (50 events)')
plt.plot(x_values, y_values_mse_50, color='green', label='MSE Decision Boundary (50 events)')
plt.plot(x_values, y_values_bayes, '--', color='purple', label='Bayes Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('8) MSE and Bayes Classifiers with 50 Training Events')
plt.legend()
plt.grid(True)

# 9) Análise com outros 50 eventos de cada classe para treinamento
C1_train_new_50 = np.random.multivariate_normal(mean_C1, cov_C1, n_samples_train)
C2_train_new_50 = np.random.multivariate_normal(mean_C2, cov_C2, n_samples_train)

# Combinar os novos dados de treino e rótulos
all_data_train_new_50 = np.vstack((C1_train_new_50, C2_train_new_50))
all_labels_train_new_50 = np.vstack((-np.ones((n_samples_train, 1)), np.ones((n_samples_train, 1))))
all_data_bias_train_new_50 = np.hstack((all_data_train_new_50, np.ones((2 * n_samples_train, 1))))

# Projetar o classificador MSE usando pseudo-inversa com os novos 50 eventos
W_pseudo_new_50 = pinv(all_data_bias_train_new_50).dot(all_labels_train_new_50)

# Calcular a fronteira de decisão para o novo classificador
y_values_mse_new_50 = decision_boundary_mse(W_pseudo_new_50, x_values)

# Plot para análise com os novos 50 eventos
plt.figure()
plt.scatter(C1_test_50[:, 0], C1_test_50[:, 1], color='red', label='Class C1 (50 events)')
plt.scatter(C2_test_50[:, 0], C2_test_50[:, 1], color='blue', label='Class C2 (50 events)')
plt.plot(x_values, y_values_mse_new_50, color='green', label='MSE Decision Boundary (New 50 events)')
plt.plot(x_values, y_values_bayes, '--', color='purple', label='Bayes Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('9) MSE and Bayes Classifiers with New 50 Training Events')
plt.legend()
plt.grid(True)

# Exibir o gráfico
plt.tight_layout()
plt.show()