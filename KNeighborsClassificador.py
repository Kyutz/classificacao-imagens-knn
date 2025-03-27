import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

def carregar_mnist():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    caracteristicas, rotulos = mnist.data.astype(np.float32), mnist.target.astype(int)
    return train_test_split(caracteristicas, rotulos, test_size=0.1, random_state=42)

def classificar_amostra(teste_amostra, treinar_caract, treinar_rotulos, vizinhos):
    distancia = np.linalg.norm(treinar_caract - teste_amostra, axis=1)
    k_vizinhos = np.argsort(distancia)[:vizinhos]
    k_rotulos = treinar_rotulos[k_vizinhos]
    return np.bincount(k_rotulos).argmax()

def knn_classify(treinar_caract, treinar_rotulos, teste_caract, vizinhos=3, n_jobs=-1):
    return np.array(Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(classificar_amostra)(teste_amostra, treinar_caract, treinar_rotulos, vizinhos) for teste_amostra in teste_caract
    ))

treinar_caract, teste_caract, treinar_rotulos, teste_rotulos = carregar_mnist()

vizinhos = 3
previsao = knn_classify(treinar_caract, treinar_rotulos, teste_caract[:100], vizinhos)

precisao = np.mean(previsao == teste_rotulos[:100])
print(f'Precisão das classificações: {precisao * 100:.2f}%')

indices_aleatorios = np.random.choice(100, 5, replace=False)

fig, axes = plt.subplots(1, 5, figsize=(10, 3))
for i, ax in enumerate(axes):
    ax.imshow(teste_caract[indices_aleatorios[i]].reshape(28, 28), cmap='gray')
    ax.set_title(f'Previsto: {previsao[indices_aleatorios[i]]}')
    ax.axis('off')

plt.savefig('mnistcomknn.png')

print(f'Classificações salvas no arquivo mnistcomknn.png')