import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle

# 1. Carregar o Iris Dataset
iris = load_iris()
X = iris.data
y = iris.target

# 2. Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Treinar o modelo KNN
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# 4. Avaliar o modelo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo: {accuracy:.2f}")

# 5. Salvar o modelo
nome_arquivo = "iris_model.pkl"
try:
    # Salva o modelo na pasta atual
    with open(nome_arquivo, "wb") as arquivo:
        pickle.dump(model, arquivo)

    # Exibe o caminho completo do arquivo salvo
    caminho_completo = os.path.abspath(nome_arquivo)
    print(f"Modelo salvo com sucesso em: {caminho_completo}")
except Exception as e:
    print(f"Erro ao salvar o modelo: {e}")