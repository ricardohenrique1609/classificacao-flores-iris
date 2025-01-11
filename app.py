import streamlit as st
import pickle
import pandas as pd
from sklearn.datasets import load_iris

# Carrega o modelo
try:
    with open("iris_model.pkl", "rb") as arquivo:
        model = pickle.load(arquivo)
except FileNotFoundError:
    st.error(
        "Erro: Arquivo do modelo não encontrado. Execute o 'model.py' primeiro."
    )
    st.stop()
except Exception as e:
    st.error(f"Erro ao carregar o modelo: {e}")
    st.stop()

# Carrega as informações do dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = [iris.target_names[target] for target in iris.target]

# --- Interface ---

# Barra lateral
st.sidebar.title("Classificação de Flores")
st.sidebar.write("Iris Dataset")

# Opções na barra lateral
page = st.sidebar.radio("Navegação", ["Previsão", "Sobre"])

# Página de Previsão
if page == "Previsão":
    st.header("Previsão da Espécie da Flor")

    # Entradas do usuário com layout em colunas
    col1, col2 = st.columns(2)
    with col1:
        sepal_length = st.number_input(
            "Comprimento da Sépala (cm)", min_value=0.0, max_value=10.0, value=5.0
        )
        petal_length = st.number_input(
            "Comprimento da Pétala (cm)", min_value=0.0, max_value=10.0, value=1.5
        )
    with col2:
        sepal_width = st.number_input(
            "Largura da Sépala (cm)", min_value=0.0, max_value=10.0, value=3.0
        )
        petal_width = st.number_input(
            "Largura da Pétala (cm)", min_value=0.0, max_value=10.0, value=0.2
        )

    # Botão para classificar
    if st.button("Classificar"):
        # Faz a previsão
        input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
        prediction = model.predict(input_data)
        species = iris.target_names[prediction[0]]

        # Exibe o resultado com formatação Markdown
        st.markdown(f"**Espécie prevista:** {species}")


# Página Sobre
if page == "Sobre":
    st.header("Sobre o Projeto")
    st.write(
        """
        Este projeto demonstra a aplicação de Machine Learning para a 
        classificação de flores Iris. O modelo foi treinado com o 
        algoritmo KNN e permite prever a espécie da flor com base nas 
        características da sépala e da pétala.
        """
    )

    # Exibe a tabela de dados
    st.subheader("Dados do Iris Dataset")
    st.dataframe(df)

    # Adicione mais informações sobre o projeto, como autor, data, etc.
    st.write("---")
    st.write("Desenvolvido por Ricardo Henrique")