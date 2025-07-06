import streamlit as st
import pandas as pd
from collections import Counter
import joblib  # para carregar o modelo salvo

# Carregar modelo treinado (pkl gerado no Colab)
modelo = joblib.load("classificador_mensagens.pkl")

st.title("Classificador de Mensagens do WhatsApp")

# Upload do arquivo
arquivo = st.file_uploader("Envie seu arquivo .csv com uma coluna chamada 'mensagem':", type="csv")

df = None  # iniciar a variável fora do bloco

if arquivo is not None:
    try:
        df = pd.read_csv(arquivo, encoding="utf-8")
    except Exception as e_utf8:
        try:
            df = pd.read_csv(arquivo, encoding="latin1")
        except Exception as e_latin:
            st.error("Erro ao ler o arquivo CSV. Verifique se ele está no formato correto.")
            st.text(f"Erro UTF-8: {e_utf8}")
            st.text(f"Erro Latin-1: {e_latin}")
            df = None

if df is not None:
    st.write("Colunas detectadas:", df.columns.tolist())

    df.columns = df.columns.str.strip().str.lower()

    if "mensagem" not in df.columns:
        st.error("❌ O arquivo precisa ter uma coluna chamada 'mensagem'.")
    else:
        mensagens = df["mensagem"]
        resultados = modelo.predict(mensagens)
        df["classificacao"] = resultados

        contagem = Counter(resultados)
        st.subheader("Resumo das Classificações:")
        for categoria, quantidade in contagem.items():
            st.write(f"{categoria.capitalize()}: {quantidade}")

        # Botão para baixar o CSV com resultados
        csv_resultado = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Baixar resultados como CSV",
            data=csv_resultado,
            file_name="mensagens_classificadas.csv",
            mime="text/csv"
        )
