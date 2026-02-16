import os
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

# 🔑 Coloque sua chave da OpenAI aqui
client = OpenAI(api_key="SUA_CHAVE_AQUI")

# Função para gerar embeddings
def gerar_embedding(texto):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texto
    )
    return response.data[0].embedding

# Ler arquivo
def carregar_documento(caminho):
    with open(caminho, "r", encoding="utf-8") as file:
        return file.read()

# Dividir texto em trechos
def dividir_em_chunks(texto, tamanho=200):
    palavras = texto.split()
    chunks = []
    for i in range(0, len(palavras), tamanho):
        chunk = " ".join(palavras[i:i+tamanho])
        chunks.append(chunk)
    return chunks

# Carregar e preparar dados
texto = carregar_documento("inputs/exemplo.txt")
chunks = dividir_em_chunks(texto)

# Gerar embeddings dos chunks
embeddings_chunks = [gerar_embedding(chunk) for chunk in chunks]

print("Chatbot baseado em documentos iniciado!")
print("Digite 'sair' para encerrar.\n")

while True:
    pergunta = input("Você: ")

    if pergunta.lower() == "sair":
        break

    embedding_pergunta = gerar_embedding(pergunta)

    # Calcular similaridade
    similaridades = cosine_similarity(
        [embedding_pergunta],
        embeddings_chunks
    )[0]

    indice_mais_similar = np.argmax(similaridades)
    contexto = chunks[indice_mais_similar]

    # Gerar resposta baseada no contexto
    resposta = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Responda apenas com base no contexto fornecido."},
            {"role": "user", "content": f"Contexto: {contexto}\n\nPergunta: {pergunta}"}
        ]
    )

    print("\nChatbot:", resposta.choices[0].message.content)
    print()
