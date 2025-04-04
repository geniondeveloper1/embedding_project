import requests
import pickle
import numpy as np
import os
import re

# URL del servidor LM Studio local
LMSTUDIO_URL = "http://127.0.0.1:1234/v1/embeddings"

# Cargar embeddings de los fragmentos de los documentos
EMBEDDINGS_FILE = "embeddings/documentos_embeddings.pkl"

with open(EMBEDDINGS_FILE, "rb") as f:
    document_embeddings = pickle.load(f)

# Función para obtener embeddings desde LM Studio
def get_embedding(text):
    response = requests.post(
        LMSTUDIO_URL,
        json={"input": text, "model": "text-embedding-bge-m3"}
    )
    if response.status_code == 200:
        return response.json()["data"][0]["embedding"]
    else:
        print(f"❌ Error al obtener embedding: {response.text}")
        return None

# Solicitar una pregunta al usuario
user_question = input("🤔 Ingresa tu pregunta: ")

# Generar embedding de la pregunta
question_embedding = get_embedding(user_question)
if question_embedding is None:
    print("❌ No se pudo generar el embedding de la pregunta.")
    exit()

# Función para calcular similitud coseno
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Buscar el fragmento más relevante en todos los documentos
best_match = None
best_score = -1
best_document = None
THRESHOLD = 0.3  # Umbral mínimo de similitud

for file_path, chunks in document_embeddings.items():
    for chunk_text, chunk_embedding in chunks:
        score = cosine_similarity(question_embedding, np.array(chunk_embedding))
        
        if score > best_score:
            best_score = score
            best_match = chunk_text
            best_document = file_path

# Si no hay coincidencias relevantes, salir
if best_score < THRESHOLD:
    print("\n❌ No encontré información relevante para tu pregunta. Intenta reformularla.")
    exit()

# 🔥 1. Dividir el chunk en oraciones
sentences = re.split(r'(?<=[.!?])\s+', best_match)

# 🔥 2. Calcular la similitud para cada oración
relevant_sentences = []
for sentence in sentences:
    sentence_embedding = get_embedding(sentence)
    if sentence_embedding:
        similarity = cosine_similarity(question_embedding, np.array(sentence_embedding))
        if similarity > 0.6:  # Umbral para filtrar oracihorones irrelevantes
            relevant_sentences.append(sentence)

# Mostrar el resultado final con solo lo relevante
if relevant_sentences:
    print("\n🔍 Respuesta relevante encontrada:")
    print(f"📄 Documento: {os.path.basename(best_document)}")
    for sentence in relevant_sentences:
        print(f"✅ {sentence}")
else:
    print("\n❌ No encontré información específica dentro del fragmento.")