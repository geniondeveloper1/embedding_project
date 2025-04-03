from sentence_transformers import SentenceTransformer
import pickle
import numpy as np
import os

# Inicializar modelo
model = SentenceTransformer("all-MiniLM-L6-v2")

# Cargar embeddings de los fragmentos de los documentos
EMBEDDINGS_FILE = "embeddings/documentos_embeddings.pkl"

with open(EMBEDDINGS_FILE, "rb") as f:
    document_embeddings = pickle.load(f)

# Solicitar una pregunta al usuario
user_question = input("🤔 Ingresa tu pregunta: ")

# Generar embedding de la pregunta
question_embedding = model.encode(user_question)

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

# Mostrar la respuesta o un mensaje si no hay coincidencias relevantes
if best_score < THRESHOLD:
    print("\n❌ No encontré información relevante para tu pregunta. Intenta reformularla.")
else:
    print("\n🔍 Mejor coincidencia encontrada:")
    print(f"📄 Documento relevante: {os.path.basename(best_document)}")
    print(f"💡 Similitud: {best_score:.2f}")
    print("\n📜 Respuesta relevante:")
    print(best_match)
