import requests
import pickle
import os

# URL del servidor LM Studio para generar embeddings
LMSTUDIO_EMBEDDING_URL = "http://192.168.0.20:1234/v1/embeddings"

# Carpeta donde están los documentos
DOCUMENTS_FOLDER = "data"
EMBEDDINGS_FILE = "embeddings/documentos_embeddings.pkl"

# Función para obtener embeddings desde LM Studio
def get_embedding(text):
    response = requests.post(
        LMSTUDIO_EMBEDDING_URL,
        json={"input": text, "model": "text-embedding-bge-m3"}  # Asegura que este modelo está cargado en LM Studio
    )
    if response.status_code == 200:
        return response.json()["data"][0]["embedding"]
    else:
        print(f"❌ Error al obtener embedding: {response.text}")
        return None

# Función para dividir texto en fragmentos de un tamaño determinado
def split_text(text, chunk_size=100):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Verificar que la carpeta de documentos existe
if not os.path.exists(DOCUMENTS_FOLDER):
    print(f"❌ La carpeta '{DOCUMENTS_FOLDER}' no existe. Crea la carpeta y agrega archivos TXT.")
    exit()

# Leer y procesar los documentos
document_embeddings = {}

for filename in os.listdir(DOCUMENTS_FOLDER):
    if filename.endswith(".txt"):
        file_path = os.path.join(DOCUMENTS_FOLDER, filename)
        
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        
        # Dividir documento en fragmentos
        chunks = split_text(content)
        
        # Generar embeddings para cada fragmento
        chunk_embeddings = []
        for chunk in chunks:
            embedding = get_embedding(chunk)
            if embedding:
                chunk_embeddings.append((chunk, embedding))
        
        # Guardar fragmentos y sus embeddings
        document_embeddings[file_path] = chunk_embeddings

# Guardar embeddings en un archivo
with open(EMBEDDINGS_FILE, "wb") as f:
    pickle.dump(document_embeddings, f)

print("✅ Embeddings generados por fragmentos y guardados correctamente.")
