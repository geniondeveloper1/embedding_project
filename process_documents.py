from sentence_transformers import SentenceTransformer
import pickle
import os

# Inicializar modelo
model = SentenceTransformer("all-MiniLM-L6-v2")

# Carpeta donde están los documentos
DOCUMENTS_FOLDER = "data"
EMBEDDINGS_FILE = "embeddings/documentos_embeddings.pkl"

# Función para dividir texto en fragmentos de un tamaño determinado
def split_text(text, chunk_size=50):
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
        chunk_embeddings = [model.encode(chunk).tolist() for chunk in chunks]
        
        # Guardar fragmentos y sus embeddings
        document_embeddings[file_path] = list(zip(chunks, chunk_embeddings))

# Guardar embeddings en un archivo
with open(EMBEDDINGS_FILE, "wb") as f:
    pickle.dump(document_embeddings, f)

print("✅ Embeddings generados por fragmentos y guardados correctamente.")
