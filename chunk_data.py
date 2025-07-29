from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import json
import os

# Lecture du fichier texte (extrait nettoyé de ton PDF)
with open("data/texte_nettoye.txt", "r", encoding="utf-8") as f:
    texte = f.read()

# Initialisation du chunker
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,
    separators=["\n\n", "\n", ".", " ", ""]
)

# Création des documents (chunks)
documents = splitter.create_documents([texte])

# Affichage des premiers chunks
for i, doc in enumerate(documents[:5]):
    print(f"\n--- Chunk {i+1} ---\n{doc.page_content}")

# Création du dossier de sortie s'il n'existe pas
os.makedirs("data", exist_ok=True)

# Structuration des données en JSON
chunk_data = [
    {
        "page_content": doc.page_content,
        "metadata": {"chunk_id": i + 1, "source": "texte_nettoye.txt"}
    }
    for i, doc in enumerate(documents)
]

# Sauvegarde dans un fichier JSON
with open("data/chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunk_data, f, ensure_ascii=False, indent=2)

print(f"\n✅ {len(chunk_data)} chunks sauvegardés dans data/chunks.json")
