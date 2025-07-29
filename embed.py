from sentence_transformers import SentenceTransformer
import json

# 📥 Charger les chunks
with open("data/chunks.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 🧠 Utiliser un modèle multilingue compatible français
model = SentenceTransformer("intfloat/multilingual-e5-base")  # ou "intfloat/multilingual-e5-base" si tu veux comparer

# 🧠 Encoder les textes
for item in data:
    embedding = model.encode(item["page_content"], normalize_embeddings=True)  # très recommandé
    item["embedding"] = embedding.tolist()

# 💾 Sauvegarde des embeddings
with open("data/chunks_with_embeddings.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("✅ Embeddings multilingues calculés et sauvegardés.")
