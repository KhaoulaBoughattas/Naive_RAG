# get_vector_db.py
# -----------------------------------------------------------
# Charge les embeddings stockés dans data/chunks_with_embeddings.json
# puis les insère dans Qdrant (en lots) après avoir recréé la collection.
# -----------------------------------------------------------

import json
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# -----------------------------------------------------------
# 1️⃣ Charger les données avec embeddings
# -----------------------------------------------------------
input_file = "data/chunks_with_embeddings.json"
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

if not data:
    raise ValueError("❌ Aucun embedding trouvé dans le fichier JSON.")

# Dimension automatique
VECTOR_DIM = len(data[0]["embedding"])
print(f"📏 Dimension détectée des vecteurs : {VECTOR_DIM}")

# -----------------------------------------------------------
# 2️⃣ Connexion Qdrant
# -----------------------------------------------------------
client = QdrantClient(host="localhost", port=6333)

# -----------------------------------------------------------
# 3️⃣ Création (ou recréation) de la collection
# -----------------------------------------------------------
COLLECTION_NAME = "psybot-embedding"

if client.collection_exists(COLLECTION_NAME):
    print(f"⚡ La collection {COLLECTION_NAME} existe déjà. Suppression…")
    client.delete_collection(COLLECTION_NAME)

print(f"🟣 Création de la collection {COLLECTION_NAME}")
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE)
)

# -----------------------------------------------------------
# 4️⃣ Préparation des points
# -----------------------------------------------------------
points = [
    PointStruct(
        id=idx,
        vector=item["embedding"],
        payload={"page_content": item["page_content"]}
    )
    for idx, item in enumerate(data)
]
print(f"🔢 Total points à insérer : {len(points)}")

# -----------------------------------------------------------
# 5️⃣ Insertion par lots pour éviter 'Payload too large'
# -----------------------------------------------------------
def batch_insert(client, collection_name, all_points, batch_size=200):
    total = len(all_points)
    for start in range(0, total, batch_size):
        batch = all_points[start : start + batch_size]
        end = start + len(batch) - 1
        print(f"➡ Inserting points {start} to {end}")
        client.upsert(collection_name=collection_name, points=batch)

batch_insert(client, COLLECTION_NAME, points)

print("✅ Qdrant est prêt et opérationnel.")
