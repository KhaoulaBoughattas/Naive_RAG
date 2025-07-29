from typing import List, Dict, Any
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import subprocess


def retrieve_and_rerank(
    query: str,
    collection_name: str = "psybot-embedding",
    qdrant_url: str = "localhost",
    qdrant_port: int = 6333,
    top_k: int = 5,
    search_k: int = 20,
) -> List[Dict[str, Any]]:
    """
    Recherche hybride dense + BM25 puis rerank avec CrossEncoder.
    """
    client = QdrantClient(host=qdrant_url, port=qdrant_port)

    embedding_model = SentenceTransformer("intfloat/multilingual-e5-base")
    reranker = CrossEncoder("BAAI/bge-reranker-base")

    query_embedding = embedding_model.encode([query], normalize_embeddings=True)[0]

    search_result = client.query_points(
        collection_name=collection_name,
        query=query_embedding.tolist(),
        limit=search_k,
        with_payload=True,
    )

    docs = [item.payload['page_content'] for item in search_result.points]
    dense_scores = [item.score for item in search_result.points]

    if not docs:
        print("⚠️ Aucun document trouvé.")
        return []

    tokenized_docs = [doc.split() for doc in docs]
    bm25 = BM25Okapi(tokenized_docs)
    bm25_scores = bm25.get_scores(query.split())

    alpha, beta = 0.6, 0.4
    combined_scores = [alpha * d + beta * b for d, b in zip(dense_scores, bm25_scores)]

    top_candidates_idx = sorted(
        range(len(combined_scores)),
        key=lambda i: combined_scores[i],
        reverse=True
    )[:2 * top_k]

    pairs = [(query, docs[i]) for i in top_candidates_idx]
    rerank_scores = reranker.predict(pairs)

    reranked_idx = sorted(
        range(len(rerank_scores)),
        key=lambda i: rerank_scores[i],
        reverse=True
    )[:top_k]

    return [
        {"score": rerank_scores[i], "page_content": docs[top_candidates_idx[i]]}
        for i in reranked_idx
    ]


def generate(prompt: str, documents: List[Dict[str, Any]], model='gemma3:1b') -> str:
    """
    Génère une réponse courte et bienveillante basée sur les extraits.
    """
    # Limiter à 3 documents courts (max 400 caractères chacun)
    short_docs = [
        d['page_content'][:400].replace("\n", " ").strip() + "..."
        for d in documents[:3]
    ]
    context = " ".join(short_docs)

    full_prompt = f"""Voici des extraits : {context}

Question : {prompt}

Donne une réponse courte, claire et bienveillante basée sur ces extraits, comme le ferait un psychologue ou un assistant de santé mentale.
"""

    try:
        result = subprocess.run(
            ['ollama', 'run', model],
            input=full_prompt,
            text=True,
            capture_output=True,
            encoding='utf-8'
        )
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return f"Erreur lors de la génération : {result.stderr.strip()}"
    except Exception as e:
        return f"Exception lors de la génération : {e}"


def main():
    print("🔎 Recherche avec reranking CrossEncoder")
    query = input("❓ Votre question : ").strip()
    if not query:
        print("⚡ Question vide.")
        return

    results = retrieve_and_rerank(query)
    if not results:
        print("❌ Aucun résultat trouvé.")
        return

    response = generate(query, results)
    print("\n🔹 Réponse générée 🔹")
    print(response)


if __name__ == "__main__":
    main()
